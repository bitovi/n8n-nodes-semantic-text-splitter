import { TextSplitter } from '@langchain/textsplitters';
import type { Embeddings } from '@langchain/core/embeddings';

type Embedding = number[];

export interface SemanticTextSplitterParams {
	breakpointThreshold: number;
	delimiters: string[];
	embeddings: Embeddings;
	minChunkSize: number;
	windowSize: number;
}

export class SemanticTextSplitter extends TextSplitter implements SemanticTextSplitterParams {
	breakpointThreshold: number;
	delimiters: string[];
	embeddings: Embeddings;
	minChunkSize: number;
	windowSize: number;

	constructor(fields: SemanticTextSplitterParams) {
		super();

		this.breakpointThreshold = fields.breakpointThreshold ?? 0.95;
		this.delimiters = fields.delimiters ?? ['.', '!', '?'];
		this.embeddings = fields.embeddings;
		this.minChunkSize = fields.minChunkSize ?? 100;
		this.windowSize = fields.windowSize ?? 3;
	}

	async splitText(text: string): Promise<string[]> {
		// 1. Split into initial sentences
		const sentences = this._splitIntoSentences(text);
		if (sentences.length <= this.windowSize) {
			return [text];
		}

		// 2. Create sliding windows of sentences
		const windows = this._createSlidingWindows(sentences);

		// 3. Get embeddings for each window
		const embeddings = await this._getEmbeddingsForWindows(windows);

		// 4. Calculate cosine distances between sequential windows
		const distances = this._calculateSequentialDistances(embeddings);

		// 5. Find breakpoints based on distance threshold
		const breakpoints = this._findBreakpoints(distances);

		// 6. Create final chunks based on breakpoints
		return this._createChunksFromBreakpoints(sentences, breakpoints);
	}

	_splitIntoSentences(text: string): string[] {
		// Escape special regex characters and join with |
		const escapedDelimiters = this.delimiters
			.map((delimiter) => delimiter.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
			.join('');

		// Create regex pattern using the delimiters
		const pattern = new RegExp(`[^${escapedDelimiters}]+[${escapedDelimiters}]+`, 'g');

		return (text.match(pattern) || [text])
			.map((sentence) => sentence.trim());
	}

	_createSlidingWindows(sentences: string[]): string[] {
		const windows = [];
		for (let i = 0; i <= sentences.length - this.windowSize; i++) {
			const window = sentences.slice(i, i + this.windowSize).join(' ');
			windows.push(window);
		}
		return windows;
	}

	async _getEmbeddingsForWindows(windows: string[]): Promise<Embedding[]> {
		// Get embeddings for each window using provided embedding model
		const embeddings = [];
		for (const window of windows) {
			const embedding = await this.embeddings.embedQuery(window);
			embeddings.push(embedding);
		}
		return embeddings;
	}

	_calculateSequentialDistances(embeddings: Embedding[]): number[] {
		const distances = [];
		for (let i = 0; i < embeddings.length - 1; i++) {
			const distance = 1 - this._cosineSimilarity(embeddings[i], embeddings[i + 1]);
			distances.push(distance);
		}
		return distances;
	}

	_cosineSimilarity(embedding1: Embedding, embedding2: Embedding): number {
		const dotProduct = embedding1.reduce((sum, a, i) => sum + a * embedding2[i], 0);
		const norm1 = Math.sqrt(embedding1.reduce((sum, a) => sum + a * a, 0));
		const norm2 = Math.sqrt(embedding2.reduce((sum, a) => sum + a * a, 0));
		return dotProduct / (norm1 * norm2);
	}

	_findBreakpoints(distances: number[]): number[] {
		// Find distance threshold at specified percentile
		const sortedDistances = [...distances].sort((a, b) => a - b);
		const thresholdIndex = Math.floor(distances.length * this.breakpointThreshold);
		const threshold = sortedDistances[thresholdIndex];

		// Find indices where distance exceeds threshold
		return distances.reduce<number[]>((breakpoints, distance, index) => {
			if (distance > threshold) {
				breakpoints.push(index + 1); // +1 because index refers to distance between windows
			}
			return breakpoints;
		}, []);
	}

	_createChunksFromBreakpoints(sentences: string[], breakpoints: number[]): string[] {
		const chunks = [];
		let startIndex = 0;

		// Create chunks based on breakpoints
		breakpoints.forEach((breakpoint) => {
			const chunk = sentences.slice(startIndex, breakpoint).join(' ');
			if (chunk.length >= this.minChunkSize) {
				chunks.push(chunk);
			}
			startIndex = breakpoint;
		});

		// Add final chunk
		const finalChunk = sentences.slice(startIndex).join(' ');
		if (finalChunk.length >= this.minChunkSize) {
			chunks.push(finalChunk);
		}

		return chunks;
	}
}
