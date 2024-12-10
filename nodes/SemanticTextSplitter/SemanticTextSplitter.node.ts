/* eslint-disable n8n-nodes-base/node-dirname-against-convention */
import {
	NodeConnectionType,
	type IExecuteFunctions,
	type INodeType,
	type INodeTypeDescription,
	type SupplyData,
} from 'n8n-workflow';
import { logWrapper } from '@n8n/n8n-nodes-langchain/dist/utils/logWrapper';
import { SemanticTextSplitter as SemanticTextSplitterClass } from './SemanticTextSplitter.class';
import { Embeddings } from '@langchain/core/embeddings';
import { getConnectionHintNoticeField } from '@n8n/n8n-nodes-langchain/dist/utils/sharedFields';

export class SemanticTextSplitter implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Semantic Text Splitter',
		name: 'semanticTextSplitter',
		icon: 'fa:grip-lines-vertical',
		group: ['transform'],
		version: 1,
		description: 'Split text into embedding walk based chunks',
		defaults: {
			name: 'Semantic Text Splitter',
		},
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Text Splitters'],
			},
			resources: {
				primaryDocumentation: [
					{
						url: 'https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb',
					},
				],
			},
		},
		// eslint-disable-next-line n8n-nodes-base/node-class-description-inputs-wrong-regular-node
		inputs: [
			NodeConnectionType.Main,
			{
				displayName: 'Embedding',
				maxConnections: 1,
				type: NodeConnectionType.AiEmbedding,
				required: true,
			},
		],
		// eslint-disable-next-line n8n-nodes-base/node-class-description-outputs-wrong
		outputs: [NodeConnectionType.AiTextSplitter],
		outputNames: ['Text Splitter'],
		properties: [
			// @ts-ignore
			getConnectionHintNoticeField(['ai_document']),
			{
				displayName: 'Percentile Threshold For Breakpoints',
				name: 'breakpointThreshold',
				type: 'number',
				default: 0.95,
				required: true,
			},
			{
				displayName: 'Number Of Sentences To Consider In Sliding Window',
				name: 'windowSize',
				type: 'number',
				default: 3,
				required: true,
			},
			{
				displayName: 'Minimum Chunk Size',
				name: 'minChunkSize',
				type: 'number',
				default: 100,
				required: true,
			},
			{
				displayName: 'Sentence Delimiters',
				name: 'delimiters',
				type: 'string',
				default: '.!?',
				required: true,
			},
		],
	};

	async supplyData(this: IExecuteFunctions, itemIndex: number): Promise<SupplyData> {
		this.logger.debug('Supply Data for Text Splitter');

		const splitter = new SemanticTextSplitterClass({
			breakpointThreshold: this.getNodeParameter('breakpointThreshold', itemIndex) as number,
			delimiters: [...(this.getNodeParameter('delimiters', itemIndex) as string)],
			embeddings: (await this.getInputConnectionData(
				NodeConnectionType.AiEmbedding,
				0,
			)) as Embeddings,
			minChunkSize: this.getNodeParameter('minChunkSize', itemIndex) as number,
			windowSize: this.getNodeParameter('windowSize', itemIndex) as number,
		});

		return {
			// @ts-ignore
			response: logWrapper(splitter, this),
		};
	}
}
