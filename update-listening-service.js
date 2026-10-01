const fs = require('fs');
const path = require('path');

const listeningServiceDir = 'C:\\\\Users\\\\salun\\\\OneDrive - smarttech\\\\Documents\\\\D Drive backup\\\\creatosaurus-intership\\\\cache_backend\\\\listening-service';

// 1. src/types/listening.types.ts
const listeningTypesPath = path.join(listeningServiceDir, 'src', 'types', 'listening.types.ts');
const listeningTypesContent = `import { type Document } from 'mongoose';

export interface IEmotionDetail {
  label: string;
  score: number;
}

export interface IListening extends Document {
  title: string;
  link: string;
  pubDate: Date;
  content: string;
  keywords: string;
  hash: string;
  createdAt: Date;
  updatedAt: Date;
  source: string;
  mainSource: string;
  sentimental: string;
  sentimentScore?: number;
  sentimentConfidence?: number;
  emotions?: IEmotionDetail[];
  urgency?: string;
  topics?: string[];
  nlpProcessed?: boolean;
  method: string;
  rawResponse?: any;
}
`;
fs.writeFileSync(listeningTypesPath, listeningTypesContent, 'utf8');
console.log('Updated src/types/listening.types.ts');

// 2. src/models/Listening.model.ts
const listeningModelPath = path.join(listeningServiceDir, 'src', 'models', 'Listening.model.ts');
const listeningModelContent = `import mongoose, { Schema } from 'mongoose';
import { type IListening } from '../types/index.ts';

const listeningSchema = new Schema<IListening>({
  title: String,
  link: String,
  pubDate: Date,
  content: String,
  keywords: String,
  hash: { type: String, unique: true },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
  source: String,
  mainSource: String,
  sentimental: String,
  sentimentScore: Number,
  sentimentConfidence: Number,
  emotions: [
    {
      label: String,
      score: Number,
    },
  ],
  urgency: String,
  topics: [String],
  nlpProcessed: { type: Boolean, default: false },
  method: String,
  rawResponse: { type: Schema.Types.Mixed },
});

listeningSchema.index({ title: 'text', content: 'text' });
listeningSchema.index({ pubDate: -1 });
listeningSchema.index({ keywords: 1, pubDate: -1 });
listeningSchema.index({ keywords: 1 });
listeningSchema.index({ keywords: 1, source: 1 });
listeningSchema.index({ sentimental: 1 });
listeningSchema.index({ urgency: 1 });
listeningSchema.index({ topics: 1 });

export const Listening = mongoose.model<IListening>('Listening', listeningSchema);
`;
fs.writeFileSync(listeningModelPath, listeningModelContent, 'utf8');
console.log('Updated src/models/Listening.model.ts');

// 3. src/services/nlp.service.ts
const nlpServicePath = path.join(listeningServiceDir, 'src', 'services', 'nlp.service.ts');
const nlpServiceContent = `// @ts-nocheck
import axios from 'axios';
import Sentiment from 'sentiment';
import { logger } from '../config/logger.ts';

const sentimentAnalyzer = new Sentiment();

const NLP_SERVICE_URL = process.env.NLP_SERVICE_URL || 'http://localhost:8000';

export interface INLPAnalysisResult {
  sentimental: string;
  sentimentScore: number;
  sentimentConfidence: number;
  emotions: Array<{ label: string; score: number }>;
  urgency: string;
  topics: string[];
  keywords: string[];
  nlpProcessed: boolean;
}

/**
 * Local fast fallback if Render microservice is starting or unreachable
 */
export const fallbackLocalAnalysis = (title = '', content = ''): INLPAnalysisResult => {
  const text = \`\${title || ''} \${content || ''}\`.trim();
  if (!text) {
    return {
      sentimental: 'Neutral',
      sentimentScore: 0,
      sentimentConfidence: 0.5,
      emotions: [{ label: 'neutral', score: 1.0 }],
      urgency: 'low',
      topics: ['General'],
      keywords: [],
      nlpProcessed: false,
    };
  }

  const result = sentimentAnalyzer.analyze(text);
  let sentimental = 'Neutral';
  let score = 0;
  if (typeof result.score === 'number') {
    if (result.score > 0) {
      sentimental = 'Positive';
      score = Math.min(1.0, result.score / 5);
    } else if (result.score < 0) {
      sentimental = 'Negative';
      score = Math.max(-1.0, result.score / 5);
    }
  }

  const textLower = text.toLowerCase();
  let urgency = 'medium';
  if (textLower.includes('outage') || textLower.includes('hacked') || textLower.includes('scam') || textLower.includes('fraud')) {
    urgency = 'critical';
  } else if (textLower.includes('issue') || textLower.includes('broken') || textLower.includes('error') || sentimental === 'Negative') {
    urgency = 'high';
  } else if (sentimental === 'Positive') {
    urgency = 'low';
  }

  return {
    sentimental,
    sentimentScore: score,
    sentimentConfidence: 0.7,
    emotions: [
      { label: sentimental === 'Positive' ? 'joy' : sentimental === 'Negative' ? 'sadness' : 'neutral', score: 0.7 },
    ],
    urgency,
    topics: ['General'],
    keywords: [],
    nlpProcessed: false,
  };
};

/**
 * Analyze single item using Render NLP Service (with fallback)
 */
export const analyzeItemNLP = async (title = '', content = '', source = ''): Promise<INLPAnalysisResult> => {
  try {
    const response = await axios.post(
      \`\${NLP_SERVICE_URL}/analyze\`,
      { title, content, source },
      { timeout: 3500 }
    );
    return response.data;
  } catch (error) {
    logger.warn(\`[NLP Service] Remote call to \${NLP_SERVICE_URL} failed/timed out. Using fallback analysis. Error: \${error.message}\`);
    return fallbackLocalAnalysis(title, content);
  }
};

/**
 * Batch analyze array of items for ingestion speed
 */
export const analyzeBatchNLP = async (items: Array<{ title?: string; content?: string; source?: string }>): Promise<INLPAnalysisResult[]> => {
  if (!items || items.length === 0) return [];
  try {
    const response = await axios.post(
      \`\${NLP_SERVICE_URL}/analyze-batch\`,
      { items },
      { timeout: 7000 }
    );
    return response.data;
  } catch (error) {
    logger.warn(\`[NLP Service] Remote batch call failed/timed out. Using fallback local analysis for \${items.length} items.\`);
    return items.map((it) => fallbackLocalAnalysis(it.title || '', it.content || ''));
  }
};

/**
 * Enrich item object with NLP metrics
 */
export const enrichItemWithNLP = async (item: any): Promise<any> => {
  const nlp = await analyzeItemNLP(item.title || '', item.content || '', item.source || '');
  return {
    ...item,
    sentimental: item.sentimental || nlp.sentimental,
    sentimentScore: nlp.sentimentScore,
    sentimentConfidence: nlp.sentimentConfidence,
    emotions: nlp.emotions,
    urgency: nlp.urgency,
    topics: nlp.topics,
    nlpProcessed: nlp.nlpProcessed,
  };
};

/**
 * Enrich batch of items with NLP metrics
 */
export const enrichBatchWithNLP = async (items: any[]): Promise<any[]> => {
  if (!items || items.length === 0) return [];
  const payload = items.map((it) => ({
    title: it.title || '',
    content: it.content || '',
    source: it.source || '',
  }));

  const nlpResults = await analyzeBatchNLP(payload);

  return items.map((item, idx) => {
    const nlp = nlpResults[idx] || fallbackLocalAnalysis(item.title, item.content);
    return {
      ...item,
      sentimental: item.sentimental || nlp.sentimental,
      sentimentScore: nlp.sentimentScore,
      sentimentConfidence: nlp.sentimentConfidence,
      emotions: nlp.emotions,
      urgency: nlp.urgency,
      topics: nlp.topics,
      nlpProcessed: nlp.nlpProcessed,
    };
  });
};
`;
fs.writeFileSync(nlpServicePath, nlpServiceContent, 'utf8');
console.log('Created src/services/nlp.service.ts');

console.log('Listening service files updated successfully!');
