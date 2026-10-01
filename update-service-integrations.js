const fs = require('fs');
const path = require('path');

const listeningServiceDir = 'C:\\\\Users\\\\salun\\\\OneDrive - smarttech\\\\Documents\\\\D Drive backup\\\\creatosaurus-intership\\\\cache_backend\\\\listening-service';

// 1. Update src/services/listening.service.ts
const listeningServicePath = path.join(listeningServiceDir, 'src', 'services', 'listening.service.ts');
const listeningServiceContent = `// @ts-nocheck
import { logger } from '../config/logger.ts';
import { Listening } from '../models/Listening.model.ts';
import { handleKeywordMentionCount } from './keyword.service.ts';
import { stripHtml } from '../utils/sanitize.ts';
import { enrichItemWithNLP, enrichBatchWithNLP, fallbackLocalAnalysis } from './nlp.service.ts';

export const computeSentimentLabel = (title: string = '', content: string = '') => {
  const result = fallbackLocalAnalysis(title, content);
  return result.sentimental;
};

export const insertListeningData = async (data: any) => {
  try {
    if (data.title) data.title = stripHtml(data.title);
    if (data.content) data.content = stripHtml(data.content);

    // Enrich item with high-precision NLP (Sentiment, Emotions, Urgency, Topics)
    data = await enrichItemWithNLP(data);

    const existing = await Listening.findOne({ hash: data.hash });
    if (existing) return null;

    const result = await Listening.create(data);

    if (data.keywords) {
      await handleKeywordMentionCount(data.keywords, 1);
    }

    return result;
  } catch (error: any) {
    if (error.code === 11000) return null;
    logger.error('Error inserting listening data:', error.message);
    return null;
  }
};

export const bulkInsertListening = async (dataArray: any[]) => {
  try {
    if (!dataArray || dataArray.length === 0) return 0;

    const hashes = dataArray.map((item) => item.hash);
    const existingHashes = await Listening.find({ hash: { $in: hashes } }).distinct('hash');

    const rawNewData = dataArray
      .filter((item) => !existingHashes.includes(item.hash))
      .map((item) => ({
        ...item,
        title: stripHtml(item.title || ''),
        content: stripHtml(item.content || ''),
      }));

    if (rawNewData.length === 0) return 0;

    // Batch enrich with NLP
    const newData = await enrichBatchWithNLP(rawNewData);

    const bulkOps = newData.map((article) => ({
      updateOne: {
        filter: { hash: article.hash },
        update: { $setOnInsert: article },
        upsert: true,
      },
    }));

    const result = await Listening.bulkWrite(bulkOps);
    const insertedCount = result.upsertedCount || 0;

    if (insertedCount > 0) {
      const keywordCounts: Record<string, number> = {};
      for (const item of newData) {
        const kw = item.keywords || item.keyword;
        if (kw && typeof kw === 'string') {
          keywordCounts[kw] = (keywordCounts[kw] || 0) + 1;
        }
      }
      for (const [kw, count] of Object.entries(keywordCounts)) {
        await handleKeywordMentionCount(kw, count);
      }
    }

    return insertedCount;
  } catch (error: any) {
    logger.error('Error bulk inserting:', error.message);
    return 0;
  }
};
`;
fs.writeFileSync(listeningServicePath, listeningServiceContent, 'utf8');
console.log('Updated src/services/listening.service.ts with NLP integration');
