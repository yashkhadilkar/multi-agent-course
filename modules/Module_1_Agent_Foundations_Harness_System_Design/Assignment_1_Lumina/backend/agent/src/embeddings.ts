import OpenAI from 'openai';
import { EMBEDDING_DIMS } from '@lumina/contract';
import { env, secrets } from './env.js';

/** Thrown for any failure of the embedding provider, so callers can tell it from a store failure. */
export class EmbeddingError extends Error {}

let openai: OpenAI | null = null;
const client = () => (openai ??= new OpenAI({ apiKey: secrets.openai, maxRetries: 1 }));

/** One vector per input, in order. Every vector is checked against the index's dimensions. */
export async function embed(texts: string[], signal?: AbortSignal): Promise<number[][]> {
  if (!secrets.openai) throw new EmbeddingError('embeddings: OPENAI_API_KEY is not set');
  let res: OpenAI.CreateEmbeddingResponse;
  try {
    res = await client().embeddings.create({ model: env.embeddingModel, input: texts }, { signal });
  } catch (err) {
    if (signal?.aborted) throw err;
    throw new EmbeddingError(`embeddings: ${(err as Error).message}`);
  }
  const vectors = [...res.data].sort((a, b) => a.index - b.index).map((d) => d.embedding);
  if (vectors.length !== texts.length || vectors.some((v) => v.length !== EMBEDDING_DIMS)) {
    throw new EmbeddingError(`embeddings: expected ${texts.length} vectors of ${EMBEDDING_DIMS} dims from ${env.embeddingModel}`);
  }
  return vectors;
}

export async function embedOne(text: string, signal?: AbortSignal): Promise<number[]> {
  return (await embed([text], signal))[0]!;
}

export function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
}
