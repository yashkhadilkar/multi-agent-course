/**
 * Long-term memory: durable facts and preferences per X-User-Id, in the `memories` collection.
 *
 * Written only by an explicit save_memory call, and every row is visible at GET /memory and
 * removable with DELETE /memory/:id. Nothing is remembered anywhere else, so deleting a row
 * is the whole of forgetting it: the next recall simply does not find it.
 *
 * Recall is semantic: $vectorSearch on memories.embedding with userId as a filter inside the
 * search, capped at RECALL_LIMIT rows and RECALL_TOKENS tokens (SPEC 5.3).
 *
 * Atlas Search indexes are eventually consistent, in both directions:
 *   - a row saved seconds ago may not be in the index yet, so rows younger than INDEX_LAG_MS
 *     are read from the collection and scored here as well;
 *   - a row deleted seconds ago may still be in the index, so every hit is re-read from the
 *     collection and a hit whose row is gone is dropped. A deleted memory never comes back.
 */
import { COLLECTIONS, ListMemoryResponse, SEARCH_INDEXES, newId, type MemoryDoc } from '@lumina/contract';
import { db } from './db.js';
import { cosine, embedOne } from './embeddings.js';
import { env } from './env.js';

/** SPEC 5.3: inject at most ~10 memories / ~1 000 tokens. */
const RECALL_LIMIT = 10;
const RECALL_TOKENS = 1000;
/**
 * No similarity floor on recall. A standing preference ("answer in British English") is
 * about how to answer, not what, so it scores low against almost any question and a floor
 * would drop exactly the memories that should cross threads. The caps bound the cost instead.
 */
const NUM_CANDIDATES = 100;
/** How long a fresh row may take to reach the vector index. Rows this young are also scanned directly. */
const INDEX_LAG_MS = 2 * 60_000;
/** At or above this cosine, a new memory restates an existing one and is not saved again. */
const DUPLICATE_COSINE = 0.9;
export const MAX_MEMORY_CHARS = 500;

export type RecalledMemory = { id: string; text: string; createdAt: Date; similarity: number };
export type SaveResult = { saved: true; id: string } | { saved: false; duplicateOf: { id: string; text: string } };

type StoredMemory = Omit<MemoryDoc, 'createdAt'> & { createdAt: Date };
const memories = async () => (await db()).collection<StoredMemory>(COLLECTIONS.memories);

/** A rough count, enough for a budget: English runs about four characters to a token. */
const estimateTokens = (text: string) => Math.ceil(text.length / 4);
const normText = (t: string) => t.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();

type Scored = { id: string; similarity: number };

/**
 * The user's rows nearest to `vector`, best first. Atlas reports cosine as (1 + cos) / 2;
 * this returns plain cosine either way, so thresholds mean the same on both backends.
 */
async function nearest(userId: string, vector: number[], limit: number): Promise<Scored[]> {
  const col = await memories();
  if (env.vectorBackend === 'mongo-cosine-scan') {
    // The documented local fallback: a plain mongod has no vector index, so score in Node.
    const rows = await col.find({ userId }, { projection: { embedding: 1 } }).toArray();
    return rows
      .map((r) => ({ id: r._id, similarity: cosine(vector, r.embedding) }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);
  }
  const hits = await col
    .aggregate<{ _id: string; score: number }>([
      {
        $vectorSearch: {
          index: SEARCH_INDEXES.memoriesVector,
          path: 'embedding',
          queryVector: vector,
          numCandidates: Math.max(NUM_CANDIDATES, limit),
          limit,
          filter: { userId }
        }
      },
      { $project: { _id: 1, score: { $meta: 'vectorSearchScore' } } }
    ])
    .toArray();
  return hits.map((h) => ({ id: h._id, similarity: 2 * h.score - 1 }));
}

/**
 * The index's hits, re-read from the collection (dropping any deleted since they were
 * indexed), together with rows too young to be indexed yet, scored here. Best first.
 */
async function nearestFresh(userId: string, vector: number[], limit: number): Promise<RecalledMemory[]> {
  const hits = await nearest(userId, vector, limit);
  const score = new Map(hits.map((h) => [h.id, h.similarity]));
  const rows = await (await memories())
    .find(
      { userId, $or: [{ _id: { $in: [...score.keys()] } }, { createdAt: { $gte: new Date(Date.now() - INDEX_LAG_MS) } }] },
      { projection: { text: 1, createdAt: 1, embedding: 1 } }
    )
    .toArray();
  return rows
    .map((r) => ({ id: r._id, text: r.text, createdAt: r.createdAt, similarity: score.get(r._id) ?? cosine(vector, r.embedding) }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit);
}

// ---------------------------------------------------------------- recall

/**
 * Whether the user has saved anything at all. One indexed read ({userId, createdAt}), read
 * from the collection rather than the vector index, so a row saved a second ago counts.
 */
async function hasMemories(userId: string): Promise<boolean> {
  return (await (await memories()).findOne({ userId }, { projection: { _id: 1 } })) !== null;
}

/**
 * The memories to put in front of the model for this question, inside both caps. A user with
 * nothing saved costs one indexed read: no embedding call, no vector search.
 */
export async function recallMemories(userId: string, query: string, signal?: AbortSignal): Promise<RecalledMemory[]> {
  if (!(await hasMemories(userId))) return [];
  const vector = await embedOne(query, signal);
  const ranked = await nearestFresh(userId, vector, RECALL_LIMIT);
  const kept: RecalledMemory[] = [];
  let tokens = 0;
  for (const m of ranked) {
    const cost = estimateTokens(m.text);
    if (tokens + cost > RECALL_TOKENS) continue; // a shorter one further down may still fit
    tokens += cost;
    kept.push(m);
  }
  return kept;
}

// ---------------------------------------------------------------- save

/** Saves a fact or preference unless the user already has one that says the same thing. */
export async function saveMemory(userId: string, text: string, sourceThread: string, signal?: AbortSignal): Promise<SaveResult> {
  const clean = text.replace(/\s+/g, ' ').trim().slice(0, MAX_MEMORY_CHARS);
  const col = await memories();

  const exact = await col.findOne({ userId, text: clean }, { projection: { text: 1 } });
  if (exact) return { saved: false, duplicateOf: { id: exact._id, text: exact.text } };

  const embedding = await embedOne(clean, signal);
  const [closest] = await nearestFresh(userId, embedding, 3);
  if (closest && (closest.similarity >= DUPLICATE_COSINE || normText(closest.text) === normText(clean))) {
    return { saved: false, duplicateOf: { id: closest.id, text: closest.text } };
  }

  const _id = newId('mem');
  await col.insertOne({ _id, userId, text: clean, embedding, sourceThread, createdAt: new Date() });
  return { saved: true, id: _id };
}

// ---------------------------------------------------------------- GET /memory, DELETE /memory/:id

export async function listMemories(userId: string): Promise<ListMemoryResponse> {
  const rows = await (await memories())
    .find({ userId }, { projection: { embedding: 0 } })
    .sort({ createdAt: -1 })
    .toArray();
  return ListMemoryResponse.parse({
    memories: rows.map((m) => ({
      id: m._id,
      text: m.text,
      ...(m.sourceThread ? { sourceThread: m.sourceThread } : {}),
      createdAt: new Date(m.createdAt).toISOString()
    }))
  });
}

/** False for an unknown id and for another user's, so the caller answers 404 to both. */
export async function deleteMemory(userId: string, memoryId: string): Promise<boolean> {
  const { deletedCount } = await (await memories()).deleteOne({ _id: memoryId, userId });
  return deletedCount === 1;
}
