import { MongoClient, type Db } from 'mongodb';
import { env } from './env.js';

let connecting: Promise<MongoClient> | null = null;

/**
 * One client per process. The driver pools connections; do not open one per request.
 * Concurrent callers share one connect attempt, and a failed attempt is forgotten so the
 * next call retries instead of reusing a client that never connected.
 */
export async function db(): Promise<Db> {
  if (!env.mongoUri) throw new Error('MONGODB_URI is not set — copy .env.example to .env');
  if (!connecting) {
    const client = new MongoClient(env.mongoUri, { serverSelectionTimeoutMS: 5000 });
    connecting = client.connect().catch((err: unknown) => {
      connecting = null;
      void client.close().catch(() => {});
      throw err;
    });
  }
  return (await connecting).db(env.mongoDb);
}

/**
 * Must answer inside the gateway's 3 s budget for the agent's /health. Otherwise a dead
 * Mongo reads at the gateway as a dead agent, and /health blames the wrong component.
 */
const PING_TIMEOUT_MS = 2000;

export type DbPing = { status: 'ok' } | { status: 'down'; error: string };

/** A real round trip to the server. `down` always carries the reason, for the log. */
export async function pingDb(): Promise<DbPing> {
  let timer: NodeJS.Timeout | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`no reply within ${PING_TIMEOUT_MS}ms`)), PING_TIMEOUT_MS);
  });
  try {
    await Promise.race([db().then((d) => d.command({ ping: 1 })), timeout]);
    return { status: 'ok' };
  } catch (err) {
    return { status: 'down', error: (err as Error).message };
  } finally {
    clearTimeout(timer);
  }
}
