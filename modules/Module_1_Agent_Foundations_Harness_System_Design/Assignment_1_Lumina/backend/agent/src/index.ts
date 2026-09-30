/**
 * LUMINA agent service — the AI backend. PROVIDED SKELETON: YOU BUILD THIS OUT.
 * This is where the real work is. Provider keys live only in this process.
 *
 * What is already here: the server, /health (Mongo ping + which model, provider and
 * vector backend are live), and a 501 for every other route.
 *
 * What you build (README Part 1, in this order — each step is testable with curl -N):
 *   1. the QUICK loop: plan → choose tool → observe → repeat → answer, with web_search
 *      and fetch_page, streaming trace → sources → token → done. sources BEFORE the
 *      first token. Disable compression on this route and flush after every event.
 *   2. the search cache: in-process LRU over the searchCache collection (TTL index),
 *      key = sha256(normalized query + provider). searchCached only when every hit.
 *   3. threads + messages, so a follow-up sees the thread.
 *   4. memory: save_memory / recall_memory over the memories vector index; GET /memory,
 *      DELETE /memory/:id.
 *   5. the run log: one runs/<requestId>.json per answer, in the RunLog shape from the
 *      contract. Ten lines. The gates read it, so it is not optional.
 *   6. spaces + the jobs worker: upload → GridFS → parse → chunk → embed → upsert →
 *      read-your-write probe → indexed.
 *   7. hybrid retrieval: $vectorSearch + $search fused with RRF, page locators.
 *   8. DEEP search (depth: "deep"): plan_research decomposes the question into 3–6
 *      sub-questions, you stream a `plan` event BEFORE retrieving anything, research each
 *      sub-question, then merge the results into ONE citation numbering and synthesise.
 *      Every trace step and every source carries the subQuestion it served. Deep runs
 *      under the wider caps (maxToolCallsDeep, maxWallClockSecDeep) and behind
 *      DEEP_DAILY_CAP → 429 {error, resetsAt}.
 *
 * Three rules to hold on to while you write it:
 *   - Fail loud. A provider exception ends the run with terminated:"error" and a 502.
 *     Never a try/catch that returns a plausible answer. (Live Translate served English
 *     for weeks because of exactly that catch.)
 *   - Grounded or nothing. A citation that does not resolve to something retrieved in
 *     THIS request is an automatic fail.
 *   - Depth is opted into, never drifted into. A quick search may not call plan_research,
 *     however much the model would like to. Deep costs several times more, and a product
 *     that escalates itself is a product with an unbounded bill.
 */
import express from 'express';
import pino from 'pino';
import { mkdirSync } from 'node:fs';
import { AskBody, CreateThreadBody, HealthResponse, REQUEST_HEADER, ROUTES, USER_HEADER, newId } from '@lumina/contract';
import { env } from './env.js';
import { pingDb } from './db.js';
import { deleteMemory, listMemories } from './memory.js';
import { runQuick } from './quick.js';
import { beginTurn, createThread, findThread, getThread, listThreads, saveAnswer } from './threads.js';

const log = pino({ level: env.logLevel });
const app = express();

app.disable('x-powered-by');
app.use((req, res, next) =>
  req.path.endsWith('/documents') && req.method === 'POST'
    ? next()
    : express.json({ limit: '1mb' })(req, res, next)
);

mkdirSync(env.runsDir, { recursive: true });

// ---------------------------------------------------------------- request id + X-User-Id

type Locals = { requestId: string; userId: string };

const PUBLIC_PATHS = new Set(['/health', '/evals/report.json']);

app.use((req, res, next) => {
  // Reuse the gateway's id so one request greps end to end; mint one when called directly.
  const requestId = req.header(REQUEST_HEADER) || newId('req');
  res.locals.requestId = requestId;
  res.setHeader(REQUEST_HEADER, requestId);
  if (PUBLIC_PATHS.has(req.path)) return next();
  // Checked here as well as at the gateway: the user id is what scopes every thread read.
  const userId = req.header(USER_HEADER);
  if (!userId) return void res.status(401).json({ error: 'X-User-Id header is required', status: 401, requestId });
  res.locals.userId = userId;
  next();
});

const reject = (res: express.Response, status: number, error: string) =>
  res.status(status).json({ error, status, requestId: (res.locals as Locals).requestId });

/**
 * Express 4 does not catch a rejected promise. A store that throws is an upstream failure:
 * 502, logged. The driver's message stays in the log, since it can name the cluster host.
 */
const handle =
  (what: string, fn: (req: express.Request, res: express.Response, locals: Locals) => Promise<unknown>) =>
  (req: express.Request, res: express.Response) => {
    const locals = res.locals as Locals;
    fn(req, res, locals).catch((err: unknown) => {
      log.error({ requestId: locals.requestId, err }, `${what} failed`);
      if (!res.headersSent) reject(res, 502, `${what} failed`);
      else if (!res.writableEnded) res.end();
    });
  };

// ---------------------------------------------------------------- /health (implemented)

// No X-User-Id here: the grader and Fly's health check call it bare. `ai` is left off on
// purpose, because the gateway nests this whole body there; the agent does not grade itself.
app.get('/health', async (_req, res) => {
  const ping = await pingDb();
  // Log the reason but keep it out of the body: a driver error can name the cluster host.
  if (ping.status === 'down') log.error({ err: ping.error }, 'health: mongo ping failed');

  const body = HealthResponse.parse({
    status: ping.status === 'ok' ? 'ok' : 'degraded',
    model: env.llmModel,
    searchProvider: env.searchProvider,
    vectorStore: env.vectorBackend,
    db: ping.status
  });
  // 503 when degraded, the same convention the gateway uses; bench reads the body either way.
  res.status(ping.status === 'ok' ? 200 : 503).json(body);
});

// ---------------------------------------------------------------- threads

const badBody = (error: { issues: { path: (string | number)[]; message: string }[] }) => {
  const issue = error.issues[0];
  return issue ? `${issue.path.join('.') || 'body'}: ${issue.message}` : 'invalid body';
};

app.post(
  '/threads',
  handle('creating the thread', async (req, res, { userId }) => {
    const parsed = CreateThreadBody.safeParse(req.body ?? {});
    if (!parsed.success) return reject(res, 400, badBody(parsed.error));
    res.status(201).json({ threadId: await createThread(userId, parsed.data.title) });
  })
);

app.get(
  '/threads',
  handle('listing threads', async (_req, res, { userId }) => {
    res.json(await listThreads(userId));
  })
);

// Another user's thread is a 404, not a 403: a 403 would confirm that it exists.
app.get(
  '/threads/:threadId',
  handle('loading the thread', async (req, res, { userId }) => {
    const thread = await getThread(userId, req.params.threadId!);
    if (!thread) return reject(res, 404, `unknown thread ${req.params.threadId}`);
    res.json(thread);
  })
);

// ---------------------------------------------------------------- POST /threads/:threadId/ask

app.post(
  '/threads/:threadId/ask',
  handle('starting the answer', async (req, res, { requestId, userId }) => {
    const { threadId } = req.params as { threadId: string };
    const parsed = AskBody.safeParse(req.body ?? {});
    if (!parsed.success) return reject(res, 400, badBody(parsed.error));
    const ask = parsed.data;

    // Everything that can be known up front is a real status, before the first byte of the stream.
    if (!(await findThread(userId, threadId))) return reject(res, 404, `unknown thread ${threadId}`);
    // Deep is refused rather than quietly run as quick: the server reports the gear it ran,
    // and it never changes the one the client asked for.
    if (ask.depth === 'deep') return reject(res, 501, 'not implemented yet: depth "deep"');
    if (ask.mode === 'docs' || ask.spaceId) return reject(res, 501, 'not implemented yet: document search (mode "docs" / spaceId)');

    // The question is saved whatever the run's outcome; the answer only if it ends done or cap.
    const { questionId, history, earlierQuestions } = await beginTurn(userId, threadId, ask.query);
    await runQuick(
      {
        requestId,
        userId,
        threadId,
        query: ask.query,
        history,
        earlierQuestions,
        saveAnswer: (answer) => saveAnswer(userId, threadId, questionId, answer)
      },
      res,
      log
    );
  })
);

// ---------------------------------------------------------------- memory

app.get(
  '/memory',
  handle('listing memories', async (_req, res, { userId }) => {
    res.json(await listMemories(userId));
  })
);

// Another user's memory is a 404 too: its id says nothing about whether it exists.
app.delete(
  '/memory/:memoryId',
  handle('deleting the memory', async (req, res, { userId }) => {
    if (!(await deleteMemory(userId, req.params.memoryId!))) return reject(res, 404, `unknown memory ${req.params.memoryId}`);
    res.status(204).end();
  })
);

// ---------------------------------------------------------------- everything else: 501

const notImplemented = (route: string) => (_req: express.Request, res: express.Response) => {
  res.status(501).json({ error: `not implemented yet: ${route}. Build it in backend/agent/src/.`, status: 501 });
};

const IMPLEMENTED = new Set([
  'GET /health',
  'GET /evals/report.json',
  'POST /threads',
  'GET /threads',
  'GET /threads/:threadId',
  'POST /threads/:threadId/ask',
  'GET /memory',
  'DELETE /memory/:memoryId'
]);

for (const route of ROUTES) {
  if (IMPLEMENTED.has(`${route.method} ${route.path}`)) continue;
  const method = route.method.toLowerCase() as 'get' | 'post' | 'delete';
  app[method](route.path, notImplemented(`${route.method} ${route.path}`));
}

app.use((req, res) => res.status(404).json({ error: `no route ${req.method} ${req.path}`, status: 404 }));

app.use((err: Error & { type?: string }, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  // A body that is not JSON is the client's mistake, not an upstream failure.
  if (err.type === 'entity.parse.failed') {
    res.status(400).json({ error: 'body is not valid JSON', status: 400 });
    return;
  }
  log.error({ err }, 'agent error');
  res.status(502).json({ error: err.message, status: 502 });
});

app.listen(env.port, () => {
  log.info(
    {
      port: env.port,
      model: env.llmModel,
      searchProvider: env.searchProvider,
      vectorStore: env.vectorBackend,
      caps: {
        quick: { toolCalls: env.maxToolCalls, wallClockSec: env.maxWallClockSec },
        deep: { toolCalls: env.maxToolCallsDeep, wallClockSec: env.maxWallClockSecDeep, dailyCap: env.deepDailyCap }
      }
    },
    'agent up — every route but /health returns 501 until you build it'
  );
});
