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
import { AskBody, HealthResponse, REQUEST_HEADER, ROUTES, ThreadId, USER_HEADER, newId } from '@lumina/contract';
import { env } from './env.js';
import { pingDb } from './db.js';
import { runQuick } from './quick.js';

const log = pino({ level: env.logLevel });
const app = express();

app.disable('x-powered-by');
app.use((req, res, next) =>
  req.path.endsWith('/documents') && req.method === 'POST'
    ? next()
    : express.json({ limit: '1mb' })(req, res, next)
);

mkdirSync(env.runsDir, { recursive: true });

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

// ---------------------------------------------------------------- POST /threads/:threadId/ask

app.post('/threads/:threadId/ask', async (req, res) => {
  // Reuse the gateway's id so one request greps end to end; mint one when called directly.
  const requestId = req.header(REQUEST_HEADER) || newId('req');
  res.setHeader(REQUEST_HEADER, requestId);
  const reject = (status: number, error: string) => res.status(status).json({ error, status, requestId });

  const userId = req.header(USER_HEADER);
  if (!userId) return reject(401, 'X-User-Id header is required');
  // No threads collection yet, so any well-formed id is accepted. A malformed one cannot
  // name a thread that exists.
  if (!ThreadId.safeParse(req.params.threadId).success) return reject(404, `unknown thread ${req.params.threadId}`);

  const parsed = AskBody.safeParse(req.body ?? {});
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    return reject(400, issue ? `${issue.path.join('.') || 'body'}: ${issue.message}` : 'invalid body');
  }
  const ask = parsed.data;
  // Deep is refused rather than quietly run as quick: the server reports the gear it ran,
  // and it never changes the one the client asked for.
  if (ask.depth === 'deep') return reject(501, 'not implemented yet: depth "deep"');
  if (ask.mode === 'docs' || ask.spaceId) return reject(501, 'not implemented yet: document search (mode "docs" / spaceId)');

  await runQuick({ requestId, userId, threadId: req.params.threadId, query: ask.query }, res, log);
});

// ---------------------------------------------------------------- everything else: 501

const notImplemented = (route: string) => (_req: express.Request, res: express.Response) => {
  res.status(501).json({ error: `not implemented yet: ${route}. Build it in backend/agent/src/.`, status: 501 });
};

const IMPLEMENTED = new Set(['GET /health', 'GET /evals/report.json', 'POST /threads/:threadId/ask']);

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
