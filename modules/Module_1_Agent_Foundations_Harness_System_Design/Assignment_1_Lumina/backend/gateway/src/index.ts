/**
 * LUMINA gateway — the only service the browser talks to.
 *
 * In order, for every contract route: CORS, the request id, one pino line per request,
 * X-User-Id (401), the per-user rate limit on the two routes that cost money (429), the
 * upload size check (413), zod validation from @lumina/contract (400), then the proxy to the
 * agent service. The ask route's SSE stream passes through unbuffered; an unreachable agent
 * is a 502, an agent that dies mid-stream is an `error` event.
 *
 * No provider key is ever read here.
 */
import express from 'express';
import cors from 'cors';
import pino from 'pino';
import { randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import type { ZodTypeAny } from 'zod';
import {
  AskBody,
  CreateSpaceBody,
  CreateThreadBody,
  HealthResponse,
  MAX_UPLOAD_BYTES,
  REQUEST_HEADER,
  ROUTES,
  USER_HEADER
} from '@lumina/contract';
import { env } from './env.js';
import { proxy } from './proxy.js';

const log = pino({ level: env.logLevel });
const app = express();

app.disable('x-powered-by');

// Liveness for Fly's check: this process is up and serving HTTP, nothing more. It never
// calls the agent, so a dead agent cannot get a healthy gateway pulled from routing. Ahead
// of the request log so a check every 15 s is not a log line. /health is the full report.
app.get('/livez', (_req, res) => void res.type('text/plain').send('ok'));

app.use(
  cors({
    origin: env.corsOrigins,
    credentials: false,
    exposedHeaders: [REQUEST_HEADER, 'Retry-After', 'RateLimit-Limit', 'RateLimit-Remaining']
  })
);

const fail = (res: express.Response, status: number, error: string) =>
  res.status(status).json({ error, status, requestId: String(res.locals.requestId) });

// ---------------------------------------------------------------- request id + request log

// Reused if the caller sent a sane one, generated if not, forwarded to the agent service and
// logged by both. This is what makes one request greppable end to end.
const SANE_ID = /^[\w.:-]{1,128}$/;

app.use((req, res, next) => {
  const inbound = req.header(REQUEST_HEADER)?.trim();
  const requestId = inbound && SANE_ID.test(inbound) ? inbound : `req_${randomUUID().slice(0, 12)}`;
  res.locals.requestId = requestId;
  res.setHeader(REQUEST_HEADER, requestId);

  // One line per request, written when it ends, so a stream logs its real duration.
  const started = process.hrtime.bigint();
  res.on('close', () => {
    log.info(
      {
        method: req.method,
        route: req.route ? String(req.route.path) : req.path,
        status: res.statusCode,
        ms: Math.round(Number(process.hrtime.bigint() - started) / 1e6),
        requestId,
        userId: req.header(USER_HEADER) ?? null,
        ...(res.writableFinished ? {} : { aborted: true })
      },
      'request'
    );
  });
  next();
});

// JSON everywhere except the multipart upload route, which is streamed to the agent as is.
app.use((req, res, next) =>
  req.path.endsWith('/documents') && req.method === 'POST'
    ? next()
    : express.json({ limit: '1mb' })(req, res, next)
);

// ---------------------------------------------------------------- guards

const requireUser: express.RequestHandler = (req, res, next) => {
  if (!req.header(USER_HEADER)?.trim()) return void fail(res, 401, 'X-User-Id header is required');
  next();
};

/**
 * Fixed one-minute window per X-User-Id, shared by ask and upload: the two routes that cost
 * money. Everything else (thread lists, document status polling) is never limited.
 */
const windows = new Map<string, { start: number; count: number }>();
setInterval(() => {
  const cutoff = Date.now() - 60_000;
  for (const [user, w] of windows) if (w.start < cutoff) windows.delete(user);
}, 60_000).unref();

const rateLimit: express.RequestHandler = (req, res, next) => {
  const user = req.header(USER_HEADER)!.trim();
  const now = Date.now();
  let w = windows.get(user);
  if (!w || now - w.start >= 60_000) windows.set(user, (w = { start: now, count: 0 }));
  w.count += 1;

  const limit = env.rateLimitPerMinute;
  res.setHeader('RateLimit-Limit', limit);
  res.setHeader('RateLimit-Remaining', Math.max(0, limit - w.count));
  if (w.count > limit) {
    res.setHeader('Retry-After', Math.ceil((w.start + 60_000 - now) / 1000));
    log.warn({ requestId: res.locals.requestId, userId: user, limit }, 'rate limited');
    return void fail(res, 429, `rate limit: ${limit} requests per minute on ask and upload`);
  }
  next();
};

// Multipart framing (boundary, part headers) rides on top of the file itself.
const MULTIPART_OVERHEAD = 64 * 1024;
const TOO_LARGE = `file is larger than ${MAX_UPLOAD_BYTES / 1024 / 1024} MB`;

/** Refuse an oversized upload from its Content-Length, before a byte reaches the agent. */
const uploadSize: express.RequestHandler = (req, res, next) => {
  const declared = Number(req.header('content-length'));
  if (Number.isFinite(declared) && declared > MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD) {
    fail(res, 413, TOO_LARGE);
    // Drain the rest so the browser reads the 413 instead of a reset connection.
    req.resume();
    return;
  }
  next();
};

const validate =
  (schema: ZodTypeAny): express.RequestHandler =>
  (req, res, next) => {
    const parsed = schema.safeParse(req.body ?? {});
    if (!parsed.success) {
      const issue = parsed.error.issues[0];
      return void fail(res, 400, issue ? `${issue.path.join('.') || 'body'}: ${issue.message}` : 'invalid body');
    }
    res.locals.body = parsed.data;
    next();
  };

// ---------------------------------------------------------------- /health

app.get('/health', async (_req, res) => {
  let ai: { status: 'ok' | 'down' } & Record<string, unknown> = { status: 'down' };
  try {
    const upstream = await fetch(`${env.agentUrl}/health`, {
      headers: { [REQUEST_HEADER]: String(res.locals.requestId) },
      signal: AbortSignal.timeout(3000)
    });
    const body = (await upstream.json()) as Record<string, unknown>;
    ai = { ...body, status: upstream.ok ? 'ok' : 'down' };
  } catch (err) {
    // Health tells the truth about a dead dependency. It never pretends.
    ai = { status: 'down', error: (err as Error).message };
  }

  const body: HealthResponse = {
    status: ai.status === 'ok' ? 'ok' : 'degraded',
    model: String(ai.model ?? 'unset'),
    searchProvider: (ai.searchProvider as HealthResponse['searchProvider']) ?? 'tavily',
    vectorStore: (ai.vectorStore as HealthResponse['vectorStore']) ?? 'atlas-vector-search',
    db: (ai.db as HealthResponse['db']) ?? 'down',
    ai
  };
  res.status(ai.status === 'ok' ? 200 : 503).json(body);
});

// ---------------------------------------------------------------- /evals/report.json

// The eval's output, read from disk on every request so a rebuilt report shows without a
// restart. Served as written: the gateway never edits or fills in a number.
app.get('/evals/report.json', (_req, res, next) => {
  res.sendFile(env.evalsReport, { headers: { 'Cache-Control': 'no-cache' } }, (err?: NodeJS.ErrnoException) => {
    if (!err) return;
    if (err.code === 'ENOENT') {
      return void fail(res, 404, 'no eval report yet: run /fde-lumina-eval (or eval/build-report.mjs) to write reports/report.json');
    }
    next(err);
  });
});

// ---------------------------------------------------------------- the contract routes

const BODIES: Record<string, ZodTypeAny> = {
  'POST /threads': CreateThreadBody,
  'POST /threads/:threadId/ask': AskBody,
  'POST /spaces': CreateSpaceBody
};
const UPLOAD = 'POST /spaces/:spaceId/documents';
const LIMITED = new Set(['POST /threads/:threadId/ask', UPLOAD]);

for (const route of ROUTES) {
  // Both answered by the gateway itself, above.
  if (route.path === '/health' || route.path === '/evals/report.json') continue;
  const key = `${route.method} ${route.path}`;
  const chain: express.RequestHandler[] = [];

  if (route.auth) chain.push(requireUser);
  if (LIMITED.has(key)) chain.push(rateLimit);
  if (key === UPLOAD) chain.push(uploadSize);
  const schema = BODIES[key];
  if (schema) chain.push(validate(schema));

  chain.push((req, res) =>
    proxy(
      req,
      res,
      key === UPLOAD
        ? { kind: 'stream', maxBytes: MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD, tooLarge: TOO_LARGE }
        : schema
          ? { kind: 'json', body: res.locals.body }
          : { kind: 'none' },
      log
    )
  );

  const method = route.method.toLowerCase() as 'get' | 'post' | 'delete';
  app[method](route.path, ...chain);
}

// ---------------------------------------------------------------- static UI

// In production the gateway serves the built UI, so / and /evals come from one origin. Any
// browser navigation that no route above claimed gets index.html; the UI routes itself.
if (existsSync(env.webDist)) {
  app.use(express.static(env.webDist));
  app.get('*', (req, res, next) =>
    req.header('accept')?.includes('text/html') ? res.sendFile(`${env.webDist}/index.html`) : next()
  );
}

app.use((req, res) => {
  fail(res, 404, `no route ${req.method} ${req.path}`);
});

app.use((err: Error & { type?: string }, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  // A malformed or oversized body is the client's mistake, not an upstream failure.
  if (err.type === 'entity.parse.failed') return void fail(res, 400, 'body is not valid JSON');
  if (err.type === 'entity.too.large') return void fail(res, 413, 'body is too large');
  // Anything else thrown is a 502 with a log line, never a 200 with a plausible body (rule A1).
  log.error({ err, requestId: res.locals.requestId }, 'gateway error');
  if (res.headersSent) return void res.destroy(err);
  fail(res, 502, err.message);
});

app.listen(env.port, () => {
  log.info(
    {
      port: env.port,
      agentUrl: env.agentUrl,
      cors: env.corsOrigins,
      rateLimitPerMinute: env.rateLimitPerMinute,
      servesWeb: existsSync(env.webDist)
    },
    'gateway up'
  );
});
