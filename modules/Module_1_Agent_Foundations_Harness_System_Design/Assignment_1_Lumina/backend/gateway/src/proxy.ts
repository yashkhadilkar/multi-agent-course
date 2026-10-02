import http from 'node:http';
import https from 'node:https';
import { Transform } from 'node:stream';
import type express from 'express';
import type { Logger } from 'pino';
import { REQUEST_HEADER, USER_HEADER } from '@lumina/contract';
import { env } from './env.js';
import { sseHeaders, sseSend } from './sse.js';

/** Headers that describe one hop, not the message; never copied across the proxy. */
const HOP_BY_HOP = new Set([
  'connection',
  'keep-alive',
  'transfer-encoding',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'upgrade'
]);

/** A frame that ends an answer. If the agent's stream closes without one, it died. */
const TERMINAL_EVENT = /(^|\n)event: (done|error)\n/;

export type Forward =
  /** A validated JSON body, re-serialised. */
  | { kind: 'json'; body: unknown }
  /** The raw request streamed through, refused once it passes `maxBytes`. */
  | { kind: 'stream'; maxBytes: number; tooLarge: string }
  | { kind: 'none' };

const fail = (res: express.Response, status: number, error: string) =>
  res.status(status).json({ error, status, requestId: String(res.locals.requestId) });

/**
 * Forward one request to the agent service and relay its answer. Status codes from the agent
 * (400, 404, 413, 429, 502) pass through untouched; the gateway only adds the failures that
 * are its own to see: the agent unreachable (502), or gone mid-stream (an `error` event).
 */
export function proxy(req: express.Request, res: express.Response, forward: Forward, log: Logger): void {
  const requestId = String(res.locals.requestId);
  const target = new URL(env.agentUrl.replace(/\/$/, '') + req.originalUrl);

  const headers: http.OutgoingHttpHeaders = {
    [REQUEST_HEADER]: requestId,
    accept: req.header('accept') ?? '*/*'
  };
  const user = req.header(USER_HEADER);
  if (user) headers[USER_HEADER] = user;

  let payload: string | undefined;
  if (forward.kind === 'json') {
    payload = JSON.stringify(forward.body);
    headers['content-type'] = 'application/json';
    headers['content-length'] = Buffer.byteLength(payload);
  } else if (forward.kind === 'stream') {
    for (const name of ['content-type', 'content-length', 'transfer-encoding'] as const) {
      const value = req.headers[name];
      if (value) headers[name] = value;
    }
  }

  const upstream = (target.protocol === 'https:' ? https : http).request(target, { method: req.method, headers });
  // Set when the gateway itself cancels the upstream, so the error that follows is not a 502.
  let cancelled = false;

  // The browser left: cancel the agent's request so it aborts the LLM call instead of
  // spending tokens nobody will read. `close` on the response, not the request: the
  // request's `close` fires as soon as its body has been read.
  res.on('close', () => {
    if (res.writableFinished) return;
    cancelled = true;
    upstream.destroy(new Error('client disconnected'));
  });

  upstream.on('error', (err) => {
    if (cancelled || res.destroyed) return;
    if (!res.headersSent) {
      log.error({ err, requestId, target: target.pathname }, 'agent service unreachable');
      // A refused connection is an AggregateError with an empty message; its code says why.
      fail(res, 502, `agent service unreachable: ${err.message || (err as NodeJS.ErrnoException).code || 'connection failed'}`);
    } else if (!res.writableEnded) {
      // Headers already sent on a non-SSE body: nothing honest left to write.
      res.destroy(err);
    }
  });

  upstream.on('response', (up) => {
    const isSse = String(up.headers['content-type'] ?? '').startsWith('text/event-stream');
    for (const [name, value] of Object.entries(up.headers)) {
      if (value === undefined || HOP_BY_HOP.has(name) || name === REQUEST_HEADER) continue;
      res.setHeader(name, value);
    }
    res.status(up.statusCode ?? 502);
    if (isSse) return relaySse(up, res, requestId, log);
    up.pipe(res);
    up.on('close', () => {
      if (!up.complete && !res.writableEnded) res.destroy();
    });
  });

  if (forward.kind === 'stream') {
    let seen = 0;
    const counter = new Transform({
      transform(chunk: Buffer, _enc, done) {
        seen += chunk.length;
        // Backstop for a body sent without Content-Length: refuse it the moment it is too big.
        if (seen > forward.maxBytes) return done(new Error(forward.tooLarge));
        done(null, chunk);
      }
    });
    counter.on('error', (err) => {
      cancelled = true;
      upstream.destroy();
      req.unpipe(counter);
      req.resume();
      if (!res.headersSent) fail(res, 413, err.message);
    });
    req.pipe(counter).pipe(upstream);
  } else {
    upstream.end(payload);
  }
}

/**
 * Pass SSE frames through as they arrive: one write per upstream chunk, no buffering, no
 * re-parsing. If the agent's stream ends without a `done` or `error` frame, the agent died
 * mid-answer, and the browser gets an `error` event instead of a silently truncated answer.
 */
function relaySse(up: http.IncomingMessage, res: express.Response, requestId: string, log: Logger): void {
  sseHeaders(res);
  let terminal = false;
  let tail = '';

  up.on('data', (chunk: Buffer) => {
    if (!terminal) {
      const text = tail + chunk.toString('utf8');
      terminal = TERMINAL_EVENT.test(text);
      tail = text.slice(-32); // a frame header split across two chunks still matches
    }
    res.write(chunk);
  });

  up.on('error', () => {
    // Reported on `close` below, which always follows.
  });

  up.on('close', () => {
    if (res.writableEnded || res.destroyed) return;
    if (!terminal) {
      const reason = up.complete ? 'agent service ended the stream without finishing the answer' : 'agent service disconnected mid-stream';
      log.error({ requestId }, reason);
      sseSend(res, 'error', { status: 502, error: reason });
    }
    res.end();
  });
}
