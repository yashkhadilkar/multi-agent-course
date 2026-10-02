#!/usr/bin/env node
/**
 * A tuning tool, not a gate: the real numbers come from benchmark/bench.mjs.
 *
 * Replays the bench's quick web workload (benchmark/bench.mjs buildWorkload + runWebWorkload:
 * 20 fresh questions then 20 repeats, one thread, concurrency 4, mode web) through the gateway,
 * timing TTFT exactly as benchmark/lib.mjs ask() does (clock before fetch, stop at first
 * `token` frame), and scoring grounding with the bench's own helpers.
 *
 *   node backend/agent/scripts/measure-quick.mjs <label> [agentLogFile] [--no-clear]
 *
 * Run it against a local gateway (:8787) whose agent writes its pino log to agentLogFile; the
 * log's `answer` lines give the server-side breakdown of the time before the first token.
 * Unless --no-clear, it first deletes the searchCache rows these questions would hit, so
 * every pass starts equally cold (restart the agent too, to empty its in-process LRU).
 * Writes <label>.json next to the current directory.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const LUMINA = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const { citationNumbers, snippetIsGrounded, percentile, mean, pool } = await import(pathToFileURL(`${LUMINA}/benchmark/lib.mjs`).href);
const require = createRequire(`${LUMINA}/backend/agent/package.json`);
const { MongoClient } = require('mongodb');
require('dotenv').config({ path: `${LUMINA}/.env` });

const [label = 'run', agentLog, ...flags] = process.argv.slice(2);
const TARGET = 'http://localhost:8787';
const USER = process.env.MEASURE_USER || 'bench';
const queries = JSON.parse(readFileSync(`${LUMINA}/benchmark/queries.json`, 'utf8'));
const sla = JSON.parse(readFileSync(`${LUMINA}/benchmark/sla.json`, 'utf8'));

// bench.mjs buildWorkload, verbatim in effect.
const distinct = queries.web;
const total = sla.workload.web_queries;
const repeats = Math.round(total * sla.workload.repeat_fraction);
const fresh = total - repeats;
const work = [];
for (let i = 0; i < fresh; i++) work.push(distinct[i % distinct.length]);
for (let i = 0; i < repeats; i++) work.push(work[i % Math.max(1, fresh)]);

// ---- clear the search cache rows these questions would hit (cache.ts normalizeQuery/cacheKey)
const normalizeQuery = (q) => q.normalize('NFKC').toLowerCase().replace(/\s+/g, ' ').trim().replace(/[\s?!.,;:]+$/, '');
if (!flags.includes('--no-clear')) {
  const client = new MongoClient(process.env.MONGODB_URI);
  await client.connect();
  const col = client.db(process.env.MONGODB_DB || 'lumina').collection('searchCache');
  const norms = distinct.map(normalizeQuery);
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const r = await col.deleteMany({
    $or: [
      { _id: { $in: norms.map((n) => createHash('sha256').update(JSON.stringify([n, 'tavily'])).digest('hex')) } },
      // follow-up queries that were searched with an earlier question prepended
      { query: { $regex: `(${norms.map(esc).join('|')})` } }
    ]
  });
  console.log(`cleared ${r.deletedCount} search cache rows`);
  await client.close();
}

// bench.mjs stripHtml + fetchPageText, verbatim.
const stripHtml = (html) =>
  html.replace(/<script[\s\S]*?<\/script>/gi, ' ').replace(/<style[\s\S]*?<\/style>/gi, ' ').replace(/<[^>]+>/g, ' ').replace(/&[a-z#0-9]+;/gi, ' ');
const pageCache = new Map();
async function fetchPageText(url) {
  if (pageCache.has(url)) return pageCache.get(url);
  let text = null;
  try {
    const res = await fetch(url, { redirect: 'follow', headers: { 'user-agent': 'lumina-bench/0.1 (+course benchmark)' }, signal: AbortSignal.timeout(12000) });
    if (res.ok) text = stripHtml(await res.text());
  } catch {
    text = null;
  }
  pageCache.set(url, text);
  return text;
}

async function ask(threadId, query) {
  const started = Date.now();
  const res = await fetch(`${TARGET}/threads/${threadId}/ask`, {
    method: 'POST',
    headers: { 'x-user-id': USER, 'content-type': 'application/json' },
    body: JSON.stringify({ query, mode: 'web', depth: 'quick' }),
    signal: AbortSignal.timeout(300000)
  });
  const out = { query, requestId: res.headers.get('x-request-id'), status: res.status, headersMs: Date.now() - started, events: [], trace: [], sources: [], text: '', done: null, error: null, ttftMs: null };
  if (!res.ok) {
    out.error = await res.text();
    return out;
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  const handle = (frame) => {
    let event = 'message';
    const data = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).trim());
    }
    if (!data.length) return;
    let p;
    try { p = JSON.parse(data.join('\n')); } catch { return; }
    const t = Date.now() - started;
    out.events.push({ event, t, tool: p.tool });
    if (event === 'trace') out.trace.push({ ...p, t });
    else if (event === 'sources') { out.sources = p; out.sourcesMs = t; }
    else if (event === 'token') { if (out.ttftMs === null) out.ttftMs = t; out.text += p.text ?? ''; }
    else if (event === 'done') out.done = p;
    else if (event === 'error') out.error = p;
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n\n')) !== -1) { handle(buf.slice(0, i)); buf = buf.slice(i + 2); }
  }
  if (buf.trim()) handle(buf);
  out.latencyMs = Date.now() - started;
  return out;
}

const thread = await (await fetch(`${TARGET}/threads`, { method: 'POST', headers: { 'x-user-id': USER, 'content-type': 'application/json' }, body: '{}' })).json();
console.log(`${label}: ${work.length} asks on ${thread.threadId}, concurrency ${sla.workload.concurrency}`);
const t0 = Date.now();
const results = (await pool(work.map((q) => () => ask(thread.threadId, q)), sla.workload.concurrency)).map((r) => (r.ok ? r.value : { error: String(r.error) }));
console.log(`  done in ${((Date.now() - t0) / 1000).toFixed(0)}s`);

// ---- grounding, as bench.mjs scoreGrounding (web answers)
const g = { checked: 0, grounded: 0, unverifiable: 0, dangling: 0 };
for (const a of results) {
  if (!a.done) continue;
  const bySource = new Map((a.sources ?? []).map((s) => [s.n, s]));
  for (const n of citationNumbers(a.text)) {
    g.checked++;
    const src = bySource.get(n);
    if (!src) { g.dangling++; continue; }
    const page = src.url ? await fetchPageText(src.url) : null;
    if (!page) g.unverifiable++;
    else if (snippetIsGrounded(src.snippet, page)) g.grounded++;
  }
}

// ---- join with the agent's answer log lines
const logs = new Map();
if (agentLog) {
  for (const line of readFileSync(agentLog, 'utf8').split('\n')) {
    if (!line.startsWith('{')) continue;
    try {
      const j = JSON.parse(line);
      if (j.msg === 'answer' && j.requestId) logs.set(j.requestId, j);
    } catch { /* not json */ }
  }
}

const ok = results.filter((a) => a.done);
const rows = ok.map((a) => {
  const l = logs.get(a.requestId);
  const tm = l?.timing;
  const firstTurn = tm?.turns?.[0];
  const firstTraceT = a.trace[0]?.t ?? null;
  return {
    query: a.query.slice(0, 50),
    cached: a.done.searchCached,
    ttft: a.ttftMs,
    latency: a.latencyMs,
    cost: a.done.costUsd,
    tokIn: a.done.tokens.in,
    tokOut: a.done.tokens.out,
    tools: a.trace.length,
    sources: a.sources.length,
    turns: tm?.turns?.length ?? null,
    // server-side breakdown (ms from run start)
    serverTtft: a.done.ttftMs,
    firstStep: tm?.firstStepMs ?? null,
    reads: tm?.readsMs ?? null,
    llm1Start: firstTurn?.startMs ?? null,
    llm1FirstThinking: firstTurn?.firstThinkingMs ?? null,
    llm1FirstText: firstTurn?.firstTextMs ?? null,
    turnsDetail: tm?.turns ?? null,
    clientOverhead: a.ttftMs - a.done.ttftMs,
    firstTraceT,
    markdown: /\*\*|^#{1,6}\s|^\s*\*\s/m.test(a.text),
    text: a.text
  };
});

const p = (xs, q) => percentile(xs.filter((x) => x !== null && x !== undefined), q);
const summary = {
  label,
  answered: ok.length,
  errors: results.length - ok.length,
  errorDetail: results.filter((a) => !a.done).map((a) => String(a.error).slice(0, 200)),
  ttftP50: p(rows.map((r) => r.ttft), 50),
  ttftP95: p(rows.map((r) => r.ttft), 95),
  ttftP50Uncached: p(rows.filter((r) => !r.cached).map((r) => r.ttft), 50),
  ttftP95Uncached: p(rows.filter((r) => !r.cached).map((r) => r.ttft), 95),
  ttftP50Cached: p(rows.filter((r) => r.cached).map((r) => r.ttft), 50),
  ttftP95Cached: p(rows.filter((r) => r.cached).map((r) => r.ttft), 95),
  answerP50: p(rows.map((r) => r.latency), 50),
  answerP95: p(rows.map((r) => r.latency), 95),
  costMean: mean(rows.map((r) => r.cost)),
  costMax: Math.max(...rows.map((r) => r.cost)),
  overCap: rows.filter((r) => r.cost > 0.05).length,
  multiTurn: rows.filter((r) => (r.turns ?? 1) > 1).length,
  cacheHitRatePct: (rows.filter((r) => r.cached).length / rows.length) * 100,
  grounding: { ...g, rate: g.grounded / Math.max(1, g.checked - g.unverifiable) },
  markdownAnswers: rows.filter((r) => r.markdown).length,
  breakdownMedians: {
    clientOverhead: p(rows.map((r) => r.clientOverhead), 50),
    firstStep: p(rows.map((r) => r.firstStep), 50),
    readsAfterFirstStep: p(rows.map((r) => (r.reads ?? 0) - (r.firstStep ?? 0)), 50),
    llm1ToFirstThinking: p(rows.map((r) => (r.llm1FirstThinking === null ? null : r.llm1FirstThinking - r.llm1Start)), 50),
    llm1ToFirstText: p(rows.map((r) => (r.llm1FirstText === null ? null : r.llm1FirstText - r.llm1Start)), 50),
    firstTextToServerTtft: p(rows.filter((r) => r.turns === 1).map((r) => r.serverTtft - r.llm1FirstText), 50),
    serverTtft: p(rows.map((r) => r.serverTtft), 50)
  }
};
writeFileSync(resolve(process.cwd(), `${label}.json`), JSON.stringify({ summary, rows }, null, 2));
console.log(JSON.stringify(summary, null, 2));
