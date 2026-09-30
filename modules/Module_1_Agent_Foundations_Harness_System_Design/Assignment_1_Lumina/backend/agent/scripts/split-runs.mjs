#!/usr/bin/env node
/**
 * The step after `npm run export:runs`:
 *
 *   npm run export:runs && npm run runs:split -w @lumina/agent
 *
 * 1. Moves every run in runs/ that did not end as `done` into runs/failing/. Rule A2 fails
 *    any run in runs/ that is not `done`, and rule P1 wants a failing trajectory kept, so
 *    runs/ holds the graded workload and runs/failing/ the failures (DESIGN.md, State).
 *    eval/build-report.mjs reads both folders.
 * 2. Puts `depth` back on each file. The provided exporter writes only the fields
 *    quality/check.mjs reads, but the RunLog contract and AGENTS.md want `depth` on the
 *    file, so it is read back from the `runs` collection.
 *
 * Safe to re-run: files already in runs/failing/ are left alone, and a failing run that is
 * exported again is moved again.
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { config } from 'dotenv';
import { MongoClient } from 'mongodb';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
config({ path: join(ROOT, '.env') });

const runsDir = join(ROOT, 'runs');
const failingDir = join(runsDir, 'failing');

if (!existsSync(runsDir)) {
  console.error('no runs/ folder — run `npm run export:runs` first.');
  process.exit(2);
}
const files = readdirSync(runsDir).filter((f) => f.endsWith('.json'));
if (!files.length) {
  console.log('runs/ has no run logs — nothing to split.');
  process.exit(0);
}

const uri = process.env.MONGODB_URI;
if (!uri) {
  console.error('MONGODB_URI is not set.');
  process.exit(2);
}

const client = new MongoClient(uri, { serverSelectionTimeoutMS: 8000 });
try {
  await client.connect();
  const ids = files.map((f) => f.replace(/\.json$/, ''));
  const depths = new Map(
    (
      await client
        .db(process.env.MONGODB_DB ?? 'lumina')
        .collection('runs')
        .find({ requestId: { $in: ids } }, { projection: { _id: 0, requestId: 1, depth: 1 } })
        .toArray()
    ).map((r) => [r.requestId, r.depth])
  );

  mkdirSync(failingDir, { recursive: true });
  const moved = [];
  const noDepth = [];
  for (const [i, file] of files.entries()) {
    const id = ids[i];
    const path = join(runsDir, file);
    const { tokens, wallClockSec, costUsd, terminated, toolCalls } = JSON.parse(readFileSync(path, 'utf8'));
    const depth = depths.get(id);
    if (!depth) noDepth.push(id);
    writeFileSync(path, JSON.stringify({ tokens, wallClockSec, costUsd, terminated, depth, toolCalls }, null, 2));
    if (terminated !== 'done') {
      renameSync(path, join(failingDir, file));
      moved.push(`${id} (${terminated})`);
    }
  }

  console.log(`${files.length - moved.length} run(s) stay in runs/`);
  console.log(`${moved.length} moved to runs/failing/${moved.length ? `: ${moved.join(', ')}` : ''}`);
  if (noDepth.length) console.warn(`no depth in Mongo for: ${noDepth.join(', ')}`);
} finally {
  await client.close();
}
