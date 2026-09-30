/**
 * Threads and their messages. A thread belongs to the X-User-Id that created it, and every
 * read is filtered by that id, so another user's thread is indistinguishable from one that
 * does not exist: both are a 404.
 *
 * Each message is its own document, inserted once and never rewritten, so two asks on one
 * thread at the same time each add their own rows and cannot overwrite each other. An
 * answer points at the question it answers (`replyTo`), which keeps the pairs straight when
 * two runs finish out of order.
 *
 * What is saved (DESIGN.md, State):
 *   - the question, when the ask is accepted, before anything streams;
 *   - the answer with its sources and done event, when the run ends as done or cap. A
 *     capped answer carries done.terminated = "cap", and its text says it stopped early;
 *   - nothing for a run that errored, so no half-written answer can be read back as one.
 *
 * History: a follow-up sees every earlier turn, but only the latest answers go in nearly
 * whole. Older ones are cut to their opening, where the direct answer is, so a long thread
 * adds a little per turn instead of a whole answer per turn.
 */
import type Anthropic from '@anthropic-ai/sdk';
import { randomUUID } from 'node:crypto';
import {
  COLLECTIONS,
  GetThreadResponse,
  ListThreadsResponse,
  MessageDoc,
  ThreadId,
  newId,
  type DoneEvent,
  type Source,
  type ThreadDoc
} from '@lumina/contract';
import { db } from './db.js';

const DEFAULT_TITLE = 'New thread';
/** A thread created without a title takes the first question's opening as one. */
const TITLE_CHARS = 80;
/** The thread list is a sidebar, not an archive. */
const LIST_LIMIT = 100;

/** Answers kept nearly whole: a follow-up almost always leans on the last one or two. */
const RECENT_ANSWERS = 2;
const RECENT_ANSWER_CHARS = 2000;
/** Everything older is cut to about its first sentence or two, where the direct answer is. */
const OLDER_ANSWER_CHARS = 200;
const QUESTION_CHARS = 500;
/** Source titles shown with a recent answer, so "the second one" still means something. */
const HISTORY_SOURCE_TITLES = 6;

type StoredThread = ThreadDoc & { untitled?: boolean };
type StoredMessage = MessageDoc & { replyTo?: string };

export type SavedAnswer = { answerId: string; content: string; sources: Source[]; done: DoneEvent };

const threads = async () => (await db()).collection<StoredThread>(COLLECTIONS.threads);
const messages = async () => (await db()).collection<StoredMessage>(COLLECTIONS.messages);

/** Time-ordered, so ties on createdAt still sort in insertion order. */
const messageId = () => `msg_${Date.now().toString(36)}${randomUUID().slice(0, 8)}`;

const iso = (d: string | Date) => new Date(d).toISOString();

// ---------------------------------------------------------------- threads

export async function createThread(userId: string, title?: string): Promise<string> {
  const _id = newId('thr');
  await (await threads()).insertOne({
    _id,
    userId,
    title: title ?? DEFAULT_TITLE,
    ...(title ? {} : { untitled: true }),
    createdAt: new Date()
  });
  return _id;
}

export async function listThreads(userId: string): Promise<ListThreadsResponse> {
  const rows = await (await threads())
    .find({ userId })
    .sort({ createdAt: -1 })
    .limit(LIST_LIMIT)
    .toArray();
  return ListThreadsResponse.parse({
    threads: rows.map((t) => ({ threadId: t._id, title: t.title, createdAt: iso(t.createdAt) }))
  });
}

/** Null for a malformed id, an unknown one, and another user's: the caller answers 404 to all three. */
export async function findThread(userId: string, threadId: string): Promise<StoredThread | null> {
  if (!ThreadId.safeParse(threadId).success) return null;
  return (await threads()).findOne({ _id: threadId, userId });
}

export async function getThread(userId: string, threadId: string): Promise<GetThreadResponse | null> {
  const thread = await findThread(userId, threadId);
  if (!thread) return null;
  const rows = await (await messages()).find({ threadId, userId }).sort({ createdAt: 1, _id: 1 }).toArray();
  return GetThreadResponse.parse({
    threadId: thread._id,
    title: thread.title,
    messages: rows.map((m) => ({
      role: m.role,
      content: m.content,
      ...(m.role === 'assistant' ? { sources: m.sources, answerId: m.answerId, done: m.done } : {}),
      createdAt: iso(m.createdAt)
    }))
  });
}

// ---------------------------------------------------------------- one turn

/**
 * Saves the question and loads the history it follows, in parallel. The history is read
 * as it stands now: an answer another ask on this thread is still writing is not in it yet.
 */
export async function beginTurn(
  userId: string,
  threadId: string,
  query: string
): Promise<{ questionId: string; history: Anthropic.MessageParam[]; earlierQuestions: string[] }> {
  const questionId = messageId();
  const col = await messages();
  const question = MessageDoc.parse({
    _id: questionId,
    threadId,
    userId,
    role: 'user',
    content: query,
    createdAt: new Date()
  });
  const [rows] = await Promise.all([
    col.find({ threadId, userId, _id: { $ne: questionId } }).sort({ createdAt: 1, _id: 1 }).toArray(),
    col.insertOne(question),
    // Only the first question names an untitled thread; the filter makes that atomic.
    threads().then((t) =>
      t.updateOne(
        { _id: threadId, userId, untitled: true },
        { $set: { title: clip(query.replace(/\s+/g, ' ').trim(), TITLE_CHARS) }, $unset: { untitled: '' } }
      )
    )
  ]);
  return {
    questionId,
    history: historyMessages(rows),
    earlierQuestions: rows.filter((r) => r.role === 'user').map((r) => r.content)
  };
}

/** Called before `done` is sent, so a failed save can still end the stream as an error. */
export async function saveAnswer(userId: string, threadId: string, replyTo: string, a: SavedAnswer): Promise<void> {
  const doc = MessageDoc.parse({
    _id: messageId(),
    threadId,
    userId,
    role: 'assistant',
    content: a.content,
    answerId: a.answerId,
    sources: a.sources,
    done: a.done,
    createdAt: new Date()
  });
  await (await messages()).insertOne({ ...doc, replyTo });
}

// ---------------------------------------------------------------- history → prompt

function clip(text: string, max: number): string {
  if (text.length <= max) return text;
  const cut = text.slice(0, max);
  // End on a sentence or line if one is reasonably close; mid-word otherwise.
  const end = Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('\n'));
  return `${(end > max / 2 ? cut.slice(0, end + 1) : cut).trimEnd()} […]`;
}

/**
 * An old answer's [n] pointed at that request's sources. Left in, it would read as a
 * citation of whatever is numbered n now, so it goes.
 */
const stripCitations = (text: string) =>
  text
    .replace(/\[\d{1,3}\]/g, '')
    .replace(/[ \t]+([.,;:!?])/g, '$1')
    .replace(/[ \t]{2,}/g, ' ')
    .trim();

function renderAnswer(a: StoredMessage, recent: boolean): string {
  const parts: string[] = [];
  if (a.done?.terminated === 'cap') parts.push('(This answer stopped early at its research limit.)');
  parts.push(clip(stripCitations(a.content), recent ? RECENT_ANSWER_CHARS : OLDER_ANSWER_CHARS) || '(empty answer)');
  if (recent && a.sources.length) {
    const titles = a.sources.slice(0, HISTORY_SOURCE_TITLES).map((s) => s.title);
    parts.push(`(Read for this answer, not citable now: ${titles.join('; ')})`);
  }
  return parts.join('\n\n');
}

/**
 * Every earlier question, each followed by its answer, as alternating turns. A question
 * with no saved answer (its run errored, or is still running) gets a placeholder, so the
 * turns still alternate and the model is not left guessing what was said.
 */
export function historyMessages(rows: StoredMessage[]): Anthropic.MessageParam[] {
  const answers = new Map<string, StoredMessage>();
  for (const r of rows) if (r.role === 'assistant' && r.replyTo) answers.set(r.replyTo, r);
  const questions = rows.filter((r) => r.role === 'user');
  const recent = new Set(
    questions
      .filter((q) => answers.has(q._id))
      .slice(-RECENT_ANSWERS)
      .map((q) => q._id)
  );

  return questions.flatMap((q): Anthropic.MessageParam[] => {
    const a = answers.get(q._id);
    return [
      { role: 'user', content: clip(q.content, QUESTION_CHARS) },
      { role: 'assistant', content: a ? renderAnswer(a, recent.has(q._id)) : '(No answer was saved for this question.)' }
    ];
  });
}
