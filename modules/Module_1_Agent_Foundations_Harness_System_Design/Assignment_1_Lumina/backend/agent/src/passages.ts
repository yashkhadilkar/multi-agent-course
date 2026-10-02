/**
 * Trimming a fetched page down to what the model reads, and choosing the snippet a
 * citation rests on. Both are cut out of the fetched text — whitespace collapsed, words
 * untouched — never written by the model, so the grounding check can find them in the page.
 *
 * Scoring is plain term overlap weighted by rarity within the page. No model call: it
 * runs on every fetch inside the quick gear's latency budget.
 */

const STOPWORDS = new Set(
  (
    'a an and are as at be but by can do does for from has have how i if in into is it its of on or ' +
    'so than that the their then there these this to was were what when where which who why will with ' +
    'you your about after all also any been before being between both each more most not only other ' +
    'our out over same should some such them they through under up very we would'
  ).split(' ')
);

/** Blocks shorter than this are headings, nav and captions: not worth reading or citing. */
const MIN_PASSAGE_WORDS = 8;
/** Long paragraphs are split into sentence windows of about this many words. */
const WINDOW_WORDS = 90;
/** Snippet length. Above 12 words so the bench's 12-token window can match inside it. */
const SNIPPET_MIN_WORDS = 20;
const SNIPPET_MAX_WORDS = 60;

export function terms(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

const wordCount = (s: string) => s.split(' ').filter(Boolean).length;
const sentences = (s: string) => s.split(/(?<=[.!?])\s+(?=["'([A-Z0-9])/);

type Passage = { text: string; ord: number; score: number };

/** Paragraphs, with long ones cut into sentence windows. Each is a contiguous span. */
function split(text: string): string[] {
  const out: string[] = [];
  for (const raw of text.split(/\n+/)) {
    const block = raw.replace(/\s+/g, ' ').trim();
    const n = wordCount(block);
    if (n < MIN_PASSAGE_WORDS) continue;
    if (n <= WINDOW_WORDS * 1.5) {
      out.push(block);
      continue;
    }
    let cur: string[] = [];
    let words = 0;
    for (const s of sentences(block)) {
      cur.push(s);
      words += wordCount(s);
      if (words >= WINDOW_WORDS) {
        out.push(cur.join(' '));
        cur = [];
        words = 0;
      }
    }
    if (words >= MIN_PASSAGE_WORDS) out.push(cur.join(' '));
  }
  return out;
}

function score(passages: string[], query: string[]): Passage[] {
  const q = new Set(query);
  const bags = passages.map((p) => terms(p));
  const df = new Map<string, number>();
  for (const bag of bags) for (const t of new Set(bag)) if (q.has(t)) df.set(t, (df.get(t) ?? 0) + 1);
  const idf = (t: string) => Math.log(1 + passages.length / (df.get(t) ?? passages.length));

  return passages.map((text, ord) => {
    const bag = bags[ord] ?? [];
    const unique = new Set(bag.filter((t) => q.has(t)));
    let s = 0;
    for (const t of unique) s += idf(t);
    // A small bonus for repeated hits, so a paragraph about the topic beats one mention.
    s += 0.1 * bag.filter((t) => q.has(t)).length;
    return { text, ord, score: s };
  });
}

/**
 * Joins soft line breaks: a single newline inside a paragraph becomes a space, a blank line
 * stays a paragraph break. Hard-wrapped source (Markdown wrapped at ~90 columns) would
 * otherwise split into one-line "paragraphs", and a snippet would be a line fragment.
 */
export const unwrap = (text: string) => text.replace(/([^\n])\n(?!\s*\n)/g, '$1 ');

export type Selection = {
  /** What the model reads: the best passages, in page order, within the word budget. */
  passages: string[];
  totalPassages: number;
  /** The one passage a citation of this page points at: candidates[0]. */
  snippet: string;
  /**
   * Every snippet a citation of this page could rest on, best first: windows of the passages
   * the model reads, so whichever is chosen is text the model actually saw. verify.ts takes
   * the first that also appears in the page's plain HTML.
   */
  candidates: string[];
};

export function selectPassages(text: string, query: string, maxWords: number): Selection | null {
  const all = split(text);
  if (!all.length) return null;
  const scored = score(all, terms(query));
  const ranked = [...scored].sort((a, b) => b.score - a.score || a.ord - b.ord);

  const kept: Passage[] = [];
  let budget = maxWords;
  for (const p of ranked) {
    const n = wordCount(p.text);
    if (n > budget) continue;
    kept.push(p);
    budget -= n;
    if (budget < MIN_PASSAGE_WORDS) break;
  }
  if (!kept.length && ranked[0]) kept.push({ ...ranked[0], text: ranked[0].text.split(' ').slice(0, maxWords).join(' ') });

  // Candidates come from the kept passages, best-scoring first, and long enough ones first,
  // so the snippet is never shorter than the grounding check's window when it can be helped.
  const byScore = [...kept].sort((a, b) => b.score - a.score || a.ord - b.ord);
  const long = byScore.filter((p) => wordCount(p.text) >= SNIPPET_MIN_WORDS);
  const candidates = [...new Set([...long, ...byScore.filter((p) => !long.includes(p))].flatMap((p) => windowsOf(p.text, query)))];
  kept.sort((a, b) => a.ord - b.ord);
  return { passages: kept.map((p) => p.text), totalPassages: all.length, snippet: candidates[0]!, candidates };
}

/** The passage's best snippet, then a window starting at each of its other sentences. */
function windowsOf(passage: string, query: string): string[] {
  const best = snippetOf(passage, query);
  if (wordCount(passage) <= SNIPPET_MAX_WORDS) return [best];
  const ss = sentences(passage);
  const out = [best];
  for (let start = 0; start < ss.length; start++) {
    const words: string[] = [];
    for (const s of ss.slice(start)) {
      words.push(...s.split(' ').filter(Boolean));
      if (words.length >= SNIPPET_MIN_WORDS) break;
    }
    if (words.length < SNIPPET_MIN_WORDS) break;
    out.push(words.slice(0, SNIPPET_MAX_WORDS).join(' '));
  }
  return out;
}

/**
 * The best-scoring sentence plus the ones after it, until the snippet is long enough to
 * verify. Always a contiguous run of the passage's own words.
 */
function snippetOf(passage: string, query: string): string {
  const ss = sentences(passage);
  if (wordCount(passage) <= SNIPPET_MAX_WORDS) return passage;
  const q = new Set(terms(query));
  let start = 0;
  let bestScore = -1;
  ss.forEach((s, i) => {
    const hits = new Set(terms(s).filter((t) => q.has(t))).size;
    if (hits > bestScore) {
      bestScore = hits;
      start = i;
    }
  });
  const words: string[] = [];
  for (const s of ss.slice(start)) {
    words.push(...s.split(' ').filter(Boolean));
    if (words.length >= SNIPPET_MIN_WORDS) break;
  }
  // The best sentence may sit at the end of the passage; borrow backwards to reach length.
  for (let i = start - 1; words.length < SNIPPET_MIN_WORDS && i >= 0; i--) {
    words.unshift(...(ss[i] ?? '').split(' ').filter(Boolean));
  }
  return words.slice(0, SNIPPET_MAX_WORDS).join(' ');
}
