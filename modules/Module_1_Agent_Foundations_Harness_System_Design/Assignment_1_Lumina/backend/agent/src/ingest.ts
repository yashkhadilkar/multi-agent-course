/**
 * Parsing and chunking for the jobs worker. Pure functions: bytes in, located chunks out.
 *
 * Every chunk carries a locator, because a document citation is only as good as the place
 * it points at (SPEC 5.4):
 *   PDF       {page}            one segment per page, and no chunk crosses a page break,
 *                               so "p. N" is always exact
 *   Markdown  {heading, line}   one segment per ATX section, no chunk crosses a heading; the
 *                               line keeps two chunks of one long section apart
 *   Text      {line}            the line the chunk starts on
 *
 * Chunk size and overlap come from config (CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS), and a
 * chunk ends at the best break inside its window: a paragraph, then a sentence, then a
 * line, then a word.
 */
import { getDocument } from 'pdfjs-dist/legacy/build/pdf.mjs';
import type { ACCEPTED_UPLOAD_TYPES, Locator } from '@lumina/contract';

export type AcceptedType = (typeof ACCEPTED_UPLOAD_TYPES)[number];

/** A stretch of the document that no chunk may cross, with the locator its chunks inherit. */
type Segment = { text: string; page?: number; heading?: string; startLine?: number };
export type Piece = { ord: number; text: string; locator: Locator };
export type Parsed = { segments: Segment[]; pages?: number };

export async function parseDocument(bytes: Buffer, mimeType: AcceptedType): Promise<Parsed> {
  if (mimeType === 'application/pdf') return parsePdf(bytes);
  const text = bytes.toString('utf8').replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n');
  return mimeType === 'text/markdown' ? parseMarkdown(text) : { segments: [{ text, startLine: 1 }] };
}

// ---------------------------------------------------------------- PDF

async function parsePdf(bytes: Buffer): Promise<Parsed> {
  // verbosity 0: text extraction needs no font files, so pdfjs's font warnings are noise.
  const pdf = await getDocument({ data: new Uint8Array(bytes), isEvalSupported: false, verbosity: 0 }).promise;
  try {
    const segments: Segment[] = [];
    for (let page = 1; page <= pdf.numPages; page++) {
      const content = await (await pdf.getPage(page)).getTextContent();
      const text = content.items
        .map((item) => ('str' in item ? item.str + (item.hasEOL ? '\n' : '') : ''))
        .join('')
        .replace(/\s+/g, ' ')
        .trim();
      if (text) segments.push({ text, page });
    }
    return { segments, pages: pdf.numPages };
  } finally {
    await pdf.destroy();
  }
}

// ---------------------------------------------------------------- Markdown

const HEADING = /^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$/;
const FENCE = /^ {0,3}(```|~~~)/;

function parseMarkdown(text: string): Parsed {
  const lines = text.split('\n');
  const sections: { heading?: string; startLine: number; lines: string[]; body: boolean }[] = [];
  let current: (typeof sections)[number] = { startLine: 1, lines: [], body: false };
  let inFence = false;

  lines.forEach((line, i) => {
    if (FENCE.test(line)) inFence = !inFence;
    const heading = inFence ? null : HEADING.exec(line);
    if (heading) {
      // A heading straight after another (a title over its first section) joins that
      // section instead of becoming a chunk of nothing but a heading.
      if (current.body) {
        sections.push(current);
        current = { startLine: i + 1, lines: [], body: false };
      }
      current.heading = heading[1]!.replace(/[*_`]/g, '');
    } else if (line.trim()) {
      current.body = true;
    }
    current.lines.push(line);
  });
  sections.push(current);

  return {
    segments: sections.map((s) => ({
      text: s.lines.join('\n'),
      startLine: s.startLine,
      ...(s.heading ? { heading: s.heading } : {})
    }))
  };
}

// ---------------------------------------------------------------- chunking

/** Breaks to end a chunk on, best first. A break must land in the back half of the window. */
const BREAKS = [/\n\s*\n/g, /[.!?]["')\]]?\s/g, /\n/g, /\s/g];

function breakPoint(text: string, start: number, end: number): number {
  const window = text.slice(start, end);
  const min = Math.floor(window.length / 2);
  for (const re of BREAKS) {
    let best = -1;
    for (const m of window.matchAll(re)) {
      const at = m.index + m[0].length;
      if (at >= min && at < window.length) best = at;
    }
    if (best > 0) return start + best;
  }
  return end;
}

const skipSpace = (text: string, i: number) => {
  while (i < text.length && /\s/.test(text[i]!)) i++;
  return i;
};

/** [start, end) ranges of at most `size` characters, each starting `overlap` back from the last end. */
export function splitRanges(text: string, size: number, overlap: number): [number, number][] {
  const ranges: [number, number][] = [];
  let start = skipSpace(text, 0);
  while (start < text.length) {
    let end = Math.min(start + size, text.length);
    if (end < text.length) end = breakPoint(text, start, end);
    ranges.push([start, end]);
    if (end >= text.length) break;
    // Back up by the overlap, then forward to the next word so no chunk opens mid-word.
    let next = Math.max(end - overlap, start + 1);
    if (/\S/.test(text[next - 1]!)) {
      const space = text.slice(next, end).search(/\s/);
      next = space === -1 ? end : next + space;
    }
    start = skipSpace(text, next);
  }
  return ranges;
}

export function chunkSegments(segments: Segment[], size: number, overlap: number): Piece[] {
  const pieces: Piece[] = [];
  for (const seg of segments) {
    for (const [start, end] of splitRanges(seg.text, size, overlap)) {
      const text = seg.text.slice(start, end).trim();
      if (!text) continue;
      const line = seg.startLine === undefined ? undefined : seg.startLine + countNewlines(seg.text, start);
      const locator: Locator = {
        ...(seg.page !== undefined ? { page: seg.page } : {}),
        ...(seg.heading !== undefined ? { heading: seg.heading } : {}),
        ...(line !== undefined ? { line } : {})
      };
      pieces.push({ ord: pieces.length, text, locator });
    }
  }
  return pieces;
}

function countNewlines(text: string, end: number): number {
  let n = 0;
  for (let i = 0; i < end; i++) if (text[i] === '\n') n++;
  return n;
}
