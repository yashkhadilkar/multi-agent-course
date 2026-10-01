/**
 * Spaces and their documents. A Space belongs to the X-User-Id that created it and every read
 * is filtered by that id, so another user's Space is a 404, the same as one that does not exist.
 *
 * An upload does only what has to happen before the 202 (SPEC 5.4): the file goes into GridFS,
 * then a `pending` document row and its `index_document` job are inserted together. Parsing,
 * chunking, embedding and the read-your-write probe all happen on the jobs worker, which is the
 * only writer of a document's status from then on.
 */
import { GridFSBucket, ObjectId } from 'mongodb';
import { extname } from 'node:path';
import { finished } from 'node:stream/promises';
import type express from 'express';
import multer from 'multer';
import {
  ACCEPTED_UPLOAD_TYPES,
  COLLECTIONS,
  DocumentDoc,
  GRIDFS_BUCKETS,
  ListDocumentsResponse,
  ListSpacesResponse,
  MAX_UPLOAD_BYTES,
  SpaceId,
  UploadDocumentResponse,
  newId,
  type CreateSpaceResponse,
  type SpaceDoc
} from '@lumina/contract';
import { db } from './db.js';
import type { AcceptedType } from './ingest.js';
import { indexJob, jobs } from './jobs.js';

type StoredSpace = Omit<SpaceDoc, 'createdAt'> & { createdAt: Date };
export type StoredDocument = Omit<DocumentDoc, 'createdAt'> & { createdAt: Date };

const spaces = async () => (await db()).collection<StoredSpace>(COLLECTIONS.spaces);
export const documents = async () => (await db()).collection<StoredDocument>(COLLECTIONS.documents);

// ---------------------------------------------------------------- GridFS

/**
 * One bucket per process. A bucket checks its indexes on its first write, three round trips
 * to Atlas; a fresh bucket per request would pay that on every upload, inside the 300 ms.
 */
let bucket: GridFSBucket | null = null;
const uploads = async () => (bucket ??= new GridFSBucket(await db(), { bucketName: GRIDFS_BUCKETS.uploads }));

/** Stores the bytes and returns the file's id as hex, the form `documents.fileId` holds. */
export async function putFile(bytes: Buffer, filename: string, metadata: Record<string, unknown>): Promise<string> {
  const stream = (await uploads()).openUploadStream(filename, { metadata });
  stream.end(bytes);
  await finished(stream);
  return stream.id.toHexString();
}

export async function readFile(fileId: string): Promise<Buffer> {
  const parts: Buffer[] = [];
  for await (const part of (await uploads()).openDownloadStream(new ObjectId(fileId))) parts.push(part as Buffer);
  return Buffer.concat(parts);
}

/** Deletes every file whose metadata matches. Matching nothing is not an error. */
export async function deleteFiles(metadata: Record<string, unknown>): Promise<number> {
  const bucket = await uploads();
  const filter = Object.fromEntries(Object.entries(metadata).map(([k, v]) => [`metadata.${k}`, v]));
  const files = await bucket.find(filter, { projection: { _id: 1 } }).toArray();
  await Promise.all(files.map((f) => bucket.delete(f._id)));
  return files.length;
}

// ---------------------------------------------------------------- spaces

export async function createSpace(userId: string, name: string): Promise<CreateSpaceResponse> {
  const _id = newId('spc');
  await (await spaces()).insertOne({ _id, userId, name, createdAt: new Date() });
  return { spaceId: _id, name };
}

export async function listSpaces(userId: string): Promise<ListSpacesResponse> {
  const rows = await (await spaces()).find({ userId }).sort({ createdAt: -1 }).toArray();
  return ListSpacesResponse.parse({
    spaces: rows.map((s) => ({ spaceId: s._id, name: s.name, createdAt: s.createdAt.toISOString() }))
  });
}

/** Null for a malformed id, an unknown one, and another user's: the caller answers 404 to all three. */
export async function findSpace(userId: string, spaceId: string): Promise<StoredSpace | null> {
  if (!SpaceId.safeParse(spaceId).success) return null;
  return (await spaces()).findOne({ _id: spaceId, userId });
}

// ---------------------------------------------------------------- receiving the file

export type Upload = { bytes: Buffer; filename: string; mimeType: AcceptedType };
export type Refusal = { status: 400 | 413; error: string };

const receive = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_UPLOAD_BYTES, files: 1 }
}).single('file');

const BY_EXTENSION: Record<string, AcceptedType> = {
  '.pdf': 'application/pdf',
  '.md': 'text/markdown',
  '.markdown': 'text/markdown',
  '.txt': 'text/plain'
};

/**
 * The declared type when it is one of the contract's. A client that has no type for the
 * extension (curl sends .md as application/octet-stream) sends a generic one, and only then
 * does the extension decide. Anything else is refused, whatever the extension says.
 */
function acceptedType(declared: string, filename: string): AcceptedType | null {
  const type = declared.split(';')[0]!.trim().toLowerCase();
  if ((ACCEPTED_UPLOAD_TYPES as readonly string[]).includes(type)) return type as AcceptedType;
  if (type === 'text/x-markdown') return 'text/markdown';
  if (type === '' || type === 'application/octet-stream') return BY_EXTENSION[extname(filename).toLowerCase()] ?? null;
  return null;
}

/** Reads the multipart `file` field. A refusal is the client's fault: 413 too large, else 400. */
export async function receiveUpload(req: express.Request, res: express.Response): Promise<Upload | Refusal> {
  try {
    await new Promise<void>((resolve, reject) => receive(req, res, (err: unknown) => (err ? reject(err) : resolve())));
  } catch (err) {
    if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
      return { status: 413, error: `file is larger than ${MAX_UPLOAD_BYTES / 1024 / 1024} MB` };
    }
    return { status: 400, error: `could not read the upload: ${(err as Error).message}` };
  }
  const file = req.file;
  if (!file) return { status: 400, error: 'multipart field "file" is required' };
  if (!file.size) return { status: 400, error: 'file is empty' };
  const mimeType = acceptedType(file.mimetype, file.originalname);
  if (!mimeType) {
    return { status: 400, error: `unsupported file type "${file.mimetype}": upload a PDF, Markdown, or plain text file` };
  }
  return { bytes: file.buffer, filename: file.originalname, mimeType };
}

// ---------------------------------------------------------------- documents

/**
 * GridFS first, then the document row and the job in parallel: the three writes are the
 * whole of the request path. If the row or the job fails to land, the other two are removed,
 * so there is never a `pending` document with no job to move it.
 */
export async function addDocument(userId: string, spaceId: string, upload: Upload): Promise<UploadDocumentResponse> {
  const docId = newId('doc');
  const fileId = await putFile(upload.bytes, upload.filename, { kind: 'upload', userId, spaceId, docId, contentType: upload.mimeType });

  const doc: StoredDocument = {
    _id: docId,
    spaceId,
    userId,
    title: upload.filename,
    mimeType: upload.mimeType,
    bytes: upload.bytes.length,
    status: 'pending',
    pct: 0,
    fileId,
    createdAt: new Date()
  };
  DocumentDoc.parse(doc);
  const job = indexJob(userId, { docId, spaceId, fileId, mimeType: upload.mimeType, title: upload.filename });

  const [docs, queue] = await Promise.all([documents(), jobs()]);
  try {
    await Promise.all([docs.insertOne(doc), queue.insertOne(job)]);
  } catch (err) {
    await Promise.allSettled([docs.deleteOne({ _id: docId }), queue.deleteOne({ _id: job._id }), deleteFiles({ docId })]);
    throw err;
  }
  return UploadDocumentResponse.parse({ docId, status: 'pending' });
}

export async function listDocuments(userId: string, spaceId: string): Promise<ListDocumentsResponse> {
  const rows = await (await documents()).find({ spaceId, userId }).sort({ createdAt: 1 }).toArray();
  return ListDocumentsResponse.parse({
    documents: rows.map((d) => ({
      docId: d._id,
      title: d.title,
      status: d.status,
      pct: d.pct,
      ...(d.pages !== undefined ? { pages: d.pages } : {}),
      ...(d.chunks !== undefined ? { chunks: d.chunks } : {}),
      ...(d.error ? { error: d.error } : {})
    }))
  });
}
