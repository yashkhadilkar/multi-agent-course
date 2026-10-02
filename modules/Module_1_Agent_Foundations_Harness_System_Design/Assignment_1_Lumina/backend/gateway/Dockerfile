# syntax=docker/dockerfile:1
# LUMINA gateway (fly.gateway.toml). Build context is the assignment root, filtered by
# .dockerignore. Only explicit paths are copied: no .env, and no provider key ever.

FROM node:22-slim AS build
WORKDIR /app
# Every workspace manifest, so npm ci sees the lockfile's whole workspace set.
COPY package.json package-lock.json tsconfig.base.json ./
COPY packages/contract/package.json packages/contract/
COPY backend/agent/package.json backend/agent/
COPY backend/gateway/package.json backend/gateway/
COPY web/package.json web/
RUN npm ci -w @lumina/contract -w @lumina/gateway
COPY packages/contract packages/contract
COPY backend/gateway backend/gateway
RUN npm run build -w @lumina/contract && npm run build -w @lumina/gateway

FROM node:22-slim
ENV NODE_ENV=production
WORKDIR /app
COPY package.json package-lock.json ./
COPY packages/contract/package.json packages/contract/
COPY backend/agent/package.json backend/agent/
COPY backend/gateway/package.json backend/gateway/
COPY web/package.json web/
RUN npm ci --omit=dev -w @lumina/contract -w @lumina/gateway && npm cache clean --force
COPY --from=build /app/packages/contract/dist packages/contract/dist
COPY --from=build /app/backend/gateway/dist backend/gateway/dist
# The eval report, at the path env.ts resolves from this WORKDIR (../../reports/report.json).
# .dockerignore lets only report.json through, so before the eval has run this is an empty
# directory and GET /evals/report.json answers 404 until the next deploy carries the file.
COPY reports/ reports/
WORKDIR /app/backend/gateway
USER node
CMD ["node", "dist/index.js"]
