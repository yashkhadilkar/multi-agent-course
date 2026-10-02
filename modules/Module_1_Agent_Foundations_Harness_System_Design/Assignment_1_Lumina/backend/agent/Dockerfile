# syntax=docker/dockerfile:1
# LUMINA agent service and jobs worker: one image, two Fly process groups (fly.agent.toml).
# Build context is the assignment root, filtered by .dockerignore. Only explicit paths are
# copied, so nothing local (.env above all) can land in the image; keys arrive as Fly secrets.

FROM node:22-slim AS build
WORKDIR /app
# Every workspace manifest, so npm ci sees the lockfile's whole workspace set.
COPY package.json package-lock.json tsconfig.base.json ./
COPY packages/contract/package.json packages/contract/
COPY backend/agent/package.json backend/agent/
COPY backend/gateway/package.json backend/gateway/
COPY web/package.json web/
RUN npm ci -w @lumina/contract -w @lumina/agent
COPY packages/contract packages/contract
COPY backend/agent backend/agent
RUN npm run build -w @lumina/contract && npm run build -w @lumina/agent

FROM node:22-slim
ENV NODE_ENV=production
WORKDIR /app
COPY package.json package-lock.json ./
COPY packages/contract/package.json packages/contract/
COPY backend/agent/package.json backend/agent/
COPY backend/gateway/package.json backend/gateway/
COPY web/package.json web/
RUN npm ci --omit=dev -w @lumina/contract -w @lumina/agent && npm cache clean --force
COPY --from=build /app/packages/contract/dist packages/contract/dist
COPY --from=build /app/backend/agent/dist backend/agent/dist
WORKDIR /app/backend/agent
USER node
# fly.agent.toml [processes] picks the entry point per group: dist/index.js or dist/worker.js.
CMD ["node", "dist/index.js"]
