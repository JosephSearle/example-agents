# milvus

Self-hosted Milvus standalone, alongside Attu to browse it, wired into the root
`docker-compose.yml` the same way `packages/mlflow-server` is: its own package directory, its
own section of `docker-compose.yml`, started as part of the same stack as Postgres and MLflow.

Milvus standalone doesn't need a custom-built image the way `mlflow-server` does (no self-hosting
setup instructions to bake in — the official image is a complete standalone server), so this
package is documentation plus compose wiring, not a `Dockerfile`. If a future pattern needs
custom Milvus config (a `milvus.yaml` override, a custom analyzer, etc.), that file belongs here
so it stays alongside the service it configures, the same way `infra/postgres/init.sql` sits
next to the `postgres` service.

## Why three containers for one vector database

Milvus standalone still depends on two backing services it doesn't bundle:

- **etcd** — stores Milvus's own metadata (collection schemas, segment locations). Not your
  application data.
- **MinIO** — S3-compatible object storage for the actual vector/scalar segment files. Milvus
  standalone always needs an object store backing it, even single-node; MinIO is the
  batteries-included default (swap for real S3/GCS/Azure Blob in a production deployment, same
  pattern as `MLFLOW_ARTIFACT_ROOT`).

This mirrors `~/milvus-standalone` (the reference local setup this package is based on) exactly:
`etcd` + `minio` + `standalone`, same images and startup flags. The one deliberate difference is
storage: the reference setup bind-mounts `./volumes/*` into the repo directory itself; this
package uses named Docker volumes (`milvus-etcd-data`, `milvus-minio-data`, `milvus-data`)
instead, to match how `postgres-data` and `mlflow-artifacts` are already declared in
`docker-compose.yml` — nothing Milvus-related ends up half-committed to git the way a bind-mounted
`volumes/` folder can.

## Attu

[Attu](https://github.com/zilliztech/attu) is Zilliz's own web UI for Milvus — browse databases,
collections, schemas, and entities, and run ad hoc vector/scalar queries, without writing PyMilvus
code. It only needs one thing: `MILVUS_URL` pointing at the standalone server's gRPC port
(`milvus-standalone:19530` inside the compose network).

## Running it

```bash
docker compose up -d milvus-etcd milvus-minio milvus-standalone attu
open http://localhost:3000   # Attu
```

Or just `docker compose up -d` — like `postgres` and `mlflow`, these services have no profile, so
they start with the rest of the core stack. Only `agents/*` services sit behind the `agents`
profile (see the root `docker-compose.yml` comment on `react-agent`).

Milvus's own client port is `19530` (gRPC, what PyMilvus/`langchain-milvus` connect to) and `9091`
(HTTP metrics/health, what the compose healthcheck uses) — both published to the host so you can
point a local script or notebook at `localhost:19530` without going through Attu at all.

## Version pins

Matches the reference local setup (`~/milvus-standalone/docker-compose.yml`) so this repo behaves
the same as the environment it was copied from:

- `quay.io/coreos/etcd:v3.5.25`
- `minio/minio:RELEASE.2024-12-18T13-15-44Z`
- `milvusdb/milvus:v3.0-beta`
- `zilliz/attu:v2.5.6` (pinned rather than `:latest`, per the SDLC standard the rest of this repo
  follows — see `docs/decisions/0001-tech-stack.md`)

`v3.0-beta` is a pre-release tag, carried over unchanged from the reference setup rather than
substituted for a stable `v2.x` release — worth revisiting once Milvus 3.0 reaches GA, since a
beta tag can move or be pulled upstream without the usual stability guarantees of a versioned
release.

## No agent uses this yet

`agents_common.config.Settings` now exposes `milvus_uri` (default `http://localhost:19530`, or
`http://milvus-standalone:19530` inside docker-compose) so a future RAG-pattern agent can pick it
up the same way `postgres_uri` and `mlflow_tracking_uri` already work — typed, one place to change
it, no agent hand-assembling a connection string. No example agent in this repo talks to Milvus
yet; when one does, it'll depend on `pymilvus` and/or `langchain-milvus` in its own
`pyproject.toml`, not in `agents-common` (same reasoning as `agents-common` not depending on
`langchain-anthropic` — provider/store-specific clients live with the code that uses them).
