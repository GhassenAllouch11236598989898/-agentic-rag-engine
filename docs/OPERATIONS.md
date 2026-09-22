# Running and maintaining Billing Copilot

## Storage and access

Each deployment is one company's shared knowledge workspace. All authorized agents can query all knowledge documents. Admins can upload and delete documents. Conversations, saved drafts, and feedback are private to the access-key identity. There is no organization-level multi-tenant isolation: deploy separate instances and databases for separate clients.

Demo documents, conversations, and feedback live in `.data/billing-demo.sqlite3`. In connected mode, knowledge lives in PostgreSQL; conversations and drafts live in `.data/billing-live.sqlite3`. Persist and back up both stores. A process-local limiter is intended for one application worker; put a shared rate limiter at the gateway before scaling to multiple workers.

No-key demo access is limited to local loopback clients. Connected mode and APP_ENV=production require configured access keys. Set up HTTPS at the reverse proxy before sharing a client deployment. Do not expose the unauthenticated loopback demo through a forwarding proxy.

Access keys are configured in the environment and held only in browser memory. Refreshing the page requires signing in again. Change the relevant environment key and restart to revoke an identity's access. Store secrets outside version control.

## Existing database upgrade

Application startup uses a transaction to apply `sql/002_retrieval.sql` once, recorded in `relay_migrations`. It creates the full-text index and replaces weighted raw-score search with reciprocal rank fusion. Neither the initial schema nor the migration drops data tables.

The supplied schema uses 384-dimensional vectors. Set EMBEDDING_DIM=384 and an embedding model with matching output. OpenAI embeddings request the configured dimension. Ollama defaults may emit a different dimension; choose a compatible model/configuration or migrate the schema before use. Changing the embedding model requires re-indexing all documents even when the vector dimension is unchanged. Do not mix embedding spaces in one collection.

## Files

Uploads accept PDF, Markdown, or UTF-8 text, up to 10 MB, 200 PDF pages, and 2 million extracted characters. Scanned PDFs require OCR before upload. PDF extraction is performed locally; connected embeddings and answer generation can send document text to the selected external provider. Duplicate content is detected before storage. Only an administrator can add or remove knowledge.

## Reliability checks

`GET /health` checks database availability. It does not perform model inference. Generate a draft in a private test workspace to validate the selected provider.

The automated tests exercise local persistence, uploads, real source provenance, roles, ownership, failure reporting, and billing behavior. The connected generation tests use a deterministic test model; they do not contact the user's AI provider. A real PostgreSQL/provider smoke test remains a deployment gate for each connected client setup.

Review answers against source documents, especially when policies conflict or when a request concerns a specific invoice or transaction. Reference validation checks IDs, not factual entailment. The draft remains subject to human review after editing.

## Troubleshooting

- **Page will not open:** start the server with `start-demo.ps1` and keep that terminal open.
- **Port already in use:** use `./start-demo.ps1 -Port 8001`, or stop the previously started application you recognize.
- **Old interface appears:** reload the browser. Styles and scripts are served from `/static/`.
- **Connection needed:** open Settings and enter the configured workspace key, or check server logs.
- **Scanned or corrupt PDF:** export searchable text or run OCR, then upload again.
- **Provider or database error:** the request fails explicitly; inspect server logs. An outage is not reported as an empty knowledge base.
