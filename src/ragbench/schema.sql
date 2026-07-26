-- =====================================================================
-- Schema du banc d'evaluation RAG.
--
-- Trois choix structurants, qui expliquent la forme du reste :
--
-- 1. UN INDEX DE CORPUS PAR (chunking, embedder), pas un par config.
--    Comparer top_k=3 et top_k=10 ne doit pas reindexer 600 documents ;
--    comparer deux strategies de chunking, si. D'ou `corpus_indexes`,
--    identifie par un hash calcule dans PipelineConfig.index_hash().
--
-- 2. PAS D'INDEX ANN (ni ivfflat ni hnsw) SUR LES EMBEDDINGS.
--    Un index approximatif introduit une perte de recall qui se confond
--    avec la qualite du modele d'embedding : on mesurerait l'index, pas le
--    retrieval. Sur un corpus de banc d'essai (~10^4 chunks) le parcours
--    exact coute quelques dizaines de ms. A revoir seulement si un corpus
--    de production arrive.
--
-- 3. `scores` PORTE UNE COLONNE `evaluator`.
--    La meme metrique ("faithfulness") notee par Ragas, par RAGChecker et
--    par un verificateur NLI donne trois lignes comparables. Mesurer le
--    desaccord entre frameworks est l'objet meme de ce projet.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- --------------------------------------------------------------------
-- Couche donnees : corpus et questions
-- --------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS datasets (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    source      TEXT,                       -- 'multihop-rag', 'hotpotqa', 'local'...
    description TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    dataset_id  INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    external_id TEXT NOT NULL,              -- identifiant d'origine du dataset
    title       TEXT,
    body        TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (dataset_id, external_id)
);

CREATE TABLE IF NOT EXISTS corpus_indexes (
    id          SERIAL PRIMARY KEY,
    dataset_id  INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    index_hash  TEXT NOT NULL,
    chunking    JSONB NOT NULL,
    embedder    TEXT NOT NULL,
    dim         INTEGER,
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, index_hash)
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    index_id    INTEGER NOT NULL REFERENCES corpus_indexes(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,           -- position du chunk dans le document
    text        TEXT NOT NULL,
    -- Colonne `vector` sans dimension declaree : elle doit accueillir des
    -- embedders de tailles differentes (768 pour nomic, 1024 pour d'autres).
    -- Toute requete filtre d'abord sur index_id, ce qui garantit une
    -- dimension homogene au moment de la comparaison.
    embedding   VECTOR,
    tsv         TSVECTOR,                   -- pour le mode BM25 / hybride
    UNIQUE (index_id, document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_index_id_idx ON chunks (index_id);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);

CREATE TABLE IF NOT EXISTS questions (
    id            SERIAL PRIMARY KEY,
    dataset_id    INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    external_id   TEXT NOT NULL,
    question      TEXT NOT NULL,
    answer        TEXT,                     -- reponse de reference (gold)
    question_type TEXT,                     -- inference / comparison / temporal / null...
    -- Passages de reference. Structure : [{"document_external_id": ..., "fact": ...}]
    -- C'est ce qui permet les metriques de retrieval avec verite terrain
    -- (recall@k, nDCG@k) — sans ca on ne peut que juger au LLM.
    gold_evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    split         TEXT NOT NULL DEFAULT 'eval',
    UNIQUE (dataset_id, external_id)
);

CREATE INDEX IF NOT EXISTS questions_dataset_idx ON questions (dataset_id, split);

-- --------------------------------------------------------------------
-- Couche experimentation : configs, runs, predictions
-- --------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS configs (
    hash        TEXT PRIMARY KEY,           -- PipelineConfig.hash()
    name        TEXT NOT NULL,
    description TEXT,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
    id           SERIAL PRIMARY KEY,
    dataset_id   INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    config_hash  TEXT NOT NULL REFERENCES configs(hash),
    index_id     INTEGER REFERENCES corpus_indexes(id) ON DELETE SET NULL,
    label        TEXT,
    git_sha      TEXT,                      -- version du code, pour la reproductibilite
    status       TEXT NOT NULL DEFAULT 'running',  -- running | completed | failed
    n_questions  INTEGER NOT NULL DEFAULT 0,
    n_failed     INTEGER NOT NULL DEFAULT 0,
    -- Avertissements methodologiques de PipelineConfig.warnings(), figes au
    -- moment du run : un resultat doit porter ses propres reserves.
    warnings     JSONB NOT NULL DEFAULT '[]'::jsonb,
    usage        JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS runs_config_idx ON runs (config_hash, started_at DESC);

CREATE TABLE IF NOT EXISTS predictions (
    id                BIGSERIAL PRIMARY KEY,
    run_id            INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    question_id       INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    -- Contextes remontes, dans l'ordre de rang. Structure :
    -- [{"chunk_id":…, "document_external_id":…, "text":…, "score":…, "rank":…}]
    contexts          JSONB NOT NULL DEFAULT '[]'::jsonb,
    answer            TEXT,
    abstained         BOOLEAN NOT NULL DEFAULT FALSE,
    retrieval_ms      INTEGER,
    generation_ms     INTEGER,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    error             TEXT,
    UNIQUE (run_id, question_id)
);

CREATE INDEX IF NOT EXISTS predictions_run_idx ON predictions (run_id);

-- --------------------------------------------------------------------
-- Couche mesure : scores automatiques et annotations humaines
-- --------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS scores (
    id          BIGSERIAL PRIMARY KEY,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    -- NULL = score agrege au niveau du run (ex. nDCG moyen).
    question_id INTEGER REFERENCES questions(id) ON DELETE CASCADE,
    evaluator   TEXT NOT NULL,              -- 'native.ir' | 'ragas' | 'ragchecker' | ...
    metric      TEXT NOT NULL,              -- 'recall@5' | 'faithfulness' | ...
    value       DOUBLE PRECISION,
    -- Trace du jugement : claims extraites, verdict par claim, prompt du
    -- juge. Sans ca un score de 0.62 est ininterpretable et le drill-down
    -- sur les echecs est impossible.
    detail      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS scores_run_metric_idx ON scores (run_id, evaluator, metric);
CREATE UNIQUE INDEX IF NOT EXISTS scores_unique_idx
    ON scores (run_id, COALESCE(question_id, -1), evaluator, metric);

CREATE TABLE IF NOT EXISTS annotations (
    id          BIGSERIAL PRIMARY KEY,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    metric      TEXT NOT NULL,              -- la metrique jugee a la main
    value       DOUBLE PRECISION NOT NULL,  -- meme echelle que scores.value
    annotator   TEXT NOT NULL DEFAULT 'human',
    notes       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, question_id, metric, annotator)
);

-- --------------------------------------------------------------------
-- Vue de commodite : le tableau de bord lit ca, pas les tables brutes.
-- --------------------------------------------------------------------

CREATE OR REPLACE VIEW run_summary AS
SELECT
    r.id            AS run_id,
    r.label,
    r.status,
    c.name          AS config_name,
    r.config_hash,
    d.name          AS dataset,
    s.evaluator,
    s.metric,
    avg(s.value)    AS mean_value,
    stddev_samp(s.value) AS sd_value,
    count(s.value)  AS n,
    r.started_at
FROM runs r
JOIN configs  c ON c.hash = r.config_hash
JOIN datasets d ON d.id = r.dataset_id
LEFT JOIN scores s ON s.run_id = r.id AND s.question_id IS NOT NULL
GROUP BY r.id, r.label, r.status, c.name, r.config_hash, d.name,
         s.evaluator, s.metric, r.started_at;
