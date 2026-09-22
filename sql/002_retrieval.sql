-- Non-destructive migration. RRF combines ranks, never incomparable raw scores.
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING GIN (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_documents_checksum ON documents ((metadata->>'checksum'));
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector(384), query_text TEXT, match_count INT DEFAULT 10,
    text_weight FLOAT DEFAULT 0.3
) RETURNS TABLE (
    chunk_id UUID, document_id UUID, content TEXT, combined_score FLOAT,
    vector_similarity FLOAT, text_similarity FLOAT, metadata JSONB,
    document_title TEXT, document_source TEXT
) LANGUAGE sql STABLE AS $$
WITH dense_candidates AS (
    SELECT id, embedding <=> query_embedding AS distance FROM chunks
    WHERE embedding IS NOT NULL ORDER BY embedding <=> query_embedding LIMIT 100
), dense AS (
    SELECT id, 1-distance AS similarity, row_number() OVER (ORDER BY distance, id) AS rank FROM dense_candidates
), sparse_candidates AS (
    SELECT id, ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', query_text)) AS relevance
    FROM chunks WHERE to_tsvector('english', content) @@ plainto_tsquery('english', query_text)
    ORDER BY relevance DESC, id LIMIT 100
), sparse AS (
    SELECT id, relevance, row_number() OVER (ORDER BY relevance DESC, id) AS rank FROM sparse_candidates
), fused AS (
    SELECT COALESCE(d.id,s.id) AS id,
      COALESCE((1-GREATEST(0,LEAST(1,text_weight)))/(60.0+d.rank),0)
      + COALESCE(GREATEST(0,LEAST(1,text_weight))/(60.0+s.rank),0) AS score,
      COALESCE(d.similarity,0) AS vector_score, COALESCE(s.relevance,0) AS text_score
    FROM dense d FULL OUTER JOIN sparse s ON d.id=s.id
)
SELECT c.id,c.document_id,c.content,f.score::float,f.vector_score::float,f.text_score::float,
    c.metadata,doc.title,doc.source
FROM fused f JOIN chunks c ON c.id=f.id JOIN documents doc ON doc.id=c.document_id
ORDER BY f.score DESC,c.id LIMIT GREATEST(1,LEAST(50,match_count));
$$;
