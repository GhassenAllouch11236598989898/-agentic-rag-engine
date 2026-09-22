"""
RAG Confidence & Correctness Evaluator.

Implements a dual-engine evaluation approach:
  1. **LLM-as-a-Judge**: Uses the configured LLM to assess faithfulness,
     relevance, and extract/verify atomic claims.
  2. **Semantic Embedding Alignment**: Uses SentenceTransformers cosine
     similarity as an objective cross-check.

Based on modern RAG evaluation methodology:
  - RAG Triad (Faithfulness, Answer Relevance, Context Relevance)
  - Claim-level grounding verification
  - Calibrated composite confidence score
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
from dotenv import load_dotenv

from .models import (
    ClaimVerification,
    RAGMetrics,
    EvaluationResult,
)
from .providers import get_llm_model, get_embedding_provider

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation prompts
# ---------------------------------------------------------------------------

_EVALUATION_PROMPT = """You are an expert RAG system evaluator. Analyze the following triplet and produce a JSON evaluation.

## User Query
{query}

## Retrieved Context Chunks
{contexts}

## Generated Response
{response}

## Your Task
Evaluate the response along these dimensions and respond with ONLY valid JSON (no markdown, no explanation outside JSON):

{{
  "faithfulness": <float 0.0–1.0, how well the response is grounded in the context>,
  "answer_relevance": <float 0.0–1.0, how directly the response answers the query>,
  "context_relevance": <float 0.0–1.0, how relevant the retrieved chunks are to the query>,
  "claims": [
    {{
      "claim": "<atomic factual statement from the response>",
      "status": "<supported | partially_supported | unsupported>",
      "supporting_chunk_index": <int or null, 0-based index of supporting context chunk>,
      "reasoning": "<brief justification>"
    }}
  ],
  "critique": "<2-3 sentence overall assessment>"
}}

Rules:
- faithfulness: 1.0 = every claim is directly supported by context; 0.0 = entirely fabricated
- answer_relevance: 1.0 = perfectly addresses the query; 0.0 = completely off-topic
- context_relevance: 1.0 = all chunks are highly relevant; 0.0 = none are relevant
- Extract 2-5 key atomic claims from the response
- Be strict but fair in your assessment"""


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """Evaluates RAG responses for confidence, correctness, and hallucination risk."""

    # Composite confidence weights
    W_FAITHFULNESS = 0.40
    W_ANSWER_REL = 0.30
    W_CONTEXT_REL = 0.15
    W_SEMANTIC = 0.15

    def __init__(self):
        self._embedding_provider = None

    def _get_embedding_provider(self):
        """Lazy-load the embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider()
        return self._embedding_provider

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str],
    ) -> EvaluationResult:
        """
        Run full evaluation pipeline on a RAG triplet.

        Args:
            query: The original user query.
            response: The generated LLM response.
            contexts: List of retrieved context chunk texts.

        Returns:
            Complete EvaluationResult with scores, claims, and risk tier.
        """
        start_time = datetime.now()

        if not response or not response.strip():
            return self._empty_evaluation(start_time)

        # Run LLM judge and semantic analysis concurrently
        llm_task = self._llm_judge(query, response, contexts)
        semantic_task = self._semantic_alignment(response, contexts)

        llm_result, semantic_score = await asyncio.gather(
            llm_task, semantic_task, return_exceptions=True
        )

        # Handle LLM judge failures gracefully
        if isinstance(llm_result, Exception):
            raise RuntimeError('Evaluation unavailable: the judge failed.') from llm_result

        # Handle semantic failures gracefully
        if isinstance(semantic_score, Exception):
            raise RuntimeError('Evaluation unavailable: embedding comparison failed.') from semantic_score

        # Assemble metrics
        metrics = RAGMetrics(
            faithfulness=llm_result["faithfulness"],
            answer_relevance=llm_result["answer_relevance"],
            context_relevance=llm_result["context_relevance"],
            semantic_similarity=semantic_score,
        )

        # Compute calibrated composite confidence
        confidence = (
            self.W_FAITHFULNESS * metrics.faithfulness
            + self.W_ANSWER_REL * metrics.answer_relevance
            + self.W_CONTEXT_REL * metrics.context_relevance
            + self.W_SEMANTIC * metrics.semantic_similarity
        ) * 100.0

        # Determine risk level
        risk_level = self._risk_tier(confidence)

        # Build claim verifications
        claims = [
            ClaimVerification(
                claim=c.get("claim", ""),
                status=c.get("status", "unsupported"),
                supporting_chunk_index=c.get("supporting_chunk_index"),
                reasoning=c.get("reasoning", ""),
            )
            for c in llm_result.get("claims", [])
        ]

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        return EvaluationResult(
            confidence_score=round(confidence, 1),
            risk_level=risk_level,
            metrics=metrics,
            claims=claims,
            critique=llm_result.get("critique", ""),
            evaluated_at=datetime.now(),
            evaluation_time_ms=round(elapsed_ms, 1),
        )

    # ------------------------------------------------------------------
    # LLM-as-a-Judge
    # ------------------------------------------------------------------

    async def _llm_judge(
        self,
        query: str,
        response: str,
        contexts: List[str],
    ) -> Dict[str, Any]:
        """Use the configured LLM to evaluate the RAG triplet."""
        from pydantic_ai import Agent

        # Format contexts for the prompt
        if contexts:
            formatted_contexts = "\n\n".join(
                f"[Chunk {i}]: {chunk[:800]}" for i, chunk in enumerate(contexts)
            )
        else:
            formatted_contexts = "(No context chunks were retrieved)"

        prompt = _EVALUATION_PROMPT.format(
            query=query,
            response=response[:2000],
            contexts=formatted_contexts,
        )

        # Use a lightweight evaluator agent
        eval_agent = Agent(
            get_llm_model(),
            system_prompt="You are a JSON-only evaluation engine. Always respond with valid JSON only.",
        )

        result = await eval_agent.run(prompt)

        # Extract text from result
        if hasattr(result, "output"):
            raw = str(result.output)
        elif hasattr(result, "data"):
            raw = str(result.data)
        else:
            raw = str(result)

        return self._parse_llm_response(raw)

    def _parse_llm_response(self, raw: str) -> Dict[str, Any]:
        """Parse the LLM's JSON response, handling common formatting issues."""
        # Try to extract JSON from markdown code blocks
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
            cleaned = cleaned.split("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1]
            cleaned = cleaned.split("```", 1)[0]

        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse LLM evaluation JSON")
                    return self._fallback_llm_result(0.5)
            else:
                logger.warning("No JSON object found in LLM evaluation response")
                return self._fallback_llm_result(0.5)

        # Validate and clamp scores
        return {
            "faithfulness": max(0.0, min(1.0, float(data.get("faithfulness", 0.5)))),
            "answer_relevance": max(0.0, min(1.0, float(data.get("answer_relevance", 0.5)))),
            "context_relevance": max(0.0, min(1.0, float(data.get("context_relevance", 0.5)))),
            "claims": data.get("claims", []),
            "critique": data.get("critique", "Evaluation completed."),
        }

    def _fallback_llm_result(self, base_score: float = 0.5) -> Dict[str, Any]:
        """Never manufacture a successful evaluation when parsing fails."""
        raise ValueError('Evaluation unavailable: the judge returned invalid JSON.')

    # ------------------------------------------------------------------
    # Semantic Embedding Alignment
    # ------------------------------------------------------------------

    async def _semantic_alignment(
        self,
        response: str,
        contexts: List[str],
    ) -> float:
        """
        Compute cosine similarity between the response embedding and
        the mean of the context chunk embeddings.
        """
        if not contexts:
            return 0.0

        provider = self._get_embedding_provider()

        # Embed response and all contexts in one batch
        all_texts = [response] + contexts
        embeddings = await provider.embed_batch(all_texts)

        if not embeddings or len(embeddings) < 2:
            return 0.0

        response_vec = np.array(embeddings[0])
        context_vecs = np.array(embeddings[1:])

        # Mean-pool context embeddings
        context_mean = context_vecs.mean(axis=0)

        # Cosine similarity
        dot = np.dot(response_vec, context_mean)
        norm_r = np.linalg.norm(response_vec)
        norm_c = np.linalg.norm(context_mean)

        if norm_r == 0 or norm_c == 0:
            return 0.0

        similarity = float(dot / (norm_r * norm_c))
        return max(0.0, min(1.0, similarity))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_tier(confidence: float) -> str:
        """Map confidence score to a hallucination risk tier."""
        if confidence >= 80.0:
            return "low"
        elif confidence >= 60.0:
            return "medium"
        else:
            return "high"

    @staticmethod
    def _empty_evaluation(start_time: datetime) -> EvaluationResult:
        """Return an empty evaluation for blank responses."""
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        return EvaluationResult(
            confidence_score=0.0,
            risk_level="high",
            metrics=RAGMetrics(
                faithfulness=0.0,
                answer_relevance=0.0,
                context_relevance=0.0,
                semantic_similarity=0.0,
            ),
            claims=[],
            critique="Empty response — no evaluation possible.",
            evaluated_at=datetime.now(),
            evaluation_time_ms=round(elapsed, 1),
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_evaluator: Optional[RAGEvaluator] = None


def get_evaluator() -> RAGEvaluator:
    """Return the global RAGEvaluator singleton (lazy-init)."""
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGEvaluator()
    return _evaluator
