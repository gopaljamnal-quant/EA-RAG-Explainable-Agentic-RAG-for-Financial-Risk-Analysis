"""
Retrieval components.

DenseRetriever: a TF-IDF retriever over the unstructured corpus D (filings,
news, transcripts). This stands in for the embedding-based dense retriever
described in Section IV.C; swap `vectorize()` for a sentence-embedding model
(e.g. sentence-transformers) for production use without touching the rest
of the pipeline -- the rest of EA-RAG only depends on the RetrievedPassage
interface, not on TF-IDF specifically.

GraphRetriever: thin wrapper around FinancialKnowledgeGraph.bounded_traversal
that matches the SubTask.seed_entities to graph nodes.
"""

from __future__ import annotations

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_models import Document, GraphEvidence, RetrievedPassage, SubTask
from .kg import FinancialKnowledgeGraph


class DenseRetriever:
    def __init__(self, documents: List[Document]) -> None:
        self.documents = documents
        self._vectorizer = TfidfVectorizer(stop_words="english")
        corpus_texts = [d.text for d in documents] if documents else [""]
        self._matrix = self._vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.05) -> List[RetrievedPassage]:
        if not self.documents:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).ravel()
        ranked = self._ranked_indices(scores)
        results: List[RetrievedPassage] = []
        for idx in ranked[:top_k]:
            if scores[idx] < min_score:
                continue
            results.append(self._to_passage(idx, float(scores[idx])))
        return results

    @staticmethod
    def _ranked_indices(scores) -> List[int]:
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    def _to_passage(self, idx: int, score: float) -> RetrievedPassage:
        doc = self.documents[idx]
        return RetrievedPassage(
            doc_id=doc.id,
            text=doc.text,
            score=score,
            source_type=doc.source_type,
        )


class GraphRetriever:
    def __init__(self, kg: FinancialKnowledgeGraph, max_hops: int = 3) -> None:
        self.kg = kg
        self.max_hops = max_hops

    def retrieve(self, subtask: SubTask) -> GraphEvidence:
        seeds = self._resolve_seed_entities(subtask.seed_entities)
        if not seeds:
            return GraphEvidence(entities=[], relations=[], seed_entities=[])
        return self.kg.bounded_traversal(seeds, max_hops=self.max_hops)

    def _resolve_seed_entities(self, seed_entities: List[str]) -> List[str]:
        return [
            entity_id
            for name_or_id in seed_entities
            for entity_id in [self._resolve(name_or_id)]
            if entity_id is not None
        ]

    def _resolve(self, name_or_id: str):
        if self.kg.get_entity(name_or_id) is not None:
            return name_or_id
        entity = self.kg.find_entity_by_name(name_or_id)
        return entity.id if entity else None
