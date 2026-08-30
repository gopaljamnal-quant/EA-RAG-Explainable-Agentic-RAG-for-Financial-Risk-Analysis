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
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: List[RetrievedPassage] = []
        for idx in ranked[:top_k]:
            if scores[idx] < min_score:
                continue
            doc = self.documents[idx]
            results.append(
                RetrievedPassage(
                    doc_id=doc.id,
                    text=doc.text,
                    score=float(scores[idx]),
                    source_type=doc.source_type,
                )
            )
        return results


class GraphRetriever:
    def __init__(self, kg: FinancialKnowledgeGraph, max_hops: int = 3) -> None:
        self.kg = kg
        self.max_hops = max_hops

    def retrieve(self, subtask: SubTask) -> GraphEvidence:
        seeds = [
            eid
            for name_or_id in subtask.seed_entities
            for eid in [self._resolve(name_or_id)]
            if eid is not None
        ]
        if not seeds:
            return GraphEvidence(entities=[], relations=[], seed_entities=[])
        return self.kg.bounded_traversal(seeds, max_hops=self.max_hops)

    def _resolve(self, name_or_id: str):
        if self.kg.get_entity(name_or_id) is not None:
            return name_or_id
        entity = self.kg.find_entity_by_name(name_or_id)
        return entity.id if entity else None
