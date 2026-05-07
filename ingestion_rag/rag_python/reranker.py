from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL_NAME


class Reranker:
    _model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(RERANKER_MODEL_NAME)
        return self._model

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        pairs = [(query, chunk["compressedContent"]) for chunk in chunks]
        scores = self._get_model().predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        result = []
        for score, chunk in ranked[:top_k]:
            chunk = dict(chunk)
            chunk["_rerank_score"] = float(score)
            result.append(chunk)
        return result
