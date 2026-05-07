from sentence_transformers import SentenceTransformer
from config import BGE_MODEL_NAME, BATCH_SIZE


class Embedder:
    _model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(BGE_MODEL_NAME)
        return self._model

    def embed(self, text: str) -> list[float]:
        return self._get_model().encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._get_model().encode(
            texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True
        ).tolist()
