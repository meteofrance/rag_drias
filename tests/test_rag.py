from langchain_core.documents.base import Document
from rag_drias.embedding import get_embedding
from main import rerank, query, answer
from rag_drias.data import filter_similar_chunks
from pathlib import Path

PATH_DB = Path(
    "/scratch/shared/rag_drias/tests/chroma_database/sentence-camembert-large"
)

CHUNKS = [
    Document(
        page_content="Les trous noirs sont des régions de l'espace où la gravité est si forte que rien, pas même la lumière, ne peut s'en échapper.",
        metadata={"title": "Trous noirs", "url": "https://trous-noirs.com"},
    ),
    Document(
        page_content="Le chat est un animal domestique de la famille des félidés.",
        metadata={"title": "Chat", "url": "https://chat.com"},
    ),
    Document(
        page_content="Le chat est un animal domestique de la famille des félidés.",
        metadata={"title": "Chat", "url": "https://chat.com"},
    ),
    Document(
        page_content="Le chat aime dormir, manger et jouer.",
        metadata={"title": "Chat", "url": "https://chat.com"},
    ),
]


def test_similarity():
    embedding = get_embedding("sentence-camembert-large")
    unique_chunks = filter_similar_chunks(CHUNKS, embedding, threshold=0.98)
    assert len(unique_chunks) == 3


def test_reranker():
    embedding = get_embedding("sentence-camembert-large")
    unique_chunks = filter_similar_chunks(CHUNKS, embedding, threshold=0.98)
    ranking_chunks = rerank(
        model_name="bge-reranker-v2-m3",
        text="Qu'es-ce qu'un chat ?",
        docs=unique_chunks,
        k=3,
    )
    assert (
        ranking_chunks[0] == unique_chunks[1]
        and ranking_chunks[1] == unique_chunks[2]
        and ranking_chunks[2] == unique_chunks[0]
    )


def test_query():
    # Without reranker
    retrieved_chunks = query(
        text="Qu'es-ce qu'un chat ?",
        embedding_name="sentence-camembert-large",
        n_samples=3,
        path_db=PATH_DB,
    )
    assert (
        retrieved_chunks[0] == CHUNKS[1]
        and retrieved_chunks[1] == CHUNKS[3]
        and retrieved_chunks[2] == CHUNKS[0]
    )
    # With reranker
    retrieved_chunks = query(
        text="Qu'es-ce qu'un chat ?",
        embedding_name="sentence-camembert-large",
        n_samples=4,
        reranker="bge-reranker-v2-m3",
        path_db=PATH_DB,
    )
    assert retrieved_chunks[0] == CHUNKS[3] and retrieved_chunks[1] == CHUNKS[1]


def test_answer():
    response = answer(
        question="Qu'es-ce qu'un chat ?",
        embedding_model="sentence-camembert-large",
        generative_model="Llama-3.2-3B-Instruct",
        n_samples=4,
        path_db=PATH_DB,
    )
    assert response == "Un chat est un animal domestique de la famille des félidés."


if __name__ == "__main__":
    test_similarity()
    test_reranker()
    test_query()
    test_answer()