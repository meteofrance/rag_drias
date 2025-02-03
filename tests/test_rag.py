from pathlib import Path

from langchain_core.documents.base import Document

from main import answer, create_bm25_idx, create_chroma_db, get_db_path, query, rerank
from rag_drias.crawler import crawl_website
from rag_drias.data import filter_similar_chunks
from rag_drias.embedding import get_embedding
from rag_drias.settings import BASE_URL

PATH_TMP = Path("tmp/")

CHUNKS = [
    Document(
        page_content="Les trous noirs sont des régions de l'espace où la gravité est\
 si forte que rien, pas même la lumière, ne peut s'en échapper.",
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
    embedding = get_embedding("sentence-transformers/all-MiniLM-L12-v2")
    unique_chunks = filter_similar_chunks(CHUNKS, embedding, threshold=0.98)
    assert unique_chunks == [CHUNKS[0], CHUNKS[1], CHUNKS[3]]


def test_create_chroma_db():
    unique_chunks = filter_similar_chunks(
        CHUNKS, get_embedding("sentence-transformers/all-MiniLM-L12-v2"), threshold=0.98
    )
    embedding_model = "sentence-transformers/all-MiniLM-L12-v2"
    embedding = get_embedding(embedding_model)
    path_db = get_db_path(embedding_model, PATH_TMP)
    create_chroma_db(path_db, embedding, unique_chunks)


def test_create_bm25_idx():
    unique_chunks = filter_similar_chunks(
        CHUNKS, get_embedding("sentence-transformers/all-MiniLM-L12-v2"), threshold=0.98
    )
    create_bm25_idx(PATH_TMP, unique_chunks)
    assert (PATH_TMP / "without_pdfs" / "bm25_index.json").exists()


def test_crawl():
    crawl_website(BASE_URL, max_depth=0, path_data=PATH_TMP)
    path_html = PATH_TMP / "HTMLs"
    assert path_html.exists() and len(list(path_html.glob("*.html"))) == 1


def test_reranker():
    embedding = get_embedding("sentence-transformers/all-MiniLM-L12-v2")
    unique_chunks = filter_similar_chunks(CHUNKS, embedding, threshold=0.98)
    ranking_chunks = rerank(
        model_name="Alibaba-NLP/gte-multilingual-reranker-base",
        text="Qu'es-ce qu'un chat ?",
        docs=unique_chunks,
        k=3,
    )
    assert (
        ranking_chunks[0] == unique_chunks[1] and ranking_chunks[1] == unique_chunks[2]
    )


def test_query():
    # Without reranker
    retrieved_chunks = query(
        text="Qu'es-ce qu'un chat ?",
        embedding_name="sentence-transformers/all-MiniLM-L12-v2",
        n_samples=3,
        path_db=PATH_TMP,
    )
    assert (
        retrieved_chunks[0] == CHUNKS[3]
        and retrieved_chunks[1] == CHUNKS[1]
        and retrieved_chunks[2] == CHUNKS[0]
    )
    # With reranker
    retrieved_chunks = query(
        text="Qu'es-ce qu'un chat ?",
        embedding_name="sentence-transformers/all-MiniLM-L12-v2",
        n_samples=4,
        reranker="Alibaba-NLP/gte-multilingual-reranker-base",
        path_db=PATH_TMP,
    )
    assert retrieved_chunks[0] == CHUNKS[1] and retrieved_chunks[1] == CHUNKS[3]


def test_answer():
    response = answer(
        question="Qu'es-ce qu'un chat ?",
        embedding_model="sentence-transformers/all-MiniLM-L12-v2",
        generative_model="tiiuae/Falcon3-1B-Instruct",
        n_samples=2,
        path_db=PATH_TMP,
        max_new_tokens=5,
    )
    assert response == "Un chat est un animal"
