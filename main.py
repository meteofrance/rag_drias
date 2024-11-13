import shutil
from pathlib import Path
import time
from typing import List, Literal

import torch
import typer
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from rag_drias import data
from rag_drias.crawler import scrape_page
from rag_drias.embedding import TypeEmbedding, get_embedding
from rag_drias.settings import PATH_DATA, PATH_VDB, PATH_LLM, PATH_RERANKER, URLS

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise Exception("GPU non disponible.")

app = typer.Typer(pretty_exceptions_enable=False)


# ----- Chroma Database -----


def chunks_similarity_filter(
    chunks: List[Document], embedding: TypeEmbedding, threshold: float = 0.98
) -> List[Document]:
    """Returns a list of chunks with a similarity below a threshold"""
    chunks_embeddings = [
        embedding.encode(chunk.page_content)
        for chunk in tqdm(chunks, desc="filtering embedding")
    ]
    mat_sim = cosine_similarity(chunks_embeddings, chunks_embeddings)
    idx_to_remove = []
    for i in range(len(chunks) - 1):
        for j in range(i + 1, len(chunks)):
            if mat_sim[i, j] > threshold:
                idx_to_remove.append(i)
                continue
    unique_chunks = [chunks[i] for i in range(len(chunks)) if i not in idx_to_remove]
    print(f"{len(unique_chunks)} unique chunks load")

    return unique_chunks


def get_db_path(
    embedding_model: Literal["Camembert", "E5"] = "Camembert",
    data_source: str = "Confluence",
) -> Path:
    """Get path of the database."""
    return PATH_VDB / f"{data_source}_{embedding_model}"


def create_chroma_db(
    path_db: Path,
    embedding: TypeEmbedding,
    docs: List[Document],
    overwrite: bool = False,
):
    """Create a vector database from the documents"""
    if overwrite and path_db.exists():
        shutil.rmtree(path_db)
    path_db.mkdir(parents=True, exist_ok=True)
    if any(path_db.iterdir()):  # case overwrite = False
        raise FileExistsError(f"Vector database directory {path_db} is not empty")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=str(path_db),  # Does not accept Path
    )
    vectordb.persist()  # Save database to use it later
    print(f"Vector database created in {path_db}")
    return vectordb


def load_chroma_db(path_db: Path, embedding: TypeEmbedding):
    if not (path_db.exists() and any(path_db.iterdir())):
        raise FileExistsError(f"Vector database {path_db} needs to be prepared.")
    return Chroma(embedding_function=embedding, persist_directory=str(path_db))


def rerank(text: str, docs: List[Document], k: int = 4) -> List[Document]:
    """Returns the k most relevant chunks for the question chosen by a reranker llm."""
    rerank_tokenizer = AutoTokenizer.from_pretrained(PATH_RERANKER)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(PATH_RERANKER)
    rerank_model = rerank_model.to(device)
    rerank_model.eval()

    rerank_inp = [[text, doc.page_content] for doc in docs]
    with torch.no_grad():
        inputs = rerank_tokenizer(
            rerank_inp,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(device)
        scores = (
            rerank_model(**inputs, return_dict=True)
            .logits.view(
                -1,
            )
            .float()
        )

    _, indices = scores.topk(k)
    return [docs[i] for i in indices]


def retrieve(
    text: str,
    embedding_name: str,
    data_source: str,
    n_samples: int,
    use_rerank: bool,
) -> List[Document]:
    """Retrieve the most relevant chunks in relation to the query."""
    path_db = get_db_path(embedding_name, data_source)
    embedding = get_embedding(embedding_name)
    vectordb = load_chroma_db(path_db, embedding)

    if use_rerank:
        chunks = vectordb.similarity_search(text, k=n_samples)
        chunks = rerank(text, chunks, k=n_samples // 2)
        # we return the chunks by ascending score because better results when the relevant chunks are closer to the question
        chunks.reverse()
    else:
        chunks = vectordb.max_marginal_relevance_search(
            text, k=n_samples, fetch_k=int(n_samples * 1.5)
        )
    return chunks


# ----- Typer commands -----


@app.command()
def prepare(
    max_crawl_depth: int = 3,
    embedding_name: str = "Camembert",
    data_source: str = "Drias",
    overwrite: bool = False,
):
    """Prepare the Chroma vector database by crawling the URL and embedding all the text data.

    Args:
        max_crawl_depth (int, optional): Maximum depth of the crawl. Defaults to 3.
        embedding_name (Camembert or E5): Embedding model name. Defaults to Camembert.
        path_data (Path, optional): Name of the data source. Defaults to Drias.
        overwrite (bool, optional): Whether. Defaults to False.
    """

    print(f"Start crawling {URLS[data_source]}")
    start_time = time()
    scrape_page(URLS[data_source], max_depth=max_crawl_depth)
    print(f"Execution time : {time() - start_time}")

    path_data = PATH_DATA / data_source
    docs = data.create_docs(path_data)
    docs = data.split_to_paragraphs(docs)
    chunks = data.split_to_chunks(docs)
    embedding = get_embedding(embedding_name)
    chunks = chunks_similarity_filter(chunks, embedding)
    path_db = get_db_path(embedding_name, data_source)
    create_chroma_db(path_db, embedding, chunks, overwrite)


@app.command()
def query(
    text: str,
    embedding_name: str = "Camembert",
    data_source: str = "Confluence",
    n_samples: int = 4,
    use_rerank: bool = False,
):
    """Makes a query to the vector database and retrieves the closest chunks.

    Args:
        text (str): Your query.
        embedding_name (str, optional): Embedding model name. Defaults to "Camembert".
        data_source (str, optional): Name of the data source. Defaults to "Confluence".
    """
    chunks = retrieve(text, embedding_name, data_source, n_samples, use_rerank)
    for i, chunk in enumerate(chunks):
        print(f"---> Relevant chunk {i} <---")
        data.print_doc(chunk)
        print("-" * 20)


@app.command()
def answer(
    text: str,
    embedding_name: str = "Camembert",
    data_source: str = "Confluence",
    n_samples: int = 10,
    use_rag: bool = True,
    use_rerank: bool = False,
):
    """Generate text from a prompt after rag and print it."""

    model = AutoModelForCausalLM.from_pretrained(
        PATH_LLM,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,  # Allow using code that was not written by HuggingFace
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(PATH_LLM)

    if use_rag:
        chunks = retrieve(text, embedding_name, data_source, n_samples, use_rerank)

        retrieved_infos = ""
        for chunk in chunks:
            retrieved_infos += f"\n-- Page Title : {chunk.metadata['title']} --\n"
            retrieved_infos += f"-- url : {chunk.metadata['url']} --\n"
            retrieved_infos += chunk.page_content

        message = [
            {
                "role": "system",
                "content": "Le portail DRIAS mets à disposition les projections climatiques régionalisées de référence, pour l'adaptation en France. Tu es un chatbot qui reponds aux questions à l'aide d'informations récupérées sur le site.",
            },
            {
                "role": "user",
                "content": f"Avec les informations suivantes si utiles: {retrieved_infos}\nRéponds à cette question de manière claire et concise: {text}\nRéponse:",
            },
        ]
    else:
        message = [
            {
                "role": "system",
                "content": "Le portail DRIAS mets à disposition les projections climatiques régionalisées de référence, pour l'adaptation en France. Tu es un chatbot qui reponds aux questions sur le site.",
            },
            {"role": "user", "content": f"Réponds à cette question: {text}"},
        ]

    prompt = tokenizer.apply_chat_template(
        message, add_generation_prompt=True, tokenize=False
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device=0,
        pad_token_id=tokenizer.eos_token_id,
    )

    sequences = pipeline(
        prompt,
        do_sample=True,
        temperature=0.1,
        num_return_sequences=1,
        max_new_tokens=500,
    )

    print("#" * 50 + f"\nLLM input:\n{prompt}\n" + "#" * 50)
    print(f"LLM output:\n{sequences[0]['generated_text'][len(prompt):]}")


if __name__ == "__main__":
    app()
