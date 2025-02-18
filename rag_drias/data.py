import json
import re
from pathlib import Path
from typing import List

import pymupdf4llm
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents.base import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from markdownify import markdownify as md
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from rag_drias.embedding import Embedding
from rag_drias.settings import PATH_MENU_JSON


def replace_many_newlines(string: str) -> str:
    """Replace many newlines with two newlines or two newlines with one newline."""
    string = re.sub(r"\n{3}", "\n\n", string)
    string = re.sub(r"\n{2}", "\n", string)
    string = re.sub(r"\n{3,}", "\n\n", string)
    string = re.sub(r"(?<=\?)\n", " ", string)
    return string


def get_id_page(doc: Document) -> tuple[int, int]:
    """Returns the page and main page ids if they exist."""
    page_id = None
    firstpage_id = None

    id_match = re.search(r'var id\s*=\s*[\'"]?([^\'";]+)', doc.page_content)
    if id_match:
        page_id = int(id_match.group(1))
    firstid_match = re.search(
        r'var firstSectionId\s*=\s*[\'"]?([^\'";]+)', doc.page_content
    )
    if firstid_match:
        firstpage_id = int(firstid_match.group(1))

    return page_id, firstpage_id


def find_parent(id: str, titles: List) -> str:
    """Returns the titles of the parts where the page is located."""
    for title_dict in titles:
        if id == title_dict["id"]:
            return title_dict["title"]
        elif title_dict["children"] != []:
            subtitle = find_parent(id, title_dict["children"])
            if subtitle:
                return title_dict["title"] + " > " + subtitle


def get_title_drias(doc: Document, titles_list: List) -> str:
    """Returns the title of the page."""
    title_match = re.search(r"<title>(.*?)</title>", doc.page_content)
    suptitle = title_match.group(1) if title_match else ""
    page_id, firstpage_id = get_id_page(doc)
    if page_id and firstpage_id:
        for title_dict in titles_list:
            if title_dict["id"] == firstpage_id:
                if title_dict["id"] == page_id:
                    return suptitle + " > " + title_dict["title"]
                return (
                    suptitle
                    + " > "
                    + title_dict["title"]
                    + " > "
                    + find_parent(page_id, title_dict["children"])
                )
    return suptitle


def get_url(doc_str: str) -> str:
    """Returns the url of the page which is written on the first line."""
    lines = doc_str.split("\n")
    url = lines[0][len("<!-- URL: ") : -len(" -->")]
    return url


def clean_drias_html(doc_str: str) -> str:
    """Remove repetitives parts of the HTML."""
    soup = BeautifulSoup(doc_str, "html.parser")

    head = soup.find("head", id="head")
    if head:
        head.decompose()
    bandeau_div = soup.find("div", id="bandeau")
    if bandeau_div:
        bandeau_div.decompose()
    menu_div = soup.find("table", class_="maTable")
    if menu_div:
        menu_div.decompose()
    footer_div = soup.find("div", class_="footer")
    if footer_div:
        footer_div.decompose()
    login_div = soup.find("div", id="accueilUneLien")
    if login_div:
        login_div.decompose()

    # remove images
    images = soup.find_all("img")
    for img in images:
        img.decompose()

    section_div = soup.find("div", id="section")
    if section_div:
        # delete title
        title_div = soup.find("div", id="titleSection")
        if title_div:
            title_div.decompose()

    return str(soup)


def clean_drias_doc(doc_str: str) -> str:
    """Remove multiple new lines and errors."""

    if doc_str == "Expired session":
        doc_str = ""

    doc_str = doc_str.replace(r"\-", "-")
    doc_str = doc_str.replace(r"\_", "_")
    doc_str = doc_str.replace(r"\[", "[")
    doc_str = doc_str.replace(r"**", "")

    lines = doc_str.split("\n")
    for i, line in enumerate(lines):
        if line == " No translation available yet":
            line = ""
        line = re.sub(r"\|(\s*(-{3,})?\s*\|)+\s*", "", line)
        line = re.sub(r"(\|\s*\|\s*\*\s*)+\+", "", line)
        line = re.sub(r"^(##)?(-{3,})?\*?\s*$", "", line)
        line = re.sub(r"\s{4}", " ", line)
        line = re.sub(r"#{4,}", "####", line)

        # remove table because the information is divided after the split
        line = re.sub(r"^\|.*\|$", "", line)
        lines[i] = line
    doc_str = "\n".join(lines)
    doc_str = replace_many_newlines(doc_str)
    return doc_str


def create_docs_html(source_html_path: Path) -> List[Document]:
    """Loads .html files as langchain docs, convert to md and clean them."""

    path_json = PATH_MENU_JSON
    with open(path_json, "r", encoding="utf-8") as file:
        titles_list = json.load(file)

    loader = DirectoryLoader(source_html_path, glob="*.html", loader_cls=TextLoader)
    docs = loader.load()
    for doc in tqdm(docs, desc="Cleaning docs"):
        doc.metadata["title"] = get_title_drias(doc, titles_list)
        doc.metadata["url"] = get_url(doc.page_content)
        doc.page_content = clean_drias_html(doc.page_content)
        doc.page_content = md(doc.page_content, heading_style="ATX", strip=["a"])
        doc.page_content = clean_drias_doc(doc.page_content)
    print(f"{len(docs)} html files loaded from {source_html_path}")
    return docs


def create_docs_pdf(source_pdf_path: Path) -> List[Document]:
    """Load every .pdf file in the source directory into a langchain document"""
    print(f"Loading pdf files from {source_pdf_path}...")
    docs = []
    for source_pdf_path in source_pdf_path.glob("*.pdf"):
        md_text = pymupdf4llm.to_markdown(source_pdf_path)
        clean_text = clean_drias_doc(md_text)
        clean_text = re.sub(r"(?<![.!?:])\n+(?=[a-zà-ÿ0-9 '\(])", " ", clean_text)
        clean_text = re.sub(r".{4,}", " ", clean_text)
        doc = Document(page_content=clean_text)
        doc.metadata["title"] = Path(source_pdf_path).stem
        doc.metadata["url"] = Path(source_pdf_path).name
        docs.append(doc)
    return docs


def create_docs(path_data: Path, use_pdf: bool) -> List[Document]:
    """Load every document in the source directory into a langchain document"""
    html_paths = path_data / "HTMLs"
    if not use_pdf:
        return create_docs_html(html_paths)
    pdf_paths = path_data / "PDFs"
    return create_docs_html(html_paths) + create_docs_pdf(pdf_paths)


def split_to_paragraphs(docs: List[Document]):
    """Split Markdown documents to paragraphs using md headers."""
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    chunks = []
    for doc in tqdm(docs, desc="Splitting docs to paragraphs"):
        md_header_splits = markdown_splitter.split_text(doc.page_content)
        for split in md_header_splits:
            split.metadata["title"] = doc.metadata["title"]
            split.metadata["url"] = doc.metadata["url"]
            chunks.append(split)
    print(f"{len(chunks)} paragraphs loaded")
    return chunks


def split_to_chunks(docs) -> list[str]:
    """Split big documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "* ", "   ", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    chunks = [chunk for chunk in chunks if len(chunk.page_content) > 50]
    for chunk in chunks:
        if chunk.page_content.startswith("."):
            chunk.page_content = re.sub(r"^.\s*", "", chunk.page_content)

        header = chunk.metadata["title"].split(" > ")[-1]
        for metadata in chunk.metadata:
            if metadata != "title" and metadata != "url":
                header += " > " + chunk.metadata[metadata]
        chunk.page_content = "Titre: " + header + ".  \n" + chunk.page_content

    print(f"{len(chunks)} chunks loaded")
    return chunks


def filter_similar_chunks(
    chunks: List[Document], embedding: Embedding, threshold: float = 0.98
) -> List[Document]:
    """Returns a list of chunks with a similarity below a threshold"""
    chunks_embeddings = [
        embedding.encode(chunk.page_content)
        for chunk in tqdm(chunks, desc="Filtering chunks")
    ]
    mat_sim = cosine_similarity(chunks_embeddings, chunks_embeddings)
    idx_to_remove = []
    for i in range(len(chunks) - 1):
        for j in range(i + 1, len(chunks)):
            if mat_sim[i, j] > threshold:
                idx_to_remove.append(i)
                break
    unique_chunks = [chunks[i] for i in range(len(chunks)) if i not in idx_to_remove]
    print(f"{len(unique_chunks)} unique chunks loaded")

    return unique_chunks


def print_doc(doc: Document):
    print(f"-- Page Title : {doc.metadata['title']} ---")
    print(doc.page_content)
