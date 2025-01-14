import json
import re
import ssl
import urllib.parse
from collections import deque
from typing import List

import requests
from bs4 import BeautifulSoup

from rag_drias.settings import MENU_URL, PATH_DATA, PATH_MENU_JSON, SECTION_URL

ssl._create_default_https_context = ssl._create_unverified_context


def save_html(content: str, url: str):
    """Save HTML content in a file .html."""
    # Create a valid file name
    clean_url = re.sub(r"^https?://", "", url)
    clean_url = re.sub(r"[/\\]", "_", clean_url)
    filename = f"{clean_url}.html"
    save_path_html = PATH_DATA / "HTMLs"
    filepath = save_path_html / filename

    # Create the directory if necessary
    save_path_html.mkdir(parents=True, exist_ok=True)

    content = f"<!-- URL: {url} -->\n" + content
    # Writes HTML content to the file
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)


def download_pdf(url: str, visited_pdf: List[str]):
    """Save the pdf if not already done."""
    if url.split("//")[-1] in visited_pdf:
        return
    visited_pdf.add(url.split("//")[-1])

    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return

    save_path_pdf = PATH_DATA / "PDFs"
    filename = url.split("/")[-1]
    filepath = save_path_pdf / filename

    # Create the directory if necessary
    save_path_pdf.mkdir(parents=True, exist_ok=True)

    # Writes PDF content to the file
    with open(filepath, "wb") as file:
        file.write(response.content)


def add_menu_to_queue(json_menu: List[dict], queue: deque, depth: int = 1):
    """Add the urls in the menu to the queue"""
    for section in json_menu:
        id = section["id"]
        url = f"{SECTION_URL}/{id}"
        queue.append((url, depth))
        if "children" in section:
            add_menu_to_queue(section["children"], queue, depth + 1)


def crawl_website(start_url: str, max_depth: int = 3):
    """Download htmls and pdfs from the home page"""
    domain = urllib.parse.urlparse(start_url).netloc
    queue = deque([(start_url, 0)])
    visited = set()
    visited_pdf = set()

    # save the json main menu
    response_json = requests.get(MENU_URL, timeout=10)
    menu_json = response_json.json()
    with open(PATH_MENU_JSON, "w", encoding="utf-8") as file:
        json.dump(menu_json, file, ensure_ascii=False)

    # add urls of the main menu in depth 1
    add_menu_to_queue(menu_json, queue)
    print(f"{len(queue)} urls added to the queue")

    while queue:
        current_url, depth = queue.popleft()
        # nothing to do if the depth is exceeded or the URL has already been visited

        if depth > max_depth or current_url.split("//")[-1] in visited:
            continue

        visited.add(current_url.split("//")[-1])

        response = requests.get(current_url, timeout=10)
        if response.status_code != 200:
            continue

        html_content = response.text
        save_html(html_content, current_url)

        if max_depth > depth:
            soup = BeautifulSoup(html_content, "html.parser")

            # add the HTML and PDF links in the queue if not visited and in the domain
            for link in soup.find_all("a", href=True):
                full_url = urllib.parse.urljoin(current_url, link["href"])
                if domain in full_url and link["href"] != "/":
                    if full_url.endswith(".pdf"):
                        # Download pdf
                        download_pdf(full_url, visited_pdf)
                    # if no extension and not already visited
                    elif (len(full_url.split("/")[-1].split(".")) == 1) and (
                        full_url.split("//")[-1] not in visited
                    ):
                        queue.append((full_url, depth + 1))
    print(f"Number of HTMLs pages downloaded : {len(visited)}")
    print(f"Number of PDFs downloaded : {len(visited_pdf)}")
