from pathlib import Path

URLS = {
    "Drias": "https://www.drias-climat.fr",
}

PATH_SCRATCH = Path("/scratch/shared/Chatbot_DRIAS/")
PATH_DATA = PATH_SCRATCH / "text_data"
PATH_MENU_JSON = PATH_DATA / "Drias/getAllTopSectionsJson.json"
PATH_VDB = PATH_SCRATCH / "chroma_database"
PATH_MODELS = Path("/scratch/shared/RAG/") / "hf_models"
# PATH_LLM = PATH_SCRATCH / "hf_models" / "Chocolatine-3B-Instruct-DPO-v1.0"
PATH_LLM = PATH_SCRATCH / "hf_models" / "Llama-3.2-3B-Instruct"
PATH_RERANKER = PATH_SCRATCH / "hf_models" / "bge-reranker-v2-m3"
