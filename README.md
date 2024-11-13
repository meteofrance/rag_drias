# RAG DRIAS

Our goal is to make a Retrieval Augmented Generation (RAG) on the [DRIAS portal](https://www.drias-climat.fr/).

>LLMs used in specialized fields may create hallucinations due to their lack of knowledge. RAG helps solve this problem by retrieving relevant documents from external knowledge bases.

## Repository Structure
```
rag_drias
└─── docs
└─── rag_drias
│   └─── data.py          # text data management
│   └─── embedding.py     # wrapper for embedding models
│   └─── crawl.py         # website crawling tools
│   └─── settings.py      # settings (paths, model names,...)
└─── main.py              # Main python script
```
## Documentation

Full code documentation of Rag_drias can be found here.

- [1 - Data download](docs/1_download_data.md)
- [2 - Database construction](docs/2_database_construction.md)
- [3 - Retrieval](docs/3_retrieval.md)
- [4 - Generation](docs/4_generation.md)


## Install

```git clone https://github.com/meteofrance/rag_drias.git```

Build conda environment:

```conda env create --file environment.yaml```
```conda activate ragdrias```

You have to download manually the different models :

1. ```cd /my/large/folder/```
2. ```git lfs install```
3. Embedding model: ```git clone https://huggingface.co/dangvantuan/sentence-camembert-large```
4. Generation model: ```git clone https://huggingface.co/jpacifico/Chocolatine-14B-Instruct-4k-DPO``` or
```git clone https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct```
5. Reranker model: ```git clone https://huggingface.co/BAAI/bge-reranker-v2-m3```

## Usage

1. Prepare the vector database :

```bash
python main.py prepare
```

2. Make a query and retrieve the most relevant samples :

```bash
python main.py query "Quels formats de données sont disponibles pour le téléchargement sur DRIAS ?"
```

3. Make a query and retrieve the answer :

```bash
python main.py answer "Quels formats de données sont disponibles pour le téléchargement sur DRIAS ?"
```

Use `--help` to see all available options in the `main.py` script.
