# RAG DRIAS

The goal is to make a Retrieval Augmented Generation (RAG) on DRIAS https://www.drias-climat.fr/.

>LLMs used in specialized fields may create hallucinations due to their lack of knowledge. RAG helps solve this problem by retrieving relevant documents from external knowledge bases.

## Structure du dossier
```
rag_drias
└─── rag_drias                         # Custom Library
│   └─── data.py
│   └─── embedding.py
│   └─── crawl.py
│   └─── settings.py
└─── main.py
```
## Documentation

Full product code documentation of Rag_drias can be found here.

- [1 - Data download](docs/1_download_data.md)
- [2 - Database construction](docs/2_database_construction.md)
- [3 - Retrieval](docs/3_retrieval.md)
- [4 - Generation](docs/4_generation.md)

## Data

241 HTML pages and 44 PDFs from www.drias-climat.fr are available on ewc in `/scratch/shared/Chatbot_DRIAS/text_data`.

## Models

You have to download manually the embedding models :
1. ```runai exec```
2. ```cd /scratch/<PATH>/hf_models```
3. ```git lfs install```
4. ```git clone https://huggingface.co/dangvantuan/sentence-camembert-large```
5. ```git clone https://huggingface.co/jpacifico/Chocolatine-14B-Instruct-4k-DPO``` or  
```git clone https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct```
6. ```git clone https://huggingface.co/BAAI/bge-reranker-v2-m3```

## Usage

1. Start by building the Docker image from the `rag_drias` folder:

```bash
runai build
```

2. Start the container from the `rag_drias` folder:

```bash
runai exec_gpu
```

3. Prepare the vector database :

```bash
python main.py prepare --data-source Drias
```

4. Make a query and retrieve the most relevant samples :

```bash
python main.py query "Quels formats de données sont disponibles pour le téléchargement sur DRIAS ?" --data-source Drias
```

5. Make a query and retrieve the answer :

```bash
python main.py answer "Quels formats de données sont disponibles pour le téléchargement sur DRIAS ?" --data-source Drias
```

Use `--help` to see available options in the `main.py` script.
