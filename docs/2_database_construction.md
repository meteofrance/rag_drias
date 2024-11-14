# Database construction

We create a Vector Database (VDB) from the HTMLs pages of Drias site. The VDB will serve as the foundation for the Retrieval Augmented Generation (RAG) technique. The VDB will store vector representations of the documents, allowing us to efficiently retrieve relevant information during the generation phase.

## Prepare data

We clean the documents by removing repetitive parts such as headers and footers from the page in order to keep the interesting content of the page. References to images are also removed. We then switch to markdown and remove the unnecessary characters produced by the conversion.
We add metadata such as the URL and the path of the titles leading to the page.

## Separate documents into chunks

To be able to find relevant information more easily in relation to a query, we separate the documents into small chunks. To avoid dividing the information, we try to split the document by paragraph or sentence if possible.
However, it may happen that some chunks are repeated (due to repetitions on different HTML pages), that causes problems during the retrieval part. Thus, we select chunks if their cosine similarity is below a certain threshold.

## Create a vector database

We use an embedding model to pass our chunks into a vector space.

> The Camembert embedding model: https://huggingface.co/dangvantuan/sentence-camembert-large

We chose this model because it is trained in French and well placed in the MTEB Leaderboard of HuggingFace.

### Usage

```bash
python main.py prepare-database
```
*Create a chroma database in /scratch/shared/rag_drias/chroma_database/sentence-camembert-large*
