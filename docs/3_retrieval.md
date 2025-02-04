# Retrieval

Retrieval consists of retrieving the most relevant chunks in relation to the query. To do this, we use a hybrid search which includes  a lexical search (sparse search) and a semantic search (dense search).

* **Lexical search :**  
We use the BM25 algorithm to retrieve the *k* most relevant chunks from a collection based on a given query by ranking them according to their term frequency and the lenght of the chunk.

*  **Semantic search :**   
We can use algorithms such the Maximum Marginal Relevance which consists of taking the *k\*2* closest chunks and then selecting the *k* farest chunks from among them.  
In our case, we use cosine similarity search to select the chunks closest to the query in the embedding space.

After that, we combine weight the scores of the chunks retrieved by the two methods. So we have at most *2\*k* chunks and we only keep the *k* chunks with the highest scores.

## Re-Ranking
Re-ranking is an additional step that can be performed to enhance the accuracy of the retrieved information. It consists of comparing the similarity between the retrieved chunks and the query in order to only give the user the *k* most relevant chunks.  
In our case, we then apply a dynamic threshold to not retrieve chunks with a low relevance score.

We use the BGE reranker model because it is a multilingual model.

> the BGE reranker model : https://huggingface.co/BAAI/bge-reranker-v2-m3

## Usage

```bash
python main.py query "Quels formats de données sont disponibles pour le téléchargement sur DRIAS ?" --n-samples 4 --use-rerank
```
*n_samples is the number of retrived chunks*

### Exemple
```bash
ferreiram@dev-ia-mathilde:~/monorepo4ai/projects/rag_drias$ python main.py query "Peut-on obtenir des projections climatiques spécifiques à un département français sur DRIAS ?" --n-samples 4
---> Relevant chunk 0 <---
-- Page Title : DRIAS, Les futurs du climat - Accompagnement > Drias[CLIMAT] > Contexte ---
Les projections climatiques distribuées via le Climate Data Store (CDS) de Corpernicus, sont issus de modèles climatiques globaux et régionaux. Ainsi dans le cadre de travaux sur l’adaptation de nos territoires au changement climatique, ces données nécessitent une correction de biais pour être utilisées à l’échelle locale, c’est ce que propose le portail DRIAS sur la France.
--------------------
...
```
