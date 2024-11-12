# Generation

The generation part consists of gathering the retrieved chunks and the user's question and asking an llm to answer. To do this, you must ensure that you respect the llm instruction template and give it good instructions.

We can use either the Chocolatine model because it was trained in French and is on the open LLM leaderboard of HuggingFace or the llama 3.2 3b instruct trained in 8 language including french which is a instruction-tuned generative model. 

> the Chocolatine model : https://huggingface.co/jpacifico/Chocolatine-14B-Instruct-4k-DPO
> the llama 3.2 3b model : https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

## Usage

```bash
python main.py answer "Peut-on obtenir des projections climatiques spécifiques à un département français sur DRIAS ?" --data-source Drias --n-samples 10 --use-rag --use-rerank
```
*n_samples is the number of retrived chunks*

### Exemples
- without rag : 
```bash
ferreiram@dev-ia-mathilde:~/monorepo4ai/projects/rag_drias$ python main.py answer "Quels formats de données sont disponibles pour le téléchargement sur DRIAS ?" --data-source Drias --no-use-rag
##################################################
LLM input:
<|system|>
Le portail DRIAS mets à disposition les projections climatiques régionalisées de référence, pour l'adaptation en France. Tu es un chatbot qui reponds aux questions sur le site.<|end|>
<|user|>
Réponds à cette question: Quels formats de données sont disponibles pour le téléchargement sur DRIAS ?<|end|>
<|assistant|>

##################################################
LLM output:
 DRIAS propose plusieurs formats de données pour le téléchargement, notamment des fichiers CSV, Excel, et parfois des fichiers JSON ou XML, selon les projections climatiques spécifiques et les régions concernées. Les utilisateurs peuvent accéder à ces données via le portail en sélectionnant les paramètres de recherche appropriés et en cliquant sur le bouton de téléchargement.
```
- with rag : 
```bash
ferreiram@dev-ia-mathilde:~/monorepo4ai/projects/rag_drias$ python main.py answer "Quels formats de données sont disponibles pour le téléchargement sur DRIAS ?" --data-source Drias
...
-- Page Title : DRIAS, Les futurs du climat - Accompagnement > Les données disponibles > Expériences atmosphériques > Simulations Métropole > Simulations EXPLORE2-Climat 2022 ---
-- url : https://www.drias-climat.fr/accompagnement/sections/354 ---
Les simulations sont disponibles au format NetCDF dans l'espace "Données et Produits", au pas de temps journalier, couvrant la période continue 1971-2100 et pour 8 à 11 variables voir le tableau ci-dessous (tableau 3\).  
*Table 2 : Liste des variables corrigées et disponibles sur le portail DRIAS.*
Réponds à cette question de manière claire et concise en indiquant les urls des informations si utilisées: Quels formats de données sont disponibles pour le téléchargement sur DRIAS ?
Réponse:<|end|>
<|assistant|>

##################################################
LLM output:
 Les formats de données disponibles pour le téléchargement sur DRIAS sont le format NetCDF et le format CSV.
```