# Data download

We download htmls pages and pdfs from the Drias site. Currently, 241 HTML pages and 44 PDFs are available on ewc in `/scratch/shared/Chatbot_DRIAS/text_data`.

## Web crawling

Web crawling consists of browsing the web in an automated manner by following links to download web pages or files. The crawler starts with a given page (seed url) and follows the links found on this page to access other pages or files. The same process is applied for each page found. 
 
In the case of Rag Drias we start from the home page of the site (https://www.drias-climat.fr/), and we add to the queue the pages provided in the json menu (https://www.drias-climat.fr/accompagnement/getAllTopSectionsJson) of the site in order to access interesting information more quickly.

### Usage

crawl.py has a max_depth argument which corresponds to the maximum depth relative to the home page. So the home page has a depth of 0 and a page linked in the home page has a depth of 1.

```bash
python rag_drias/crawl.py --max_depth 3
```
*By default max_depth=3 and files are saved in /scratch/shared/Chatbot_DRIAS/text_data*

### Exemple

- for a depth of 3 :
```bash
[ferreiram@dev-ia-mathilde rag_drias]$ runai exec python rag_drias/crawl.py
--> json menu saved in /scratch/shared/Chatbot_DRIAS/text_data/Drias/getAllTopSectionsJson.json
Number of pages downloaded : 213
execution time : 252.1725
```
- for a depth of 4 :
```bash
[ferreiram@dev-ia-mathilde rag_drias]$ runai exec python rag_drias/crawl.py
--> json menu saved in /scratch/shared/Chatbot_DRIAS/text_data/Drias/getAllTopSectionsJson.json
Number of pages downloaded : 242
execution time : 268.90387
```


