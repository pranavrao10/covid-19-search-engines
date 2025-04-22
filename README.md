# covid-19-search-engines

### Introduction
Hello Dr Zhang and demonstrators, welcome to our search engines space! Because our project aims to determine if BM25 or Dirichlet Language Model (DLM) is better for retrieving COVID-19 literature, we implemented two search engines (BM25 + BioBERT re-ranking and DLM + BioBERT re-ranking). Below are detailed instructions on how to run our implemented search engines for your testing purposes. Since it is a comparison, we have also implemented the `BM25_and_LM_evaluation.ipynb` notebook that generates the evaluation metrics we showcased in the presentation, in order to determine which is truly better. This is also available for your testing purposes should you wish to recreate our results. The notebook also contains pretty visualisations so we implore you to check it out! Hope you enjoy :)

### Queries for Testing
Please click on this [link here](https://ir.nist.gov/trec-covid/data/topics-rnd5.xml) to access all the queries we used to generate retrieval results for evaluation purposes. Feel free to use any query from this webpage. We are proud of our search engine implementations as it handles all the different query types found at that link, demonstrating robustness !

### Instructions on How to Run Dirichlet LM Search + BioBERT Search Engine
1) Run `dirichlet_search.py` on your terminal and everything should work (fingers crossed). You will be prompted to enter queries 
2) Go crazy!! The search engine will not terminate until you hit keyboard interupt or type in 'exit'.
NOTE: The zip file that you will download already contains the pre-computed index for DLM. Optionally if you want to generate the index yourself, please first delete all the files in `dirichlet_lm_index_variables` **EXCEPT stop_words.json** (please never delete this :( ). Then, please run `dirichlet_lm_index_builder.py` before proceeding to step 1. It takes about a minute to build the DLM index, so you are welcomed to build it to verify our implementation :)

### Instructions on How to Run BM25 + BioBERT Search Engine
1) Run `bm25+biobert-search-engine.py` on your terminal and it should load up (again, fingers crossed).
2) The search engine will not terminate until you hit keyboard interupt or type in 'exit'.
NOTE: As outlined in the instructions beforehand, all the necessary indexes are already pre-computed so that you can run the search engine `.py` files directly. Again, if you wish to build the index for this search engine yourself, please delete all files in the `bm25+biobert files` and just run `build_indexes.ipynb` before proceeding to step 1. This takes longer much longer than the index building for DLM as it generates the embeddings for the documents (with which both search engine implementation relies on) as well.

### Instructions on How to Generate Evaluation Metrics
The `BM25_and_LM_evaluation.ipynb` notebook already contains the generated results (including the aforementioend pretty visuals at the bottom) for your inspection. However, if you wish to generate these results yourself for testing purposes, simply re-execute all the cells within `BM25_and_LM_evaluation.ipynb`. This notebook reads data (top 100 retrieval results per query for both search engines) in the `trec_covid_query_results` folder. Because we aim to prove that our results are genuine and not generated using some fishy method, we also provide the option for you to generate the retrieval results with which we use to compute metrics. To do so, kindly: 
1) Delete all files in the `trec_covid_query_results` folder
2) Run `produce_trec_query_results.py` in your terminal (this takes a while as the script intialises both search engines before feeding them all 50 queries from the link we exposed previously)
3) Execute all the cells in `BM25_and_LM_evaluation.ipynb`.

We appreciate that this is a very lengthy README doc so we thank you in advance for reading this and for your time. This was inevitable as we had to implement a lot of code to produce our results! *SPOILER ALERT* we found that BM25 retrieval was better overall :)
