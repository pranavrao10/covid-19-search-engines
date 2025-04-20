print('\n........Initialising Search Engine. Please wait :)........')
import re
import json
import numpy as np
import torch
from nltk.stem import PorterStemmer
from scipy.sparse import load_npz
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


#The code below loads the list of stop words, instantiates a RegEx object for stripping punctuation, and instantiates a Porter Stemmer object.
#These variables are used by the 'process_query()' function which ensures the user input query is in a suitable format for the retrieval function
with open('../covid-19-search-engines/dirichlet_lm_index_variables/stop_words.json', 'r') as file:
    stop_words = json.load(file)['stop_words']
strip_punctuation = re.compile(r"\b[\w]+\b")
stemmer = PorterStemmer()

#The code below loads the necessary data stored in arrays, scipy.sparse_matrix, and a python dictionary to calculate dirichlet prior scores (p(q|d))
loaded_arrays_for_index_building = np.load('dirichlet_lm_index_variables/necessary_arrays_for_index_building.npz')
probability_of_term_in_collection_multiplied_with_mu = loaded_arrays_for_index_building['mu_scaled_term_collection_probability']
document_lengths_plus_mu = loaded_arrays_for_index_building['mu_scaled_document_lengths']
term_frequency_matrix = load_npz('dirichlet_lm_index_variables/term_frequency_matrix.npz')
with open('dirichlet_lm_index_variables/document_ids_titles_snippets_vocabulary.json', 'r') as file:
    data = json.load(file)
vocabulary = data['vocabulary']
document_ids = np.array(data['document_ids']) #stored as a np.ndarray to allow ease of retrieval of top 100 document IDs using multi indices 
document_titles = data['document_titles']
abstract_snippets = data['abstract_snippets']

#The code below loads the pre-computed embeddings for all the documents and initialises a biobert model to be used for reranking
document_embeddings = np.load('bm25+biobert files/biobert_embeddings.npy', mmap_mode='r')    
tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
model = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed")
model.eval() #because we only need to generate embeddings for query and no training is necessary

def process_query(query:str) -> list:
    processed_query = strip_punctuation.findall(query.lower())
    processed_query = [stemmer.stem(word) for word in processed_query if word not in stop_words]
    return processed_query

#This code block calculates dirichlet prior scores (p(q|d)) for each document using the loaded variables when given a query
def calculate_dirichlet_scores(query:str) -> np.ndarray:
    indices_of_query_terms = [vocabulary[term] for term in process_query(query)]
    relevant_term_frequencies = term_frequency_matrix[:, indices_of_query_terms]
    relevant_mu_scaled_term_probabilities = probability_of_term_in_collection_multiplied_with_mu[indices_of_query_terms]
    relevancy_scores_per_document = ((relevant_term_frequencies + relevant_mu_scaled_term_probabilities)/document_lengths_plus_mu).sum(axis = 1)
    return relevancy_scores_per_document.A1

#This code block fetches the corresponding embeddings of the top 100 retrieved documents and re-ranks them using cosine similarity
#between embedded query and embedded documents
def biobert_reranker(query:str, relevancy_scores:list) -> list:
    with torch.no_grad(): 
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        embedded_query = model(**inputs).last_hidden_state[:,0,:].numpy()
    indices_of_top_100_document_relevancy_scores = np.argsort(relevancy_scores)[::-1][:100]
    corresponding_embeddings = document_embeddings[indices_of_top_100_document_relevancy_scores]
    similarity_between_query_and_top_100_documents = cosine_similarity(embedded_query, corresponding_embeddings).flatten()
    return indices_of_top_100_document_relevancy_scores[np.argsort(similarity_between_query_and_top_100_documents)[::-1]]

#This code block below combines the dirichlet prior scores (p(q|d)) with their corresponding document IDs
#and returns the results in descending order. Note, only the top 20 results are shown for brevity purposes.
def dirichlet_searcher(query:str, top_k_results:int = 5) -> np.ndarray:
    relevancy_scores_per_document = calculate_dirichlet_scores(query)
    biobert_reranked_indices_of_top_100_documents = biobert_reranker(query, relevancy_scores_per_document)
    for index in biobert_reranked_indices_of_top_100_documents[:top_k_results]:
        print(document_titles[index])
        print('-' * len(document_titles[index]))
        print(f'ID: {document_ids[index]}      P(Q|D): {round(relevancy_scores_per_document[index], 6)}')
        print('Snippet:')
        print(abstract_snippets[index], '\n','\n')
    return document_ids[biobert_reranked_indices_of_top_100_documents]

print('\nWelcome to the Dirichlet Searcher for Covid-19 papers!')
print('Hit keyboard interrupt or type in \'exit\' to quit the search engine.')
#The while loop is so the user can keep searching for articles until she/he/they gets tired and hits keyboard interrupt :)
while True:
    print('')
    print('Please enter your query below.....')
    query = input('Search:')
    if query =='exit':
        break
    print('\nTop 5 Results:\n')
    results = dirichlet_searcher(query)
