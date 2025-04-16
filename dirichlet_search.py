import json
import numpy as np
from scipy.sparse import load_npz


#The code below loads the necessary data stored in arrays, scipy.sparse_matrix, and a python dictionary to calculate dirichlet prior scores (p(q|d))
loaded_arrays_for_index_building = np.load('dirichlet_lm_index_variables/necessary_arrays_for_index_building.npz')
probability_of_term_in_collection_multiplied_with_mu = loaded_arrays_for_index_building['mu_scaled_term_collection_probability']
document_lengths_plus_mu = loaded_arrays_for_index_building['mu_scaled_document_lengths']
term_frequency_matrix = load_npz('dirichlet_lm_index_variables/term_frequency_matrix.npz')
with open('dirichlet_lm_index_variables/document_ids_vocabulary_titles.json', 'r') as file:
    data = json.load(file)
vocabulary = data['vocabulary']
document_ids = data['document_ids']

#This code block calculates dirichlet prior scores (p(q|d)) for each document using the loaded variables when given a query
def calculate_dirichlet_scores(query:list) -> dict:
    indices_of_query_terms = [vocabulary[term] for term in query]
    relevant_term_frequencies = term_frequency_matrix[:, indices_of_query_terms]
    relevant_mu_scaled_term_probabilities = probability_of_term_in_collection_multiplied_with_mu[indices_of_query_terms]
    relevancy_scores_per_document = ((relevant_term_frequencies + relevant_mu_scaled_term_probabilities)/document_lengths_plus_mu).sum(axis = 1)
    return relevancy_scores_per_document.A1

#This code block below combines the dirichlet prior scores (p(q|d)) with their corresponding document IDs
#and returns the results in descending order. Note, only the top 20 results are shown for brevity purposes.
def dirichlet_searcher(relevancy_scores:list) -> dict:
    results = dict(zip(document_ids, relevancy_scores))
    results = dict(sorted(results.items(), key = lambda x: x[1], reverse = True))
    for key, value in list(results.items())[:20]:
        print(f'Doc ID: {key}   P(Q|D): {value}')
    return dict(sorted(results.items(), key = lambda x: x[1], reverse = True))

print('Welcome to the Dirichlet Searcher for Covid-19 papers!')
#The while loop is so the user can keep searching for articles until she/he/they gets tired and hits keyboard interrupt :)
while True:
    print('')
    print('Please enter your query below.....')
    query = input('Search:')
    relevancy_scores_per_document = calculate_dirichlet_scores(query.split())
    search_results = dirichlet_searcher(relevancy_scores_per_document)
