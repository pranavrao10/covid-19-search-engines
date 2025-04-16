import json
import numpy as np
import string
#from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz


#The below code block removes punctuation from document body texts, extracts document ids, and extracts document titles. 
#These three variables are then returned.
def process_documents(documents:list) -> list:
    #stemmer = SnowballStemmer("english")
    document_ids = []
    body_texts = []
    for document in documents:
        document_ids.append(document['doc_id'])
        #below should be uncommented if stemming is to be included
        #cleaned_body_text = document['contents'].translate(str.maketrans('', '', string.punctuation)).split()
        #cleaned_body_text = ' '.join([stemmer.stem(word) for word in cleaned_body_text])
        body_texts.append(document['cleaned_text'].translate(str.maketrans('', '', string.punctuation)))
    return document_ids, body_texts

#The below code block builds a term frequency matrix and creates arrays containing values that are necessary to calculate
#the dirichlet prior scores (p(q|d)) for each document.
def build_index(document_body_texts:list) -> np.ndarray:
    vectorizer = CountVectorizer()
    term_frequency_matrix = vectorizer.fit_transform(document_body_texts)
    vocabulary = vectorizer.vocabulary_
    probability_of_term_in_collection_multiplied_with_mu = (term_frequency_matrix.sum(axis = 0)/term_frequency_matrix.sum()).A1 * 2000
    document_lengths_plus_mu = np.reshape(term_frequency_matrix.sum(axis = 1), (term_frequency_matrix.shape[0], -1)) + 2000
    return term_frequency_matrix, vocabulary, probability_of_term_in_collection_multiplied_with_mu, document_lengths_plus_mu

print('')
print('Please enter the full filepath where the \'trec_covid_preprocessed_minimal.json\' file is located.')
filepath = input('Full Filepath:')
print('...Processing...')
with open(filepath) as file:
    dataset = json.load(file)

document_ids, body_texts = process_documents(dataset)
term_frequency_matrix, vocabulary, probability_of_term_in_collection_multiplied_with_mu, document_lengths_plus_mu = build_index(body_texts)
data = {'document_ids': document_ids, 'vocabulary': vocabulary}#storing these like so, as it is easier to dump them in json files and load them later

#The following code block saves the necessary pre-computed variables (arrays, scipy.sparse_matrix, and dictionaries) which 
#will be loaded in main.py and used to calculate dirichlet prior scores (p(q|d)). This ensures that computing p(q|d)
#upon receival of a query is fast as it is not bogged down by index building, which takes quite a bit of time.
with open('dirichlet_lm_index_variables/document_ids_vocabulary_titles.json', 'w') as file:
    json.dump(data, file)
np.savez('dirichlet_lm_index_variables/necessary_arrays_for_index_building.npz', 
         mu_scaled_term_collection_probability = probability_of_term_in_collection_multiplied_with_mu, 
         mu_scaled_document_lengths = document_lengths_plus_mu)
save_npz('dirichlet_lm_index_variables/term_frequency_matrix.npz', term_frequency_matrix)

print('DONE. All necessary variables created!')
