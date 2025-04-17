import json
import numpy as np
import pickle
import time
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

class COVIDSearchEngine:
    def __init__(self):
        self.load_resources()
        
    def load_resources(self):
        with open('bm25_model.pkl', 'rb') as f:
            self.bm25 = pickle.load(f)
        
        with open('metadata.json') as f:
            self.metadata = json.load(f)
            
        self.doc_embeddings = np.load('biobert_embeddings.npy', mmap_mode='r')
        
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
        self.model = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed")
        self.model.eval()
        
    def search(self, query, top_k=5, bm25_first_n=100):
        start_time = time.time()
        
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(doc_scores)[-bm25_first_n:][::-1]
        
        with torch.no_grad():
            inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
            query_embed = self.model(**inputs).last_hidden_state[:,0,:].numpy()
        
        top_embeddings = self.doc_embeddings[bm25_top_indices]
        similarities = cosine_similarity(query_embed, top_embeddings).flatten()
        
        combined_scores = 0.6 * doc_scores[bm25_top_indices] + 0.4 * similarities
        final_indices = bm25_top_indices[np.argsort(combined_scores)[::-1][:top_k]]
        
        print(f"Search completed in {time.time() - start_time:.2f}s")
        return [self.metadata[i] for i in final_indices]

if __name__ == '__main__':
    se = COVIDSearchEngine()
    print("COVID-19 Hybrid Search Engine Ready!\n")
    
    while True:
        query = input("Enter your search query (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        results = se.search(query)
        print(f"\nTop {len(results)} results:")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['original_title']}")
            print(f"   Doc ID: {res['doc_id']}")
            print(f"   Snippet: {res['biobert_text'][:150]}...\n")