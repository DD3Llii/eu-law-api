from fastapi import FastAPI
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

app = FastAPI()

# 1. Het model laden (het 'brein')
model = SentenceTransformer('paraphrase-albert-small-v2')

# 2. Jouw product laden (de data)
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']
    index = data['index']

@app.get("/")
def home():
    return {"status": "De EU Law API is online en klaar voor verkoop."}

@app.get("/ask")
def ask_law(query: str):
    # Vraag omzetten naar getallen
    query_vector = model.encode([query])
    
    # Zoek de 3 beste antwoorden in de 836 units
    D, I = index.search(np.array(query_vector).astype('float32'), k=3)
    
    answers = [chunks[i] for i in I[0]]
    return {
        "query": query,
        "top_answers": answers
    }
