from fastapi import FastAPI
import pickle
import numpy as np
import requests

app = FastAPI()

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']
    embeddings = np.array(data['embeddings']).astype('float32')

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

@app.get("/")
def home():
    return {"status": "Online - Safe Mode Active"}

@app.get("/ask")
def ask_law(query: str):
    response = requests.post(API_URL, json={"inputs": query})
    query_vector = np.array(response.json()).astype('float32')
    
    # Zoeken met Numpy (vlijmscherp en crasht nooit)
    distances = np.linalg.norm(embeddings - query_vector, axis=1)
    top_indices = np.argsort(distances)[:3]
    
    answers = [chunks[i] for i in top_indices]
    return {"query": query, "top_answers": answers}
