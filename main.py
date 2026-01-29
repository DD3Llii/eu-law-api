from fastapi import FastAPI
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

# Dwing de server om minder geheugen te gebruiken
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

app = FastAPI()

# Het allerkleinste model dat er bestaat
model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']
    index = data['index']

@app.get("/")
def home():
    return {"status": "Online en klaar voor gebruik."}

@app.get("/ask")
def ask_law(query: str):
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector).astype('float32'), k=3)
    answers = [chunks[i] for i in I[0]]
    return {"query": query, "top_answers": answers}
