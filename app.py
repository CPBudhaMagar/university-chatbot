from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import faiss
from flask import Flask, request, jsonify, send_file
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# === Load CSV Data ===
df = pd.read_csv("university_data.csv")
if 'text' not in df.columns:
    raise ValueError("CSV must contain a 'text' column.")
all_text = " ".join(df['text'].dropna().astype(str))

# === Split Text ===
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

chunks = split_text(all_text)

# === Embeddings + FAISS ===
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# === Retrieval + QA ===
def retrieve_chunks(query, k=3):
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

def ask_qa(context, query):
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}"
    result = qa_pipeline(prompt, max_length=100, do_sample=False)
    return result[0]['generated_text']

def generate_follow_ups(answer):
    return f"""1. Can you explain more about '{answer}'?
2. Where can I find more information related to '{answer}'?
3. Is there a deadline or policy related to '{answer}'?
"""

def university_chatbot(query):
    context_chunks = retrieve_chunks(query)
    context = "\n\n".join(context_chunks)
    answer = ask_qa(context, query)
    followups = generate_follow_ups(answer)
    return answer, followups

# === Flask App ===
app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get('message', '')
    if not user_query:
        return jsonify({'error': 'No message provided'}), 400
    answer, followups = university_chatbot(user_query)
    return jsonify({'response': answer, 'follow_up': followups})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    app = Flask(__name__)

@app.route('/')
def home():
    return send_file('index.html')
