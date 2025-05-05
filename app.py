
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load and process CSV
df = pd.read_csv("university_chatbot_formatted.csv")
all_text = " ".join(df["text"].dropna().astype(str))

def split_text(text, chunk_size=500, overlap=100):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

chunks = split_text(all_text)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

def ask_qa(context, query):
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}"
    return qa_pipeline(prompt, max_length=100, do_sample=False)[0]['generated_text']

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('message', '')
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, 3)
    context = "\n\n".join([chunks[i] for i in indices[0]])
    answer = ask_qa(context, query)
    followups = f"1. Can you explain more about '{answer}'?\n2. Where can I find details?\n3. Any deadlines?"
    return jsonify({"response": answer, "follow_up": followups})

if __name__ == '__main__':
    app.run(port=5000)
