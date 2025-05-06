# app.py — Flask Version for Render Deployment
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

# === Load CSV Data ===
df = pd.read_csv("university_data.csv")
df.columns = [col.lower().strip() for col in df.columns]
df = df.dropna(subset=['question', 'answer'])
df['question'] = df['question'].astype(str).str.strip()
df['answer'] = df['answer'].astype(str).str.strip()

questions = df['question'].tolist()
answers = df['answer'].tolist()

# === Embed Questions and Build FAISS Index ===
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = embed_model.encode(questions)
question_embeddings = np.array(question_embeddings).astype('float32')

index = faiss.IndexFlatL2(question_embeddings.shape[1])
index.add(question_embeddings)

# === Load FLAN-T5 Pipeline ===
qg_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# === Helper Functions ===
SIMILARITY_THRESHOLD = 0.90

def is_exact_match(query):
    vec = embed_model.encode([query])
    D, I = index.search(vec, 1)
    score = 1 - (D[0][0] / 2)
    if score >= SIMILARITY_THRESHOLD:
        idx = I[0][0]
        return True, questions[idx], answers[idx]
    return False, None, None

def get_csv_matches(query, max_results=10):
    keyword = query.lower().strip()
    matched = []
    for q in questions:
        if keyword in q.lower() and q not in matched:
            matched.append(q)
        if len(matched) >= max_results:
            break
    return matched

def generate_raqg_followups(matches):
    if not matches:
        return ["Sorry, no related content found to generate questions."]
    context = "\n".join(matches)
    prompt = (
        "Generate 3 helpful and distinct follow-up questions based on the following questions:\n"
        f"{context}\n\nList them as:\n1."
    )
    result = qg_pipeline(prompt, max_length=120, do_sample=False)
    lines = result[0]['generated_text'].strip().split("\n")
    followups = [line.strip() for line in lines if line.strip().startswith(tuple("123"))]
    return followups[:3] if followups else [result[0]['generated_text']]

# === Flask App ===
app = Flask(__name__)
conversation_state = {"awaiting_more_info": False}

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    global conversation_state

    if conversation_state["awaiting_more_info"]:
        conversation_state["awaiting_more_info"] = False
        match, q, a = is_exact_match(user_input)
        if match:
            return jsonify({"answer": a})
        else:
            matches = get_csv_matches(user_input)
            followups = generate_raqg_followups(matches)
            return jsonify({"answer": "I couldn't find an exact answer.", "follow_ups": followups})

    match, q, a = is_exact_match(user_input)
    if match:
        conversation_state["awaiting_more_info"] = True
        return jsonify({
            "answer": a,
            "follow_up": "Would you like to know more about us?"
        })
    else:
        matches = get_csv_matches(user_input)
        followups = generate_raqg_followups(matches)
        return jsonify({"answer": "I couldn’t find an exact answer.", "follow_ups": followups})

@app.route("/")
def home():
    return "✅ University Enquiry Chatbot API is running."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
