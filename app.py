from flask import Flask, request, render_template_string
import pandas as pd
import re

# === Load CSV ===
df = pd.read_csv("university_data.csv")
df['question'] = df['question'].astype(str).str.strip()
df['answer'] = df['answer'].astype(str).str.strip()
questions = df['question'].tolist()
answers = df['answer'].tolist()

# === Initialize App ===
app = Flask(__name__)

# === Text Normalizer ===
def normalize(text):
    return re.sub(r'[^\w\s]', '', text.strip().lower())

# === Exact Match Function ===
def is_exact_match(query):
    nq = normalize(query)
    for i, q in enumerate(questions):
        if normalize(q) == nq:
            return True, answers[i]
    return False, None

# === Related Questions (Substring Matching) ===
def get_related_questions(query, limit=10):
    keywords = normalize(query).split()
    related = []
    seen = set()
    for q in questions:
        q_norm = normalize(q)
        if any(k in q_norm for k in keywords) and q not in seen:
            related.append(q)
            seen.add(q)
    return related[:limit]

# === Bootstrap-based Frontend ===
HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>University Chatbot</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: #f4f6f9;
            font-family: 'Segoe UI', sans-serif;
            padding-top: 50px;
        }
        .container {
            max-width: 600px;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }
        .chat-title {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 20px;
            text-align: center;
            color: #2c3e50;
        }
        .form-control {
            font-size: 18px;
        }
        .response, .suggestions {
            margin-top: 20px;
        }
        .suggestions ul {
            padding-left: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="chat-title">🎓 University Chatbot</div>
        <form method="post">
            <div class="mb-3">
                <input name="query" class="form-control" placeholder="Ask your question..." required>
            </div>
            <button type="submit" class="btn btn-primary w-100">Ask</button>
        </form>

        {% if response %}
        <div class="response alert alert-success mt-4">
            <strong>🤖 Bot:</strong> {{ response }}
        </div>
        {% endif %}

        {% if suggestions %}
        <div class="suggestions alert alert-info">
            <strong>Related questions you might be looking for:</strong>
            <ul>
                {% for q in suggestions %}
                <li>{{ q }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
    </div>

    <div class="text-center mt-4">
        <small class="text-muted">created by cp</small>
    </div>
</body>
</html>
'''

# === Route ===
@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    suggestions = []
    if request.method == "POST":
        query = request.form["query"]
        match, answer = is_exact_match(query)
        if match:
            response = answer + " Would you like to ask another question?"
        else:
            response = "I couldn’t find an exact answer."
            suggestions = get_related_questions(query)
    return render_template_string(HTML, response=response, suggestions=suggestions)

# === Run App ===
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
