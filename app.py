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

# === HTML Frontend ===
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
            padding-top: 60px;
        }
        .container {
            max-width: 650px;
            background: white;
            padding: 35px;
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
        }
        .chat-title {
            font-size: 30px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 25px;
            color: #004d99;
        }
        .form-control {
            border-radius: 12px;
            padding: 14px;
            font-size: 17px;
        }
        .btn-primary {
            border-radius: 12px;
            font-size: 16px;
            padding: 12px;
            font-weight: 500;
        }
        .response {
            margin-top: 25px;
        }
        .response strong::before {
            content: "\1F916 ";
        }
        .suggestions ul {
            margin-top: 10px;
            padding-left: 20px;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            font-size: 14px;
        }
        .btn-outline-primary, .btn-outline-secondary, .btn-outline-dark {
            margin: 5px;
            border-radius: 20px;
            padding: 6px 16px;
            font-size: 14px;
        }
        #welcome {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #fff;
            z-index: 9999;
            font-size: 24px;
            font-weight: bold;
            color: #004d99;
            flex-direction: column;
            transition: opacity 1s ease;
        }
        #welcome.fade-out {
            opacity: 0;
            pointer-events: none;
        }
    </style>
   <script>
    function toggleDesignerInfo() {
        const info = document.getElementById('designer-info');
        info.style.display = (info.style.display === "none") ? "block" : "none";
    }

    window.onload = function () {
        if (!sessionStorage.getItem('welcomeShown')) {
            setTimeout(() => {
                document.getElementById('welcome').classList.add('fade-out');
                setTimeout(() => {
                    document.getElementById('welcome').style.display = 'none';
                    sessionStorage.setItem('welcomeShown', 'true');
                }, 1000);
            }, 2000);
        } else {
            document.getElementById('welcome').style.display = 'none';
        }
    }
</script>

</head>
<body>
    <div id="welcome">Welcome to the University of East London Chatbot
    </div>

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
            <strong>Bot:</strong> {{ response }}
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

    <div class="footer">
        <a href="https://docs.google.com/forms/d/e/1FAIpQLSekBu2bbRAbi7JuCEwZ3d9ggSLfxXJk0BmiC8CC0JTdCy9hIA/viewform?usp=header" target="_blank" class="btn btn-outline-primary btn-sm">💬 Leave a Review</a>
        <a href="#" onclick="toggleDesignerInfo()" class="btn btn-outline-dark btn-sm">🧑‍💻 Contact Designer</a>
        <div id="designer-info" class="mt-3" style="display:none;">
            <div class="alert alert-secondary">
                <strong>Designer:</strong> C P Budha Magar<br>
                <strong>Email:</strong> <a href="mailto:budhacpmagar2@gmail.com">budhacpmagar2@gmail.com</a><br>
                <strong>University:</strong> University of East London<br>
                <strong>Role:</strong> Artificial Intelligence and Machine Learning Developer
            </div>
        </div>
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
