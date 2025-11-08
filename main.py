
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import nltk
import sqlite3
import hashlib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from env import NEWS_API_KEY
import re
import os

# Initialize Flask
app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
CORS(app)

# Ensure NLTK data
for pkg in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{pkg}') if pkg == 'punkt' else nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

# Database initialization
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        subscription TEXT DEFAULT 'Free',
        usage_count INTEGER DEFAULT 5,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_reset DATE DEFAULT CURRENT_DATE
    )''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_user(email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email=?', (email,))
    user = c.fetchone()
    conn.close()
    return user

def create_user(name, email, password):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)',
                  (name, email, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


class NewsVerifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def clean_text(self, text):
        return re.sub(r'[^\w\s]', '', text.lower())

    def get_news_articles(self, query):
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                return [
                    f"{a.get('title','')} {a.get('description','')}".strip()
                    for a in data.get('articles', [])
                ]
        except Exception as e:
            print(e)
        return []

    def verify_statement(self, statement):
        articles = self.get_news_articles(statement)
        if not articles:
            return {'verification': 'Inconclusive', 'confidence': 0}
        all_texts = [statement] + articles
        tfidf = self.vectorizer.fit_transform(all_texts)
        sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        avg_sim, max_sim = sims.mean(), sims.max()
        if max_sim > 0.4:
            verdict, conf = 'Likely True', max_sim * 100
        elif max_sim > 0.2:
            verdict, conf = 'Uncertain', max_sim * 80
        else:
            verdict, conf = 'Likely False', (1 - avg_sim) * 40
        return {'verification': verdict, 'confidence': round(conf, 2)}

# Initialize DB and verifier
init_db()
verifier = NewsVerifier()

# Routes
@app.route('/')
def home():
    return send_from_directory('../frontend', 'home.html')

@app.route('/index.html')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    claim = data.get('claim', '').strip()
    if not claim:
        return jsonify({'error': 'No claim provided'}), 400
    return jsonify(verifier.verify_statement(claim))

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if create_user(data['name'], data['email'], data['password']):
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': 'Email exists'})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = get_user(data['email'])
    if not user:
        return jsonify({'success': False})
    if hash_password(data['password']) != user[3]:
        return jsonify({'success': False})
    return jsonify({'success': True, 'user': {'email': user[2], 'subscription': user[4]}})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'Running on Vercel'})

# No app.run() here — Vercel handles that
