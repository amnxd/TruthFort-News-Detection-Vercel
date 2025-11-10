from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import nltk
import sqlite3
import hashlib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Initialize Flask
app = Flask(__name__)
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
        if not NEWS_API_KEY:
            return ["Sample news article 1: This is a test article.", 
                   "Sample news article 2: Another test article for verification."]
        
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
            return {
                'verification': 'Uncertain', 
                'confidence': 50.0,
                'statement': statement,
                'reason': 'Limited data available for verification. Please try with a more specific statement.',
                'sources': ['Data source temporarily unavailable']
            }
        
        all_texts = [statement] + articles
        try:
            tfidf = self.vectorizer.fit_transform(all_texts)
            sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
            avg_sim, max_sim = sims.mean(), sims.max()
            
            if max_sim > 0.4:
                verdict, conf = 'Likely True', min(max_sim * 100, 95)
            elif max_sim > 0.2:
                verdict, conf = 'Uncertain', min(max_sim * 80, 75)
            else:
                verdict, conf = 'Likely False', min((1 - avg_sim) * 60, 70)
                
            return {
                'verification': verdict, 
                'confidence': round(conf, 2),
                'statement': statement,
                'reason': f'Analysis of {len(articles)} articles shows {verdict.lower()} correlation.',
                'sources': articles[:5],
                'articles_analyzed': len(articles)
            }
        except Exception as e:
            return {
                'verification': 'Error', 
                'confidence': 0,
                'statement': statement,
                'reason': f'Analysis error: {str(e)}',
                'sources': []
            }

# Initialize DB and verifier
init_db()
verifier = NewsVerifier()

# Routes
@app.route('/')
def home():
    return send_from_directory('.', 'home.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
        
    claim = data.get('claim', '').strip()
    if not claim:
        return jsonify({'error': 'No claim provided'}), 400
        
    try:
        result = verifier.verify_statement(claim)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Verification failed: {str(e)}'}), 500

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'No data provided'}), 400
        
    if create_user(data.get('name', ''), data.get('email', ''), data.get('password', '')):
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': 'Email already exists'})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'No data provided'}), 400
        
    user = get_user(data.get('email', ''))
    if not user:
        return jsonify({'success': False, 'message': 'User not found'})
        
    if hash_password(data.get('password', '')) != user[3]:
        return jsonify({'success': False, 'message': 'Invalid password'})
        
    return jsonify({
        'success': True, 
        'user': {
            'email': user[2], 
            'subscription': user[4],
            'name': user[1]
        }
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'Server is running'})

# Vercel requires this
app = app
