from flask import Flask, render_template, request, Response, jsonify, redirect, url_for, session
from openai import OpenAI
import os, PyPDF2, pytesseract, platform, json, datetime
from PIL import Image
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import sqlite3
import random

# --- Flask setup ---
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Database setup ---
DB_FILE = "users.db"
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
init_db()

# --- Flask-Login User class ---
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

def get_user_by_username(username):
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, username, password FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        return User(*row) if row else None

def get_user_by_id(user_id):
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, username, password FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()
        return User(*row) if row else None

def add_user(username, password):
    hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))

@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(user_id)

# --- Study tips ---
TIPS = [
    "Break big tasks into smaller chunks for better focus.",
    "Review your notes after class — not just before tests.",
    "Use the Pomodoro technique: 25 minutes of focus, 5 minutes of rest.",
    "Teach what you learn — it helps you understand it deeper.",
    "Stay hydrated and take short walks during long study sessions."
]

# --- Per-user history ---
HISTORY_DIR = "user_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

def user_history_file(user_id):
    return os.path.join(HISTORY_DIR, f"{user_id}.json")

def load_history(user_id):
    path = user_history_file(user_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_history(user_id, history):
    path = user_history_file(user_id)
    with open(path, "w") as f:
        json.dump(history[-5:], f, indent=2)

def add_to_history(user_id, action, filename):
    history = load_history(user_id)
    entry = {
        "action": action.capitalize(),
        "file": filename,
        "time": datetime.datetime.now().strftime("%I:%M %p")
    }
    history.append(entry)
    save_history(user_id, history)

# --- OCR / text extraction ---
def strip_headers(text):
    lines = text.splitlines()
    filtered = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(k in line.lower() for k in ["name", "date", "classwork", "directions"]):
            continue
        filtered.append(line)
    return " ".join(filtered)

def extract_text(file):
    name = file.filename.lower()
    if name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in reader.pages)
    elif name.endswith((".png", ".jpg", ".jpeg")):
        return strip_headers(pytesseract.image_to_string(Image.open(file)))
    else:
        return file.read().decode("utf-8", errors="ignore")

# --- Auth routes ---
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if get_user_by_username(username):
            return "Username already exists."
        add_user(username, password)
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = get_user_by_username(username)
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("index"))
        return "Invalid credentials."
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# --- Main study routes ---
@app.route("/")
@login_required
def index():
    tip = random.choice(TIPS)
    history = load_history(current_user.id)
    return render_template("index.html", tip_of_the_day=tip, recent_history=history, username=current_user.username)

@app.route("/get_history")
@login_required
def get_history():
    return jsonify(load_history(current_user.id))

@app.route("/stream", methods=["POST"])
@login_required
def stream():
    action = request.form.get("action", "summarize")
    custom_prompt = request.form.get("custom_prompt", "")
    file = request.files.get("text_file")

    text = extract_text(file) if file else ""
    add_to_history(current_user.id, action, file.filename if file else "Custom Input")

    if action == "summarize":
        prompt = f"Summarize this educational text clearly:\n\n{text}"
    elif action == "solve":
        prompt = f"Solve and explain this educational question step-by-step:\n\n{text}"
    elif action == "custom":
        prompt = f"{custom_prompt}\n\nRelevant context:\n{text}"
    else:
        prompt = text

    def generate():
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are StudySpark, a clear and accurate educational assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return Response(generate(), mimetype="text/plain")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
