from flask import Flask, render_template, request, Response, redirect, url_for, session, jsonify, flash
from openai import OpenAI
import os, PyPDF2, pytesseract, platform, json, datetime, re
from PIL import Image
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import sqlite3, random

# --- Flask setup ---
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# NOTE: replace with your OpenAI usage or mock if not using in dev
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# --- Database setup ---
DB_FILE = "users.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
init_db()

# --- Feedback JSON ---
FEEDBACK_FILE = "feedback.json"

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return []

def save_feedback(feedback_list):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_list, f, indent=2)

# --- Flask-Login User class ---
class User(UserMixin):
    def __init__(self, id, username, email, password):
        self.id = id
        self.username = username
        self.email = email
        self.password = password

# --- Database helpers ---
def get_user_by_username(username):
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, username, email, password FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        return User(*row) if row else None

def get_user_by_email(email):
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, username, email, password FROM users WHERE email=?", (email,))
        row = cur.fetchone()
        return User(*row) if row else None

def get_user_by_id(user_id):
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, username, email, password FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()
        return User(*row) if row else None

def add_user(username, email, password):
    hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, hashed_pw))
        return True
    except sqlite3.IntegrityError:
        return False  # Username or email already exists

@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(user_id)

# --- Setup for Tesseract ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# --- Tip of the day ---
TIPS = [
    "Break big tasks into smaller chunks for better focus.",
    "Review your notes after class — not just before tests.",
    "Use the Pomodoro technique: 25 minutes of focus, 5 minutes of rest.",
    "Teach what you learn — it helps you understand it deeper.",
    "Stay hydrated and take short walks during long study sessions."
]

# --- User history (local JSON storage) ---
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

# --- Helper to strip headers ---
def strip_headers(text):
    lines = text.splitlines()
    filtered = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(keyword in line.lower() for keyword in ["name", "date", "classwork", "directions"]):
            continue
        filtered.append(line)
    return " ".join(filtered)

# --- OCR / Text extraction ---
def extract_text(file):
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        full = []
        for page in reader.pages:
            text = page.extract_text() or ""
            full.append(text)
        return " ".join(full)
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(file)
        text = pytesseract.image_to_string(img)
        return strip_headers(text)
    else:
        try:
            return file.read().decode("utf-8", errors="ignore")
        except:
            return ""

# --- Auth routes ---
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password or not confirm_password:
            flash("Please fill all fields.", "danger")
            return redirect(url_for("signup"))
        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("signup"))
        if len(password) < 8:
            flash("Password must be at least 8 characters.", "danger")
            return redirect(url_for("signup"))
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Invalid email address.", "danger")
            return redirect(url_for("signup"))

        success = add_user(username, email, password)
        if not success:
            flash("Username or email already exists.", "danger")
            return redirect(url_for("signup"))

        flash("Account created successfully! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username_or_email = request.form.get("username_or_email", "").strip()
        password = request.form.get("password", "")
        user = get_user_by_username(username_or_email) or get_user_by_email(username_or_email)

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash("Logged in successfully.", "success")
            return redirect(url_for("index"))

        flash("Invalid credentials.", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("login"))

# --- Main study page ---
@app.route("/")
@login_required
def index():
    tip = random.choice(TIPS)
    history = load_history(current_user.id)
    return render_template("index.html", tip_of_the_day=tip, recent_history=history, username=current_user.username)

@app.route("/get_history")
@login_required
def get_history_route():
    return jsonify(load_history(current_user.id))

# --- Stream educational AI responses ---
# --- Stream educational AI responses ---
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

    # --- Initialize AI output accumulator ---
    ai_output_chunks = []
    user_id = current_user.id  # capture user id safely BEFORE the generator

    def generate():
        try:
            stream_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are StudySpark, a clear and accurate educational assistant."},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            for chunk in stream_resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    ai_output_chunks.append(delta)
                    yield delta
        except Exception as e:
            # fallback if OpenAI API is not available
            fallback_text = "StudySpark (mock): Sorry, OpenAI not available. Here's a mock summary.\n\n" + (prompt[:1000] if prompt else "No text provided.")
            ai_output_chunks.append(fallback_text)
            yield fallback_text
        finally:
            # --- Join all chunks into a single string and save to feedback.json ---
            ai_output_str = "".join(ai_output_chunks)
            feedback_list = load_feedback()
            feedback_list.append({
                "user_id": user_id,
                "response_text": "",  # keep blank or store frontend response if needed
                "feedback": "AI Generated",
                "ai_output": ai_output_str,
                "timestamp": datetime.datetime.now().isoformat()
            })
            save_feedback(feedback_list)

    return Response(generate(), mimetype="text/plain")


# --- Feedback route (accepts JSON from the frontend) ---
@app.route("/feedback", methods=["POST"])
@login_required
def feedback():
    data = None
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()
        # --- Skip saving entire entry if feedback is "Good" ---
    if data.get("feedback", "").strip().lower() == "good":
        print("Skipped saving entire 'Good' feedback entry.")
        return jsonify({"message": "Skipped 'Good' feedback entry"}), 200

    choice = data.get("feedback") or data.get("feedback_type") or "Unknown"
    response_text = data.get("response_text") or data.get("extra") or ""
    ai_output = data.get("ai_output") or ""  # NEW: include AI output if available

    feedback_list = load_feedback()
    feedback_list.append({
        "user_id": current_user.id,
        "response_text": response_text,
        "feedback": choice,
        "ai_output": ai_output,  # NEW field
        "timestamp": datetime.datetime.now().isoformat()
    })
    save_feedback(feedback_list)
    return jsonify({"status": "ok"})


# --- Run server ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
