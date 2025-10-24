from flask import Flask, render_template, request, Response
from openai import OpenAI
import os, PyPDF2, pytesseract, cv2, platform, json, datetime
from PIL import Image
from flask import jsonify

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Setup for Tesseract ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Tip of the day list ---
TIPS = [
    "Break big tasks into smaller chunks for better focus.",
    "Review your notes after class — not just before tests.",
    "Use the Pomodoro technique: 25 minutes of focus, 5 minutes of rest.",
    "Teach what you learn — it helps you understand it deeper.",
    "Stay hydrated and take short walks during long study sessions."
]

# --- History storage ---
HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-5:], f, indent=2)  # keep last 5 only

def add_to_history(action, filename):
    history = load_history()
    entry = {
        "action": action.capitalize(),
        "file": filename,
        "time": datetime.datetime.now().strftime("%I:%M %p")
    }
    history.append(entry)
    save_history(history)

# --- Helper: strip headers like name/date ---
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

# --- OCR / text extraction ---
def extract_text(file):
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in reader.pages)
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(file)
        text = pytesseract.image_to_string(img)
        return strip_headers(text)
    else:
        return file.read().decode("utf-8", errors="ignore")

# --- Streaming route ---
@app.route("/stream", methods=["POST"])
def stream():
    action = request.form.get("action", "summarize")
    custom_prompt = request.form.get("custom_prompt", "")
    file = request.files.get("text_file")

    full_text = extract_text(file) if file else ""
    add_to_history(action, file.filename if file else "Custom Input")

    # Define prompt based on action
    if action == "summarize":
        prompt = f"Summarize this educational text clearly:\n\n{full_text}"
    elif action == "solve":
        prompt = f"Solve and explain this educational question step-by-step:\n\n{full_text}"
    elif action == "custom":
        prompt = f"{custom_prompt}\n\nRelevant context:\n{full_text}"
    else:
        prompt = full_text

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

# --- Main page ---
@app.route("/")
def index():
    import random
    tip = random.choice(TIPS)
    history = load_history()
    return render_template("index.html", tip_of_the_day=tip, recent_history=history)
@app.route("/get_history")
def get_history():
    history = load_history()
    return jsonify(history)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)