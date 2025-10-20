from flask import Flask, render_template, request, Response
from openai import OpenAI
import os
import PyPDF2
from PIL import Image
import pytesseract
import cv2
import platform
import re
from datetime import datetime

# --- Setup ---
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure tesseract path for Windows
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Global in-memory history ---
recent_history = []


# --- Helper: Strip headers like name/date/directions ---
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


# --- Helper: Extract text from PDFs and images ---
def extract_text_from_file(file):
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return strip_headers(text)

    elif filename.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
        return strip_headers(text)

    elif filename.endswith(".txt"):
        return strip_headers(file.read().decode("utf-8"))

    else:
        return "Unsupported file type."


# --- Main page ---
@app.route("/")
def index():
    tip_of_the_day = "💡 Tip: Break your study tasks into smaller chunks — focus on one topic at a time!"
    return render_template("index.html", tip_of_the_day=tip_of_the_day, recent_history=recent_history)


# --- Streaming route ---
@app.route("/stream", methods=["POST"])
def stream():
    action = request.form.get("action")
    custom_prompt = request.form.get("custom_prompt", "")
    uploaded_file = request.files.get("text_file")

    text = ""
    if uploaded_file:
        text = extract_text_from_file(uploaded_file)

    # Add to recent history
    timestamp = datetime.now().strftime("%I:%M %p")
    filename = uploaded_file.filename if uploaded_file else "No file"
    recent_history.append({
        "action": action.title() if action else "Custom",
        "file": filename,
        "time": timestamp
    })
    # Keep only last 5
    if len(recent_history) > 5:
        recent_history.pop(0)

    # Create prompt
    if action == "summarize":
        prompt = f"Summarize the following text clearly and educationally:\n\n{text}"
    elif action == "solve":
        prompt = f"Solve and explain this problem step-by-step in a way a student can learn:\n\n{text}"
    elif action == "custom":
        prompt = f"{custom_prompt}\n\n{text}"
    else:
        prompt = text

    # Stream AI response
    def generate():
        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"\n❌ Error: {str(e)}"

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True)
