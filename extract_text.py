from flask import Flask, request, Response, render_template
from openai import OpenAI
import os
import PyPDF2
from PIL import Image
import pytesseract
import cv2
import platform
import re
import datetime

# --- Setup ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

client = OpenAI()

# --- Recent History (metadata only) ---
recent_history = []  # list of dicts with action, file/prompt, timestamp
HISTORY_LIMIT = 5  # keep last 5 entries

def add_to_history(action, file_name_or_prompt):
    entry = {
        "action": action,
        "file": file_name_or_prompt,
        "time": datetime.datetime.now().strftime("%H:%M")
    }
    recent_history.insert(0, entry)  # newest first
    if len(recent_history) > HISTORY_LIMIT:
        recent_history.pop()  # remove oldest

# --- Tip of the Day ---
TIPS = [
    "Break your study sessions into 25-minute focused intervals.",
    "Teach someone else what you just learned to reinforce your knowledge.",
    "Write down questions you have while reading to explore later.",
    "Summarize concepts in your own words to deepen understanding.",
    "Review past mistakes to avoid repeating them."
]

def get_tip_of_the_day():
    index = datetime.datetime.now().timetuple().tm_yday % len(TIPS)
    return TIPS[index]

# --- Strip Headers Helper ---
def strip_headers(text):
    lines = text.splitlines()
    filtered = []
    for line in lines:
        line = line.strip()
        if not line:  # skip empty lines
            continue
        if any(keyword in line.lower() for keyword in ["name", "date", "classwork", "directions"]):
            continue
        filtered.append(line)
    return " ".join(filtered)

# --- Deduplication Helper ---
def deduplicate_text(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    seen = set()
    unique = []
    for s in sentences:
        s_clean = re.sub(r'\s+', ' ', s.strip().lower())
        if s_clean not in seen and len(s_clean) > 3:
            seen.add(s_clean)
            unique.append(s.strip())
    return " ".join(unique)

# --- OCR Cleaning ---
def clean_ocr_text(text):
    replacements = {
        '≤': '<=', '≥': '>=', '×': '*', '÷': '/', '−': '-', '—': '-', '–': '-',
        '∗': '*', '‘': "'", '’': "'", '“': '"', '”': '"'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'[^0-9a-zA-Z\s\+\-\*\/\=\<\>\(\)\.\^\%√∑∫πθΔλμ]', '', text)
    return text.strip()

# --- OCR Fix Spacing ---
def fix_ocr_spacing(text):
    text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Separate numbers and units ---
def separate_numbers_units(text):
    return re.sub(r'(-?\d+)([a-zA-Z])', r'\1 \2', text)

# --- Detect Math Symbols ---
MATH_SYMBOLS = ['+', '-', '=', '*', '/', '^', '√', '%', '∑', '∫', 'π', 'θ', 'Δ', 'λ', 'μ', '<', '>', '≤', '≥']
def contains_math_symbols(text):
    return any(sym in text for sym in MATH_SYMBOLS)

# --- OCR Preprocessing ---
def preprocess_image_for_ocr(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )
    processed_path = os.path.join(app.config["UPLOAD_FOLDER"], "processed_image.png")
    cv2.imwrite(processed_path, thresh)

    ocr_config = "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-=*/^%√∑∫πθΔλμ()<>"
    text_raw = pytesseract.image_to_string(Image.open(processed_path), config=ocr_config)

    text_cleaned = clean_ocr_text(text_raw)
    text_cleaned = fix_ocr_spacing(text_cleaned)
    text_cleaned = separate_numbers_units(text_cleaned)

    return text_cleaned

# --- GPT Helpers ---
def stream_gpt_response(prompt, max_length=500):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_length,
        stream=True
    )
    for chunk in response:
        content = getattr(chunk.choices[0].delta, "content", "")
        if content:
            yield content

# --- Instruction ---
EFFICIENCY_INSTRUCTION = (
    "If this contains math or scientific notation, interpret and solve it carefully. "
    "If it’s short text, focus on accuracy and reasoning. "
    "Always be solid and educational — explain *why*, not just what. "
    "Use Markdown math (`x^2 + y^2 = z^2`) and clear formatting."
)

@app.route("/stream", methods=["POST"])
def stream_text():
    file = request.files.get("text_file")
    user_question = request.form.get("custom_prompt", "").strip()
    action = request.form.get("action", "summarize").lower()

    # --- Add to recent history ---
    history_name = file.filename if file else (user_question if user_question else "No Name")
    add_to_history(action.capitalize(), history_name)

    if not file:
        return "No file uploaded", 400

    file_ext = file.filename.split(".")[-1].lower()
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    def generate():
        # Read file content
        if file_ext in ["jpeg", "jpg", "png"]:
            raw_text = preprocess_image_for_ocr(file_path)
            pages = [raw_text]
        elif file_ext == "pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            pages = [page.extract_text() or "" for page in pdf_reader.pages]
        elif file_ext == "txt":
            raw_text = file.read().decode("utf-8")
            text_cleaned = clean_ocr_text(raw_text)
            text_cleaned = fix_ocr_spacing(text_cleaned)
            text_cleaned = separate_numbers_units(text_cleaned)
            pages = [text_cleaned]
        else:
            yield "Unsupported file type."
            return

        # Combine pages and deduplicate
        full_text = " ".join(deduplicate_text(p) for p in pages)

        # Debug print
        print("===FULL_TEXT===")
        print(full_text)
        print("===============")

        # Build GPT prompt based on action
        if action == "summarize":
            prompt = f"{EFFICIENCY_INSTRUCTION}\nSummarize clearly:\n{full_text}"
        elif action == "solve":
            prompt = f"{EFFICIENCY_INSTRUCTION}\nSolve step-by-step:\n{full_text}"
        elif action == "custom":
            if not user_question:
                yield "⚠️ No custom prompt provided.\n"
                return
            prompt = f"{EFFICIENCY_INSTRUCTION}\nQuestion: {user_question}\nBased on:\n{full_text}"
        else:
            prompt = f"{EFFICIENCY_INSTRUCTION}\nSummarize:\n{full_text}"

        # Stream GPT response
        for part in stream_gpt_response(prompt):
            if part.strip():
                yield part
        yield "\n\n"

    return Response(generate(), content_type='text/html; charset=utf-8')

@app.route("/", methods=["GET"])
def home():
    tip = get_tip_of_the_day()
    return render_template("index.html", recent_history=recent_history, tip_of_the_day=tip)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
