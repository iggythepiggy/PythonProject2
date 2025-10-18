from flask import Flask, request, Response, render_template
from openai import OpenAI
import os
import PyPDF2
from PIL import Image
import pytesseract
import cv2
import platform
import re

# --- Setup ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

client = OpenAI()

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
        '≤': '<=', '≥': '>=', '×': '*', '÷': '/', '−': '-',
        '—': '-', '–': '-', '∗': '*', ' ‘ ': "'", ' ’ ': "'",
        '“': '"', '”': '"'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'[^0-9a-zA-Z\s\+\-\*\/\=\<\>\(\)]', '', text)
    return text.strip()

# --- OCR Preprocessing ---
def preprocess_image_for_ocr(image_path, scale_factor=1.0):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    resized = cv2.resize(gray, (int(width*scale_factor), int(height*scale_factor)), interpolation=cv2.INTER_CUBIC)
    thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.medianBlur(thresh, 3)
    processed_path = os.path.join(app.config["UPLOAD_FOLDER"], "processed_image.png")
    cv2.imwrite(processed_path, denoised)

    text = pytesseract.image_to_string(Image.open(processed_path), config="--psm 6")
    text = clean_ocr_text(" ".join(text.split()))
    return text

# --- Chunk Text ---
def chunk_text_for_streaming(text, chunk_size=150):
    words = text.split()
    if len(words) < 10:
        yield " ".join(words)
    else:
        for i in range(0, len(words), chunk_size):
            yield " ".join(words[i:i+chunk_size])

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

def gpt_response(prompt, max_length=500):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_length
    )
    return resp.choices[0].message.content

# --- Math symbols ---
MATH_SYMBOLS = ["/","+","-","*","^","%","=","<",">","(",")"]
def contains_math_symbols(text):
    return any(symbol in text for symbol in MATH_SYMBOLS)

# --- Efficiency Instruction ---
EFFICIENCY_INSTRUCTION = (
    "Summarize the text in the simplest, clearest, and most concise way possible. "
    "Keep sentences short, avoid repetition, and use plain language. "
    "Format your answer in clear paragraphs."
)

# --- Solve Keywords ---
SOLVE_KEYWORDS = ["solve","calculate","compute","evaluate","find","derivative","integral","answer"]


@app.route("/stream", methods=["POST"])
def stream_text():
    file = request.files.get("text_file")
    if not file:
        return "No file uploaded", 400

    file_ext = file.filename.split(".")[-1].lower()
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Get user question instead of action
    user_question = request.form.get("custom_prompt", "").strip()
    if not user_question:
        return "⚠️ Please provide a question in 'custom_prompt'.", 400

    def generate():
        # --- Extract text ---
        if file_ext in ["jpeg", "jpg", "png"]:
            raw_text = preprocess_image_for_ocr(file_path)
        elif file_ext == "pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            raw_text = " ".join(page.extract_text() or "" for page in pdf_reader.pages)
        elif file_ext == "txt":
            raw_text = file.read().decode("utf-8")
        else:
            yield "Unsupported file type."
            return

        text = deduplicate_text(raw_text)

        # --- Split text into chunks ---
        chunk_summaries = []

        for chunk in chunk_text_for_streaming(text):
            # For each chunk, ask for the answer if it contains relevant info
            prompt = f"{EFFICIENCY_INSTRUCTION}\nAnswer this question based only on the following text, in simple, clear language:\nQuestion: {user_question}\nText:\n{chunk}"
            summary_chunk = gpt_response(prompt)
            chunk_summaries.append(summary_chunk.strip())

        # --- Combine all chunk-level answers into one final answer ---
        combined_text = " ".join(chunk_summaries)
        final_prompt = f"{EFFICIENCY_INSTRUCTION}\nCombine the following partial answers into a single concise answer to the question '{user_question}':\n{combined_text}"

        for part in stream_gpt_response(final_prompt):
            if part.strip():
                yield part

    return Response(generate(), content_type='text/plain; charset=utf-8')



@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
