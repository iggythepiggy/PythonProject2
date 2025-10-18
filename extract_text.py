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
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

client = OpenAI()

# --- OCR Cleaning ---
def clean_ocr_text(text):
    """Fix common OCR symbol errors and normalize spacing."""
    replacements = {
        '≤': '<=',
        '≥': '>=',
        '×': '*',
        '÷': '/',
        '−': '-',
        '—': '-',
        '–': '-',
        '∗': '*',
        ' ‘ ': "'",
        ' ’ ': "'",
        '“': '"',
        '”': '"'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove stray characters like extra dots or lines
    text = re.sub(r'[^0-9a-zA-Z\s\+\-\*\/\=\<\>\(\)]', '', text)
    return text.strip()

# --- OCR Preprocessing ---
def preprocess_image_for_ocr(image_path, scale_factor=1.0):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    resized = cv2.resize(gray, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_CUBIC)

    thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.medianBlur(thresh, 3)

    processed_path = os.path.join(app.config["UPLOAD_FOLDER"], "processed_image.png")
    cv2.imwrite(processed_path, denoised)

    # Default PSM 6
    text = pytesseract.image_to_string(Image.open(processed_path), config="--psm 6")
    text = clean_ocr_text(" ".join(text.split()))

    # If text is very short and contains math, try PSM 7 and 8
    if len(text.split()) < 10 and contains_math_symbols(text):
        # PSM 7: single line
        text = pytesseract.image_to_string(Image.open(processed_path), config="--psm 7")
        text = clean_ocr_text(text)
        # PSM 8: single word/formula
        if len(text.split()) < 10 and contains_math_symbols(text):
            text = pytesseract.image_to_string(Image.open(processed_path), config="--psm 8")
            text = clean_ocr_text(text)

    return text

# --- Chunk Text for GPT ---
def chunk_text_for_streaming(text, chunk_size=150):
    words = text.split()
    if len(words) < 10:
        yield " ".join(words)
    else:
        for i in range(0, len(words), chunk_size):
            yield " ".join(words[i:i + chunk_size])

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

# --- Detect GPT uncertainty ---
def gpt_flags_uncertainty(text):
    patterns = [
        r"If more context were provided",
        r"It seems.*missing information",
        r"Based on what’s provided",
        r"Assuming the given data"
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)

# --- Detect math symbols ---
MATH_SYMBOLS = ["/", "+", "-", "*", "^", "%", "=", "<", ">", "(", ")"]
def contains_math_symbols(text):
    return any(symbol in text for symbol in MATH_SYMBOLS)

# --- Prompt Enhancement ---
PROMPT_KEYWORDS = [
    "solve", "calculate", "create", "explain", "analyze", "compare",
    "design", "generate", "summarize", "convert", "evaluate", "predict"
]
def enhance_prompt(user_prompt):
    if any(keyword in user_prompt.lower() for keyword in PROMPT_KEYWORDS):
        enhancement_instructions = (
            "You are an expert assistant. Refine this prompt so it is as clear, precise, "
            "and actionable as possible, keeping the user's original intent intact. "
            "Return the improved version only."
        )
        refine_prompt = f"{enhancement_instructions}\nUser prompt:\n{user_prompt}"
        refined_prompt = gpt_response(refine_prompt, max_length=200)
        return refined_prompt
    return user_prompt

# --- Efficiency Instruction ---
EFFICIENCY_INSTRUCTION = (
    "Answer in the simplest, most concise, and efficient way possible, "
    "while still being accurate and educational."
)

# --- Main streaming route ---
@app.route("/stream", methods=["POST"])
def stream_text():
    file = request.files.get("text_file")
    if not file:
        return "No file uploaded", 400

    file_ext = file.filename.split(".")[-1].lower()
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    action = request.form.get("action", "summarize").lower()
    custom_prompt = request.form.get("custom_prompt", "").strip()

    def generate():
        # --- IMAGE FILES ---
        if file_ext in ["jpeg", "jpg", "png"]:
            scale_factor = 1.0
            for attempt in range(2):
                ocr_text = preprocess_image_for_ocr(file_path, scale_factor=scale_factor)
                if not ocr_text:
                    yield "⚠️ OCR failed to detect any text.\n"
                    break

                # Handle short math text separately
                if len(ocr_text.split()) < 10 and contains_math_symbols(ocr_text):
                    prompt = f"{EFFICIENCY_INSTRUCTION}\nSolve this math problem step-by-step but briefly:\n{ocr_text}"
                    yield from stream_gpt_response(prompt)
                    return

                test_prompt = f"You are an expert study assistant. Solve or summarize this content concisely:\n{ocr_text}"
                gpt_test = gpt_response(test_prompt)
                if gpt_flags_uncertainty(gpt_test):
                    scale_factor *= 1.5
                    yield "⚡ OCR uncertain or text too small, retrying with higher resolution...\n"
                    continue

                for chunk in chunk_text_for_streaming(ocr_text):
                    math_present = contains_math_symbols(chunk)
                    math_instruction = (
                        "Use proper mathematical symbols in plain text (e.g., 5/6 instead of LaTeX \\frac{5}{6})."
                        if math_present else ""
                    )

                    if action == "summarize":
                        prompt = f"{EFFICIENCY_INSTRUCTION}\nSummarize this text clearly and concisely. {math_instruction}\n{chunk}"
                    elif action == "custom":
                        if not custom_prompt:
                            yield "⚠️ No custom prompt provided.\n"
                            return
                        enhanced_custom_prompt = enhance_prompt(custom_prompt)
                        prompt = f"{EFFICIENCY_INSTRUCTION}\n{enhanced_custom_prompt}\n\n{math_instruction}\nHere is the text:\n{chunk}"
                    else:
                        prompt = f"{EFFICIENCY_INSTRUCTION}\nSolve the following problem clearly and concisely. {math_instruction}\n{chunk}"

                    yield from stream_gpt_response(prompt)
            return

        # --- PDF and TXT ---
        elif file_ext == "pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = " ".join(page.extract_text() or "" for page in pdf_reader.pages)
        elif file_ext == "txt":
            text = file.read().decode("utf-8")
        else:
            yield "Unsupported file type."
            return

        for chunk in chunk_text_for_streaming(text):
            math_present = contains_math_symbols(chunk)
            math_instruction = "Use proper mathematical symbols in the answer." if math_present else ""
            if len(chunk.split()) < 10 and math_present:
                prompt = f"{EFFICIENCY_INSTRUCTION}\nSolve this math problem step-by-step but briefly:\n{chunk}"
            elif action == "summarize":
                prompt = f"{EFFICIENCY_INSTRUCTION}\nSummarize this text clearly and concisely. {math_instruction}\n{chunk}"
            elif action == "custom":
                if not custom_prompt:
                    yield "⚠️ No custom prompt provided.\n"
                    return
                enhanced_custom_prompt = enhance_prompt(custom_prompt)
                prompt = f"{EFFICIENCY_INSTRUCTION}\n{enhanced_custom_prompt}\n\n{math_instruction}\nHere is the text:\n{chunk}"
            else:
                prompt = f"{EFFICIENCY_INSTRUCTION}\nSolve the following problem clearly and concisely. {math_instruction}\n{chunk}"
            yield from stream_gpt_response(prompt)

    return Response(generate(), content_type='text/plain; charset=utf-8')

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
