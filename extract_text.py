from flask import Flask, request, Response, render_template
from openai import OpenAI
import os
import PyPDF2
from PIL import Image
import pytesseract
import cv2
import platform

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

# --- OCR Preprocessing ---
def preprocess_image_for_ocr(image_path, high_quality=False):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    if high_quality:
        # Enlarge small images for clarity
        scale_factor = max(2.0, min(1200 / width, 1200 / height))
    else:
        scale_factor = min(1200 / width, 1200 / height, 1.0)

    resized = cv2.resize(gray, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_CUBIC)

    # Only apply threshold/denoise if not high quality
    if not high_quality:
        thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.medianBlur(thresh, 3)
    else:
        denoised = resized  # keep original clarity

    processed_path = os.path.join(app.config["UPLOAD_FOLDER"], "processed_image.png")
    cv2.imwrite(processed_path, denoised)
    text = pytesseract.image_to_string(Image.open(processed_path), config="--psm 6")
    return " ".join(text.split())

# --- Chunk Text for GPT ---
def chunk_text_for_streaming(text, chunk_size=150):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

# --- GPT Streaming Helpers ---
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

# --- Routes ---
@app.route("/stream", methods=["POST"])
def stream_text():
    if "text_file" not in request.files:
        return "No file uploaded", 400

    file = request.files["text_file"]
    file_ext = file.filename.split(".")[-1].lower()
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    action = request.form.get("action", "summarize")

    def generate():
        # --- IMAGE FILES (OCR only) ---
        if file_ext in ["jpeg", "jpg", "png"]:
            initial_text = preprocess_image_for_ocr(file_path)
            high_quality = len(initial_text.split()) < 10
            ocr_text = preprocess_image_for_ocr(file_path, high_quality=high_quality)

            for chunk in chunk_text_for_streaming(ocr_text):
                if action == "summarize":
                    prompt = (
                        "You are an expert study assistant. "
                        "Summarize the following text clearly for a student. "
                        "Use plain English and never include LaTeX, brackets, or special symbols like \\[ or \\text{.} "
                        f"\n\nText:\n{chunk}"
                    )
                else:
                    prompt = (
                        "You are a helpful AI that solves academic problems clearly and neatly. "
                        "Write the math and answers in plain text (no LaTeX, backslashes, or brackets). "
                        "Keep your explanation short, clean, and formatted neatly for easy reading.\n\n"
                        f"Problem:\n{chunk}"
                    )
                yield from stream_gpt_response(prompt)

        # --- PDF FILES ---
        elif file_ext == "pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = " ".join(page.extract_text() or "" for page in pdf_reader.pages)
            for chunk in chunk_text_for_streaming(text):
                if action == "summarize":
                    prompt = (
                        "Summarize this text clearly in plain English. "
                        "Do not include LaTeX, math code, or special formatting.\n\n"
                        f"{chunk}"
                    )
                else:
                    prompt = (
                        "Solve this problem in plain text, using clear step-by-step reasoning. "
                        "Do not use LaTeX or special math symbols.\n\n"
                        f"{chunk}"
                    )
                yield from stream_gpt_response(prompt)

        # --- TEXT FILES ---
        elif file_ext == "txt":
            text = file.read().decode("utf-8")
            for chunk in chunk_text_for_streaming(text):
                if action == "summarize":
                    prompt = f"Summarize this clearly in plain text:\n\n{chunk}"
                else:
                    prompt = f"Solve this problem in plain text and explain briefly:\n\n{chunk}"
                yield from stream_gpt_response(prompt)

        else:
            yield "Unsupported file type."

    return Response(generate(), content_type='text/plain; charset=utf-8')

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
