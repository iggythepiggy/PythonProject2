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

# --- Optimized OCR preprocessing (new) ---
def preprocess_image_for_ocr(image_path):
    """
    Preprocess image for OCR:
    - Convert to grayscale
    - Resize to a reasonable resolution (reduce file size)
    - Apply thresholding & denoising for better OCR accuracy
    """
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reduce resolution while keeping text readable
    max_width = 1200
    max_height = 1200
    height, width = gray.shape
    scale_factor = min(max_width / width, max_height / height, 1.0)  # only scale down
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Apply adaptive thresholding for better OCR recognition
    thresh = cv2.adaptiveThreshold(
        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Apply median blur to reduce noise
    denoised = cv2.medianBlur(thresh, 3)

    # Save temporarily and extract text
    processed_path = os.path.join(app.config["UPLOAD_FOLDER"], "processed_image.png")
    cv2.imwrite(processed_path, denoised)
    text = pytesseract.image_to_string(Image.open(processed_path))
    return " ".join(text.split())

# --- GPT helpers ---
def summarize_text(text, max_length=200):
    prompt = f"""
    You are an expert study assistant. Summarize the following text for students preparing for a test.

    Output format:
    - Main Idea: one-sentence summary of the overall topic.
    - Key Points: 3–7 bullet points with the most important facts, terms, or events.
    - Study Tip: Highlight the most important part of the summary and how you can practice and understand it better.

    Text:
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_length
    )
    return response.choices[0].message.content

def solve_and_explain_text(text, max_length=1000):
    prompt = f"""
    You are a helpful AI that solves academic problems like math and science.
    Solve the problem briefly and clearly in no more than 5 sentences.
    Avoid unnecessary details or repetition.
    End with the final answer clearly separated.

    Problem:
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_length
    )
    return response.choices[0].message.content

# --- GPT image helpers (OCR <10 words) ---
def summarize_image(image_path, max_length=200):
    prompt = """
    You are an expert study assistant. Summarize the content in this image for students preparing for a test.

    Output format:
    - Main Idea: one-sentence summary of the overall topic.
    - Key Points: 3–7 bullet points with the most important facts, terms, or events.
    - Study Tip: Highlight the most important part of the summary and how you can practice and understand it better.
    """
    # NOTE: Using GPT-4O Mini here; adjust image handling if using a future vision-enabled model
    with open(image_path, "rb") as img_file:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_length
        )
    return response.choices[0].message.content

def solve_and_explain_image(image_path, max_length=1000):
    prompt = """
    You are a helpful AI that solves academic problems like math and science, Social Studies, and ELA.
    Solve the problems in this image clearly and concisely in no more than 5 sentences.
    Avoid unnecessary details or repetition.
    End with the final answer clearly separated.
    """
    with open(image_path, "rb") as img_file:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_length
        )
    return response.choices[0].message.content

# --- Smarter chunking for streaming (changed) ---
def chunk_text_for_streaming(text, chunk_size=150):
    """
    Yield text in chunks of ~150 words for faster streaming.
    Can be adjusted to larger/smaller sizes based on responsiveness.
    """
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

# --- Flask route for streaming with partial + final summary (new) ---
@app.route("/stream", methods=["POST"])
def stream_text():
    if "text_file" not in request.files:
        return "No file uploaded", 400
    file = request.files["text_file"]
    file_ext = file.filename.split(".")[-1].lower()
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    action = request.form.get("action")
    result_text = ""
    partial_texts = []

    # --- Decide what to send to GPT ---
    if file_ext in ["jpeg", "jpg", "png"]:
        ocr_text = preprocess_image_for_ocr(file_path)
        if len(ocr_text.split()) < 10:
            # Diagram or short text → send image directly
            if action == "summarize":
                result_text = summarize_image(file_path)
            elif action == "solve":
                result_text = solve_and_explain_image(file_path)
        else:
            # Lots of text → send OCR text
            # Partial streaming: split OCR text into chunks
            for chunk in chunk_text_for_streaming(ocr_text):
                if action == "summarize":
                    partial_texts.append(summarize_text(chunk, max_length=100))
                elif action == "solve":
                    partial_texts.append(solve_and_explain_text(chunk, max_length=150))
            # Combine all chunks and create a final accurate summary/solution
            combined_text = " ".join(partial_texts)
            if action == "summarize":
                result_text = summarize_text(combined_text, max_length=200)
            elif action == "solve":
                result_text = solve_and_explain_text(combined_text, max_length=250)

    elif file_ext == "pdf":
        pdf_reader = PyPDF2.PdfReader(file_path)
        text = " ".join(page.extract_text() or "" for page in pdf_reader.pages)
        for chunk in chunk_text_for_streaming(text):
            if action == "summarize":
                partial_texts.append(summarize_text(chunk, max_length=100))
            elif action == "solve":
                partial_texts.append(solve_and_explain_text(chunk, max_length=150))
        combined_text = " ".join(partial_texts)
        if action == "summarize":
            result_text = summarize_text(combined_text, max_length=200)
        elif action == "solve":
            result_text = solve_and_explain_text(combined_text, max_length=250)

    elif file_ext == "txt":
        text = file.read().decode("utf-8")
        for chunk in chunk_text_for_streaming(text):
            if action == "summarize":
                partial_texts.append(summarize_text(chunk, max_length=100))
            elif action == "solve":
                partial_texts.append(solve_and_explain_text(chunk, max_length=150))
        combined_text = " ".join(partial_texts)
        if action == "summarize":
            result_text = summarize_text(combined_text, max_length=200)
        elif action == "solve":
            result_text = solve_and_explain_text(combined_text, max_length=250)
    else:
        return "Unsupported file type.", 400

    # --- Streaming generator ---
    def generate():
        for chunk in partial_texts:  # stream partial summaries first
            yield chunk + " "
        yield result_text  # finally override with full final summary

    return Response(generate(), mimetype='text/plain')

# --- Main route ---
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
