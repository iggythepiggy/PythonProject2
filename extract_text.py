from flask import Flask, render_template, request, Response
from openai import OpenAI
import PyPDF2
import os
from PIL import Image
import pytesseract
import cv2

# --- Setup ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Initialize OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 1️⃣ Summarizer function ---
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


# --- 2️⃣ Solve & Explain function ---
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


# --- 3️⃣ Keyword search ---
def search_for_keywords(keyword, text):
    keyword = keyword.lower()
    results = []
    sentences = text.split(". ")
    for i, sentence in enumerate(sentences):
        if keyword in sentence.lower():
            results.append((i + 1, sentence.strip()))
    return results


# --- 4️⃣ Chunking functions ---
def chunk_text_by_words(text, chunk_size=800):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

def chunk_text_for_streaming(text, chunk_size=50):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


# --- 5️⃣ Image Preprocessing for better OCR ---
def preprocess_image_for_ocr(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(thresh, 3)

    # Slight upscaling helps OCR accuracy
    scale_percent = 150
    width = int(denoised.shape[1] * scale_percent / 100)
    height = int(denoised.shape[0] * scale_percent / 100)
    resized = cv2.resize(denoised, (width, height), interpolation=cv2.INTER_LINEAR)

    processed_path = os.path.join(app.config["UPLOAD_FOLDER"], "processed_image.png")
    cv2.imwrite(processed_path, resized)

    text = pytesseract.image_to_string(Image.open(processed_path))
    clean_text = " ".join(text.split())
    return clean_text


# --- 6️⃣ OCR / Summarize Streaming Route ---
@app.route("/stream", methods=["POST"])
def stream_text():
    if "text_file" not in request.files:
        return "No file uploaded", 400

    file = request.files["text_file"]
    file_ext = file.filename.split(".")[-1].lower()

    # Extract text depending on file type
    if file_ext in ["jpeg", "jpg", "png"]:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        text = preprocess_image_for_ocr(file_path)
    elif file_ext == "pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        text = " ".join(page.extract_text() or "" for page in pdf_reader.pages)
    elif file_ext == "txt":
        text = file.read().decode("utf-8")
    else:
        return "Unsupported file type.", 400

    # Decide action
    action = request.form.get("action")
    if action == "summarize":
        result_text = summarize_text(text, max_length=200)
    elif action == "solve":
        result_text = solve_and_explain_text(text, max_length=250)
    else:
        result_text = "No valid action selected."

    # Stream result in chunks
    def generate():
        for chunk in chunk_text_for_streaming(result_text):
            yield chunk + " "

    return Response(generate(), mimetype='text/plain')


# --- 7️⃣ Main Route (Summarize / Solve) ---
@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""
    matches = []

    if request.method == "POST":
        if "text_file" not in request.files:
            return "No file uploaded", 400
        file = request.files["text_file"]
        keyword = request.form.get("keyword", "")
        action = request.form.get("action")

        file_name = file.filename
        file_ext = file_name.split(".")[-1].lower()
        all_text = []

        # PDFs
        if file_ext == "pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text.append(page_text)
        # TXT files
        elif file_ext == "txt":
            all_text.append(file.read().decode("utf-8"))
        # Images
        elif file_ext in ["jpeg", "jpg", "png"]:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            clean_text = preprocess_image_for_ocr(file_path)
            all_text.append(clean_text)
        else:
            return "Unsupported file type.", 400

        clean_text = " ".join(all_text)

        # Keyword search
        if keyword:
            matches = search_for_keywords(keyword, clean_text)

        # Action
        if action == "summarize":
            mini_summaries = [summarize_text(chunk, max_length=100) for chunk in chunk_text_by_words(clean_text)]
            combined_summary = "\n".join(mini_summaries)
            summary = summarize_text(combined_summary, max_length=200)
        elif action == "solve":
            summary = solve_and_explain_text(clean_text, max_length=250)

    return render_template("index.html", summary=summary, matches=matches)


# --- 8️⃣ Run App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
