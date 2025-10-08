from flask import Flask, render_template, request
from openai import OpenAI
import PyPDF2
import os
from io import BytesIO
from PIL import Image
import pytesseract

app = Flask(__name__)

# Limit upload size to 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 1️⃣ Summarizer function ---
def summarize_text(text, max_length=200):
    prompt = f"""
    You are an expert study assistant. Summarize the following text for students preparing for a test. 

    Output format:
    - Main Idea: one-sentence summary of the overall topic.
    - Key Points: 3–7 bullet points with the most important facts, terms, or events.
    - Study Tip: One way to help recall the information that is fun, quick, and easy to apply.

    Text:
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_length
    )
    return response.choices[0].message.content

# --- 2️⃣ New function: solve and explain problems ---
def solve_and_explain_text(text, max_length=1000):
    """Ask AI to solve and explain the uploaded problem."""
    prompt = f"""
    You are a helpful AI that solves academic problems. 
    Solve the problem in the uploaded file **briefly** and explain the key steps in simple, clear language in **no more than 5 sentences**. 
    Avoid unnecessary details, repetition, or complex symbols. 
    Be direct, easy to understand, and concise. 
    At the end, provide the final answer clearly and separately.

    Problem:
    {text}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_length
    )
    return response.choices[0].message.content

# --- 3️⃣ Keyword search function (unchanged) ---
def search_for_keywords(keyword, text):
    keyword = keyword.lower()
    results = []
    sentences = text.split(". ")
    for i, sentence in enumerate(sentences):
        if keyword in sentence.lower():
            results.append((i + 1, sentence.strip()))
    return results

# --- 4️⃣ Split text into chunks (unchanged) ---
def chunk_text_by_words(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

# --- 5️⃣ Main route ---
@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""
    matches = []

    if request.method == "POST":
        file = request.files["text_file"]
        keyword = request.form.get("keyword", "")
        action = request.form.get("action")  # <-- NEW: Detect which button was clicked

        file_name = file.filename
        file_ext = file_name.split(".")[-1].lower()
        all_text = []

        # Handle PDF
        if file_ext == "pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text.append(page_text)

        # Handle TXT
        elif file_ext == "txt":
            file_contents = file.read().decode("utf-8")
            all_text.append(file_contents)

        # Handle Images (JPEG, JPG, PNG)
        elif file_ext in ["jpeg", "jpg", "png"]:
            image = Image.open(file)
            extract_text = pytesseract.image_to_string(image)
            clean_text = " ".join(extract_text.split())
            all_text.append(clean_text)

        # Unsupported file type
        else:
            return "Unsupported file type. Please upload a PDF, TXT file, JPEG file, JPG file, or PNG file."

        # Combine and clean text
        clean_text = " ".join(" ".join(all_text).split())

        # Perform keyword search (only for summarize mode)
        if keyword:
            matches = search_for_keywords(keyword, clean_text)

        # --- 6️⃣ Decide which AI action to perform ---
        if action == "summarize":
            # Summarize text in chunks
            mini_summaries = []
            for chunk in chunk_text_by_words(clean_text):
                mini_summary = summarize_text(chunk, max_length=100)
                mini_summaries.append(mini_summary)

            combined_summary = "\n".join(mini_summaries)
            summary = summarize_text(combined_summary, max_length=200)

        elif action == "solve":
            # Solve & explain the problem instead of summarizing
            summary = solve_and_explain_text(clean_text, max_length=250)

    return render_template("index.html", summary=summary, matches=matches)

# --- 7️⃣ Run app ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
