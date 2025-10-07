from flask import Flask, render_template, request
from openai import OpenAI
import PyPDF2
import os
from io import BytesIO

app = Flask(__name__)

# Limit upload size to 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Summarizer function
def summarize_text(text, max_length=100):
    prompt = f"""
    You are an expert study assistant. Summarize the following text for students preparing for a test.

    Output format:
    - Main Idea: one-sentence summary of the overall topic.
    - Key Points: 3–7 bullet points with the most important facts, terms, or events.
    - Study Tip: one short sentence to help remember or connect the information.

    Text:
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_length
    )
    return response.choices[0].message.content

# Keyword search function
def search_for_keywords(keyword, text):
    keyword = keyword.lower()
    results = []
    sentences = text.split(". ")
    for i, sentence in enumerate(sentences):
        if keyword in sentence.lower():
            results.append((i + 1, sentence.strip()))
    return results

# Split text into chunks
def chunk_text_by_words(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""
    matches = []

    if request.method == "POST":
        file = request.files["text_file"]
        keyword = request.form.get("keyword", "")

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

        # Unsupported file type
        else:
            return "Unsupported file type. Please upload a PDF or TXT file."

        # Clean text
        clean_text = " ".join(" ".join(all_text).split())

        # Keyword search
        if keyword:
            matches = search_for_keywords(keyword, clean_text)

        # Summarize text in chunks
        mini_summaries = []
        for chunk in chunk_text_by_words(clean_text):
            mini_summary = summarize_text(chunk, max_length=100)
            mini_summaries.append(mini_summary)

        combined_summary = "\n".join(mini_summaries)
        summary = summarize_text(combined_summary, max_length=200)

    return render_template("index.html", summary=summary, matches=matches)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
