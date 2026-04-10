import os
import pdfplumber
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks

def load_and_chunk_pdfs(folder_path):
    all_chunks = []

    for file in tqdm(os.listdir(folder_path)):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            print(f"Processing {file}...")
            text = extract_text_from_pdf(path)
            chunks = chunk_text(text)

            for c in chunks:
                all_chunks.append({
                    "text": c,
                    "source": file
                })

    return all_chunks
