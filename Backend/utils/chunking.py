# utils/chunking.py

import re

def chunk_text(text, max_words=480, overlap=50):
    """
    Splits text into word chunks with optional overlap for context.
    """
    words = re.findall(r'\S+', text)
    chunks = []
    i = 0

    while i < len(words):
        end = i + max_words
        chunk = words[i:end]
        chunks.append(' '.join(chunk))
        i += max_words - overlap

    return chunks
