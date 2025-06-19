import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline,
)
from utils.chunking import chunk_text
from utils.bias_checker import bias_check

router = APIRouter()

# === GLOBAL CANCEL FLAG ===
should_cancel = False

# === Configuration ===
LOCAL_MODEL_PATHS = {
    "pegasus": "/Users/linux/Desktop/sumbookbot/Backend/models/pegasus",
    "bart": "/Users/linux/Desktop/sumbookbot/Backend/models/bart",
    "bertsum": "/Users/linux/Desktop/sumbookbot/Backend/models/bertsum"
}

LENGTH_MAP = {
    "detailed": "pegasus",
    "medium": "bart",
    "short": "bertsum"
}

def validate_local_model(model_path: str) -> bool:
    path = Path(model_path)
    if not path.exists():
        return False
    required_files = ["config.json"]
    return all((path / file).exists() for file in required_files)

def load_model_and_tokenizer(model_key: str):
    model_path = LOCAL_MODEL_PATHS[model_key]
    if not validate_local_model(model_path):
        raise Exception(f"Model directory not found or incomplete: {model_path}")
    print(f"Loading {model_key} from local path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if model_key == "bertsum":
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    return tokenizer, model

class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Text to summarize")
    summary_length: str = Field("medium", description="'short', 'medium', or 'detailed'")

@router.post("/")
def summarize(req: SummarizeRequest):
    global should_cancel
    should_cancel = False  # Reset flag for new job

    length_key = req.summary_length.lower()
    if length_key not in LENGTH_MAP:
        raise HTTPException(400, "Invalid summary_length. Use 'detailed', 'medium', or 'short'.")

    model_key = LENGTH_MAP[length_key]

    try:
        tokenizer, model = load_model_and_tokenizer(model_key)
    except Exception as e:
        print("Failed to load model", str(e))
        raise HTTPException(500, f"Failed to load model: {str(e)}")

    # === Dynamic chunk size and overlap ===
    if model_key in ["pegasus", "bart"]:
        chunk_size = 480
        overlap = 50
    else:  # BERTSum
        chunk_size = 250
        overlap = 30

    chunks = chunk_text(req.text, max_words=chunk_size, overlap=overlap)
    MAX_PARTS = 20

    # === PEGASUS ===
    if model_key == "pegasus":
        abstractive = pipeline("summarization", model=model, tokenizer=tokenizer, truncation=True)

        drafts = []
        for chunk in chunks:
            if should_cancel:
                print("🚫 Summarization cancelled by user.")
                break
            drafts.append(abstractive(chunk, max_length=400, min_length=50, truncation=True)[0]['summary_text'])

        if should_cancel:
            return {"summary": "Summarization was cancelled."}

        combined_draft = "\n".join(drafts)

        # === Extractive with BERTSum ===
        bert_tok = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATHS['bertsum'])
        bert_mod = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATHS['bertsum'])
        extractive = pipeline("text-classification", model=bert_mod, tokenizer=bert_tok, return_all_scores=True)

        sentences = [s.strip() for s in req.text.replace('\n', ' ').split('. ') if s.strip()]
        scored = []
        for sent in sentences:
            if should_cancel:
                print("🚫 Cancelled during extractive step.")
                break
            scores = extractive(sent)[0]
            key_score = next(item['score'] for item in scores if item['label'] in ('LABEL_1', '1'))
            scored.append((key_score, sent))

        top_n = sorted(scored, key=lambda x: x[0], reverse=True)[:3]
        key_points = '. '.join([s for _, s in top_n])

        combined = combined_draft + "\n" + key_points

        # === Optional final pass ===
        if len(drafts) > MAX_PARTS:
            print("⚡ Performing second pass for compression...")
            combined = abstractive(combined, max_length=150, min_length=50, truncation=True)[0]['summary_text']

        return {"summary": bias_check(combined)}

    # === BART ===
    elif model_key == "bart":
        pipe = pipeline("summarization", model=model, tokenizer=tokenizer, truncation=True)
        summarize_chunk = lambda t: pipe(t, max_length=150, min_length=60, truncation=True)[0]['summary_text']

        parts = []
        for chunk in chunks:
            if should_cancel:
                print("🚫 Summarization cancelled by user.")
                break
            parts.append(bias_check(summarize_chunk(chunk)))

        if should_cancel:
            return {"summary": "Summarization was cancelled."}

        summary_text = "\n".join(parts)

        if len(parts) > MAX_PARTS:
            print("⚡ Performing second pass for compression...")
            summary_text = summarize_chunk(summary_text)

        return {"summary": summary_text}

    # === BERTSum ===
    else:
        cls_model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATHS[model_key])
        classifier = pipeline("text-classification", model=cls_model, tokenizer=tokenizer, return_all_scores=True, truncation=True)

        sentences = [s.strip() for s in req.text.replace('\n', ' ').split('. ') if s.strip()]
        scored = []
        for sent in sentences:
            if should_cancel:
                print("🚫 Summarization cancelled by user.")
                break
            scores = classifier(sent)[0]
            key_score = next(item['score'] for item in scores if item['label'] in ('LABEL_1', '1'))
            scored.append((key_score, sent))

        if should_cancel:
            return {"summary": "Summarization was cancelled."}

        top_n = sorted(scored, key=lambda x: x[0], reverse=True)[:5]
        extractive_summary = '. '.join([s for _, s in top_n])

        peg_tok = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATHS['pegasus'])
        peg_mod = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_PATHS['pegasus'])
        rephraser = pipeline("summarization", model=peg_mod, tokenizer=peg_tok, truncation=True)
        final = rephraser(extractive_summary, max_length=100, min_length=40, truncation=True)[0]['summary_text']

        return {"summary": bias_check(final)}

# === CANCEL ENDPOINT ===
@router.post("/cancel")
def cancel_summarization():
    global should_cancel
    should_cancel = True
    print("✅ Received cancel request from frontend.")
    return {"status": "cancelled"}
