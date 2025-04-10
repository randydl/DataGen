import re
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_answer(text):
    text = text.strip()
    if '</think>' in text:
        return text.split('</think>', 1)[1].strip() or None
    elif '<think>' in text:
        return None
    return text or None


def extract_think(text):
    patterns = [
        r'<think>(.*?)</think>',
        r'(.*?)</think>'
    ]
    for pattern in patterns:
        if match := re.search(pattern, text, re.DOTALL):
            return match.group(1).strip() or None
    return None


def extract_json(text):
    match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
    if not match: return None
    try:
        return json.loads(match.group(1).strip()) or None
    except json.JSONDecodeError:
        return None


def split_text(text, chunk_size=5000, chunk_overlap=None):
    if chunk_overlap is None:
        chunk_overlap = chunk_size // 10

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
