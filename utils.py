import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_think(text):
    patterns = [
        r'<think>(.*?)</think>',
        r'(.*?)</think>'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def extract_answer(text):
    if '</think>' in text:
        return text.split('</think>', 1)[1].strip()
    elif '<think>' in text:
        return None
    return text.strip()


def split_text(text, chunk_size=5000, chunk_overlap=None):
    if chunk_overlap is None:
        chunk_overlap = chunk_size // 10

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
