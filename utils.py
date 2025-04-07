import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_think(text):
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        match = re.search(r'(.*?)</think>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def extract_answer(text):
    if '</think>' in text:
        return text.split('</think>', 1)[1].strip()
    elif '<think>' in text:
        return None
    return text.strip()


def split_text(text, chunk_size=5000, chunk_overlap=0):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
