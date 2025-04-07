from pathlib import Path
from chonkie import RecursiveChunker
from transformers import AutoTokenizer


def split_text(text, chunk_size=5000):
    chunker = RecursiveChunker(
        AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'),
        chunk_size=chunk_size,
        return_type='texts'
    )
    return chunker.chunk(text)


if __name__ == '__main__':
    text = Path('example.md').read_text()
    chunks = split_text(text)
    print(len(chunks))
    print(chunks[0])
