from pathlib import Path
from transformers import AutoTokenizer
from chonkie import RecursiveChunker, RecursiveRules
def split_text(text, chunk_size=5000):
    rules = RecursiveRules().to_dict()
    # rules['levels'][1]['delimiters'].extend(['。', '？', '！'])
    chunker = RecursiveChunker(
        AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'),
        chunk_size=chunk_size,
        rules=RecursiveRules.from_dict(rules),
        # return_type='texts'
    )
    chunks = chunker.chunk(text)
    return chunks


if __name__ == '__main__':
    text = Path('example.md').read_text()
    chunks = split_text(text)
    print(len(chunks))
    print(chunks[0])
