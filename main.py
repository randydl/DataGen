import pandas as pd
from pathlib import Path
from loguru import logger
from joblib import Parallel, delayed
from tenacity import retry, stop_after_attempt

from llm import create_model
from utils import extract_answer, extract_think, extract_json, split_text
from prompts import get_question_prompt, get_answer_prompt, get_cot_prompt


llm = create_model(
    model='QwQ-32B',
    base_url='http://10.252.36.121:18000/v1'
    # model='Qwen2.5-72B-Instruct',
    # base_url='http://10.252.36.118:18000/v1'
)

@retry(stop=stop_after_attempt(3))
def gen_question(text):
    prompt = get_question_prompt(text)
    result = llm.invoke(prompt).content
    result = extract_answer(result)
    assert result is not None
    result = extract_json(result)
    assert result is not None
    return {
        'text': text,
        'questions': result
    }


@retry(stop=stop_after_attempt(3))
def gen_answer(text, question):
    prompt = get_answer_prompt(text, question)
    result = llm.invoke(prompt).content
    answer = extract_answer(result)
    assert answer is not None
    think = extract_think(result)
    return {
        'question': question,
        'answer': answer,
        'think': think
    }


@retry(stop=stop_after_attempt(3))
def gen_cot(question, answer, think):
    prompt = get_cot_prompt(question, answer, think)
    result = llm.invoke(prompt).content
    think = extract_answer(result)
    assert think is not None
    return {
        'question': question,
        'answer': answer,
        'think': think
    }

def process_question(text, question):
    try:
        result = gen_answer(text, question)
        if result['think'] is not None:
            result = gen_cot(**result)
        return result
    except Exception as e:
        return None


def process_chunk(text, id=None):
    try:
        result = gen_question(text)
        questions = result['questions']
        for i, question in enumerate(questions):
            answer = process_question(text, question)
            if answer is not None:
                questions[i] = answer
        return dict(id=id, **result)
    except Exception as e:
        return None


def process_chunks(chunks, ids=None):
    if ids is not None:
        assert len(chunks) == len(ids)


if __name__ == '__main__':
    text = Path('example.md').read_text()
