import json
from pathlib import Path
from joblib import Parallel, delayed
from tenacity import retry, stop_after_attempt
from utils import extract_answer, extract_think, extract_json
from prompts import get_question_prompt, get_answer_prompt, get_cot_prompt


MAX_RETRIES = 3
MAX_WORKERS = 64


class QAGenerator:
    def __init__(self, llm, output_dir):
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @retry(stop=stop_after_attempt(MAX_RETRIES))
    def generate_questions_from_text(self, text):
        prompt = get_question_prompt(text)
        result = self.llm.invoke(prompt).content
        result = extract_answer(result)
        result = extract_json(result)
        assert result, 'Failed to generate questions'
        return result

    @retry(stop=stop_after_attempt(MAX_RETRIES))
    def generate_answer_for_question(self, question, text=None):
        prompt = get_answer_prompt(text, question) if text else question
        result = self.llm.invoke(prompt).content
        answer = extract_answer(result)
        assert answer, 'Failed to generate answer'
        result = extract_think(result)
        return answer, result

    @retry(stop=stop_after_attempt(MAX_RETRIES))
    def generate_cot_for_qa(self, question, answer, think):
        if not think: return None
        prompt = get_cot_prompt(question, answer, think)
        result = self.llm.invoke(prompt).content
        result = extract_answer(result)
        assert result, 'Failed to generate COT'
        return result

    def process_question(self, question, text=None, question_id=None):
        try:
            answer, think = self.generate_answer_for_question(question, text)
            think = self.generate_cot_for_qa(question, answer, think)
            return {
                'id': question_id,
                'question': question,
                'answer': answer,
                'cot': think
            }
        except Exception as e:
            return None

    def build_questions(self, input_dir):
        def generator(text, chunk_id):
            try:
                questions = self.generate_questions_from_text(text)
                return {'id': chunk_id, 'text': text, 'questions': questions}
            except Exception as e:
                return None

        rawtxts = list(Path(input_dir).glob('*.txt'))
        workers = max(min(MAX_WORKERS, len(rawtxts)), 1)
        results = Parallel(workers, prefer='threads', verbose=10)(
            delayed(generator)(p.read_text('utf-8'), p.stem) for p in rawtxts
        )
        results = [r for r in results if r]
        self.save_results(results, 'questions.json')

    def build_datasets(self, questions_file):
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dataset = []
        for chunk in data:
            text = chunk['text']
            chunk_id = chunk['id']
            for i, q in enumerate(chunk['questions']):
                dataset.append([q, text, f'{chunk_id}-{i}'])

        workers = max(min(MAX_WORKERS, len(dataset)), 1)
        results = Parallel(workers, prefer='threads', verbose=10)(
            delayed(self.process_question)(*x) for x in dataset
        )
        results = [r for r in results if r]
        self.save_results(results, 'datasets.json')

    def save_results(self, results, file_name):
        with open(self.output_dir.joinpath(file_name), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
