import json
from pathlib import Path
from loguru import logger
from tqdm.auto import tqdm
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
    def generate_questions(self, text):
        prompt = get_question_prompt(text)
        result = self.llm.invoke(prompt).content
        result = extract_answer(result)
        result = extract_json(result)
        assert result, 'Failed to generate questions'
        return result

    @retry(stop=stop_after_attempt(MAX_RETRIES))
    def generate_answer(self, question, text=None):
        prompt = get_answer_prompt(text, question) if text else question
        result = self.llm.invoke(prompt).content
        answer = extract_answer(result)
        assert answer, 'Failed to generate answer'
        result = extract_think(result)
        return answer, result

    @retry(stop=stop_after_attempt(MAX_RETRIES))
    def generate_cot(self, question, answer, think):
        if not think: return None
        prompt = get_cot_prompt(question, answer, think)
        result = self.llm.invoke(prompt).content
        result = extract_answer(result)
        assert result, 'Failed to generate COT'
        return result

    def generate_entry(self, question, text=None, question_id=None):
        try:
            answer, think = self.generate_answer(question, text)
            think = self.generate_cot(question, answer, think)
            return {
                'id': question_id,
                'question': question,
                'answer': answer,
                'cot': think
            }
        except Exception as e:
            return None

    def split_chunks(self, input_dir, chunk_size=5000, overlap=None):
        out_dir = self.output_dir.joinpath('chunks')
        out_dir.mkdir(parents=True, exist_ok=True)

        counter = 0
        mdfiles = list(Path(input_dir).glob('*.md'))
        for p in tqdm(mdfiles):
            try:
                text = p.read_text(encoding='utf-8')
                chunks = self.llm.split_text(text, chunk_size, overlap)
                for i, chunk in enumerate(chunks):
                    path = out_dir.joinpath(f'{p.stem}-{i}.txt')
                    path.write_text(chunk, encoding='utf-8')
                counter += 1
            except Exception as e:
                logger.error(f'File: {p} - Info: {e}')

        logger.success(f'Processed {counter}/{len(mdfiles)} files.')

    def build_questions(self, input_dir):
        def generator(text, chunk_id):
            try:
                questions = self.generate_questions(text)
                return {'id': chunk_id, 'text': text, 'questions': questions}
            except Exception as e:
                return None

        results = []
        targets = list(Path(input_dir).glob('*.txt'))
        for i in tqdm(range(0, len(targets), MAX_WORKERS)):
            batch = targets[i:i + MAX_WORKERS]
            output = Parallel(len(batch), prefer='threads', verbose=10)(
                delayed(generator)(p.read_text('utf-8'), p.stem)
                for p in batch
            )
            results.extend(x for x in output if x)
            self.save_results(results, 'questions.json')

        logger.success('Question generation completed!')

    def build_datasets(self, questions_file):
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dataset = []
        for chunk in data:
            text = chunk['text']
            chunk_id = chunk['id']
            for i, q in enumerate(chunk['questions']):
                dataset.append([q, text, f'{chunk_id}-{i}'])

        results = []
        for i in tqdm(range(0, len(dataset), MAX_WORKERS)):
            batch = dataset[i:i + MAX_WORKERS]
            output = Parallel(len(batch), prefer='threads', verbose=10)(
                delayed(self.generate_entry)(*x)
                for x in batch
            )
            results.extend(x for x in output if x)
            self.save_results(results, 'datasets.json')

        logger.success('Dataset generation completed!')

    def save_results(self, results, file_name):
        with open(self.output_dir.joinpath(file_name), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def run(self, input_dir, chunk_size=5000, overlap=None):
        self.split_chunks(input_dir, chunk_size, overlap)
        self.build_questions(self.output_dir.joinpath('chunks'))
        self.build_datasets(self.output_dir.joinpath('questions.json'))
