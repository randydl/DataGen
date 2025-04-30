from pathlib import Path
from loguru import logger
from joblib import Parallel, delayed
from llm import create_model
from generator import QAGenerator


llm = create_model(
    model='Qwen3-32B',
    base_url='http://10.252.36.121:18000/v1',
    temperature=0.7,
    top_p=0.8,
    extra_body={
        'top_k': 20,
        'chat_template_kwargs': {
            'enable_thinking': False
        }
    }
)


def main(p, pipe):
    try:
        text = p.read_text('utf-8')
        refined_text = pipe.refine_text(text)
        fdir = p.parent.joinpath('filter')
        fdir.mkdir(parents=True, exist_ok=True)
        fdir.joinpath(p.name).write_text(refined_text, 'utf-8')
        return True
    except Exception as e:
        logger.error(f'File: {p} - Info: {e}')
        return False


if __name__ == '__main__':
    pipe = QAGenerator(llm, output_dir='/tmp')
    root = Path('/nas_data/userdata/zhengwei/dataset/cxmt-semi/cptest_videos/chunks')
    txts = sorted(root.glob('*.txt'))

    results = Parallel(64, prefer='threads', verbose=10)(
        delayed(main)(p, pipe) for p in txts
    )

    num_success = sum(results)
    num_failed = len(results) - num_success
    logger.info(f'processed: {len(results)} | success: {num_success} | failed: {num_failed}')
