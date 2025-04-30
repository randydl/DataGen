from pathlib import Path
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


if __name__ == '__main__':
    pipe = QAGenerator(llm,
        '/nas_data/userdata/randy/projects/DataGen/data/389'
    )
    pipe.run('/nas_data/userdata/randy/datasets/easy/1742796872855/files')
