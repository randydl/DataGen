from pathlib import Path
from llm import create_model
from generator import QAGenerator


llm = create_model(
    model='QwQ-32B',
    base_url='http://10.252.36.121:18000/v1'
    # model='Qwen2.5-72B-Instruct',
    # base_url='http://10.252.36.118:18000/v1'
)


if __name__ == '__main__':
    pipe = QAGenerator(llm,
        '/nas_data/userdata/randy/projects/DataGen/data/389'
    )
    pipe.run('/nas_data/userdata/randy/datasets/easy/1742796872855/files')
