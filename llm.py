from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from utils import split_text
from prompts import PROMPT, REFINE_PROMPT, CONDENSE_PROMPT


def create_model(**kwargs):
    return ChatOpenAI(
        model=kwargs.get('model'),
        api_key=kwargs.get('api_key', 'EMPTY'),
        base_url=kwargs.get('base_url'),
        max_tokens=kwargs.get('max_tokens'),
        temperature=kwargs.get('temperature')
    )


class LLMChain:
    def __init__(self, **kwargs):
        self.llm = create_model(**kwargs)

        self.refine = load_summarize_chain(
            self.llm, chain_type='refine',
            question_prompt=PromptTemplate.from_template(PROMPT),
            refine_prompt=PromptTemplate.from_template(REFINE_PROMPT)
        )

    def invoke(self, *args, **kwargs):
        return self.llm.invoke(*args, **kwargs)

    def summarize(self, texts):
        documents = [Document(x) for x in texts]
        summary = self.refine.invoke(documents)
        prompt_template  = PromptTemplate.from_template(CONDENSE_PROMPT)
        prompt_template = prompt_template.invoke({'structured_summary': summary.get('output_text')})

        return self.invoke(prompt_template.text)

    def split_text(self, *args, **kwargs):
        return split_text(*args, **kwargs)


if __name__ == '__main__':
    llm = LLMChain(
        # model='Qwen2.5-72B-Instruct',
        # base_url='http://10.252.36.118:18000/v1'
        model='QwQ-32B',
        base_url='http://10.252.36.121:18000/v1'
    )

    # result = llm.invoke('DRAM的主要功能是什么？')

    text = Path('example.md').read_text('utf-8')
    texts = llm.split_text(text, chunk_size=1000)
    result = llm.summarize(texts[:2])

    print(result)
