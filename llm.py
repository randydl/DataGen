from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from utils import split_text
from prompts import SUMMARY_BASE, SUMMARY_REFINE


def create_model(model, base_url, api_key='EMPTY', temperature=None, max_tokens=None, **kwargs):
    return LLMChain(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )


class LLMChain:
    def __init__(self, **kwargs):
        self.llm = ChatOpenAI(**kwargs)
        self.refine = load_summarize_chain(
            self.llm, chain_type='refine',
            question_prompt=PromptTemplate.from_template(SUMMARY_BASE),
            refine_prompt=PromptTemplate.from_template(SUMMARY_REFINE)
        )

    def invoke(self, *args, **kwargs):
        return self.llm.invoke(*args, **kwargs)

    def summarize(self, texts):
        documents = [Document(x) for x in texts]
        summary = self.refine.invoke(documents)
        return summary.get('output_text')

    def split_text(self, *args, **kwargs):
        return split_text(*args, **kwargs)


if __name__ == '__main__':
    llm = create_model(
        # model='Qwen2.5-72B-Instruct',
        # base_url='http://10.252.36.118:18000/v1'
        model='QwQ-32B',
        base_url='http://10.252.36.121:18000/v1'
    )

    result = llm.invoke('DRAM的主要功能是什么？')

    # text = Path('example.md').read_text('utf-8')
    # texts = llm.split_text(text, chunk_size=5000)
    # result = llm.summarize(texts)

    print(result)
