from langchain_openai import ChatOpenAI


def create_model(model, base_url, api_key='EMPTY', temperature=None, max_tokens=None):
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature
    )


if __name__ == '__main__':
    import os
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''

    llm = create_model(
        # model='Qwen2.5-72B-Instruct',
        # base_url='http://10.252.36.118:18000/v1'
        model='QwQ-32B',
        base_url='http://10.252.36.121:18000/v1'
    )
    res = llm.invoke('DRAM的主要功能是什么？')
    print(res)
