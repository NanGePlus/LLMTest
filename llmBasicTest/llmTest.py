# 导入python标准库
import logging
# 导入第三方库
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
# 导入本应用程序提供的方法
from utils.myLLM import my_llm



# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test(llm, input_text):
    # (1)构造prompt
    prompt_template_system = PromptTemplate.from_file("prompt_template_system.txt")
    prompt_template_user = PromptTemplate.from_file("prompt_template_user.txt")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",prompt_template_system.template),
            ("human", prompt_template_user.template)
        ]
    )
    # (2)定义chain
    chain = prompt | llm
    # (3)调用chain进行查询
    result = chain.invoke(
        {"query": input_text}
    )
    return result


if __name__ == "__main__":
    # 选择使用哪一个大模型 openai:调用gpt大模型 oneapi:调用非gpt大模型(国产大模型等) ollama:调用本地大模型
    LLM_TYPE = "oneapi"

    # 测试数据
    input_text = "有没有土豪套餐"
    # input_text = "200元以下，流量大的套餐有啥"

    # 运行chain
    result = test(my_llm(LLM_TYPE), input_text)
    logger.info(f"搜索结果: {result}")



