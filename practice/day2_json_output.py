import pprint  # 用于漂亮的打印字典
from typing import List, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# 1. 定义数据结构 (Schema)
# 这就像是给大模型下达的“填空题模板”
class UserInfo(BaseModel):
    name: str = Field(description="用户的姓名")
    age: int = Field(description="用户的年龄，推测不出则为0")
    hobbies: List[str] = Field(description="用户的爱好列表")
    is_developer: bool = Field(description="用户是否看起来像开发者")


# 2. 初始化模型
# Qwen 2.5 对 JSON 格式的支持非常好
llm = ChatOllama(model="qwen2.5:72b-instruct-q3_K_M", temperature=0)

# 3. 关键步骤：绑定结构化输出
# 这行代码的魔法在于：它会自动修改 Prompt，告诉模型“必须输出符合 UserInfo 定义的 JSON”
structured_llm = llm.with_structured_output(UserInfo)

# 4. 定义 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的信息提取助手。请从用户的输入中提取关键信息。"),
    ("user", "{input_text}")
])

# 5. 组装链 (Chain)
# 注意：这里不需要 OutputParser，因为 structured_llm 直接返回 Pydantic 对象
chain = prompt | structured_llm

# --- 测试数据 ---
user_input = "我是张三，今年30岁了。平时喜欢写Python代码，周末偶尔去跑马拉松。"

print(f"--- 原始输入 ---\n{user_input}\n")
print("--- 正在提取结构化数据 ---")

try:
    # 调用链
    result = chain.invoke({"input_text": user_input})

    # 结果不再是字符串，而是 UserInfo 类的实例！
    print(f"\n提取结果类型: {type(result)}")
    print("\n--- 字段解析 ---")
    print(f"姓名: {result.name}")
    print(f"爱好: {result.hobbies}")
    print(f"是开发者吗?: {result.is_developer}")

    # 如果你想转回标准的 dict
    print("\n--- 完整 JSON ---")
    pprint.pprint(result.model_dump())

except Exception as e:
    print(f"发生错误: {e}")