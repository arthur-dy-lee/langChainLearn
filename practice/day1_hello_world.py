import sys
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 定义模型 (Model)
# 这里直接连接你本地的 Ollama 服务
llm = ChatOllama(
    model="deepseek-r1:8b",  # 你的具体模型 tag
    temperature=0.7,
    # keep_alive="1h" # 可选：让模型在显存驻留1小时，避免频繁加载
)

# 2. 定义提示词模板 (Prompt)
# system: 设定 AI 的角色
# user: 用户的输入，使用 {topic} 作为占位符
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深的 Python 技术专家，擅长用简洁的语言解释复杂概念。"),
    ("user", "请解释一下什么是 {topic}，并给出一个简单的应用场景。")
])

# 3. 组装链 (Chain) - 这就是 LCEL 的核心， LCEL 的全称是 LangChain Expression Language（LangChain 表达式语言）
# 逻辑流向：Prompt输入 -> 模型处理 -> 结果解析为纯文本
chain = prompt | llm | StrOutputParser()

# --- 调用方式 A: 直接调用 (Invoke) ---
print("--- 正在思考 (Invoke) ---")
topic = "LangChain LCEL"
# result = chain.invoke({"topic": topic})
# print(result)

# --- 调用方式 B: 流式输出 (Stream) ---
# 对于 72B 这样的大模型，流式输出能极大提升体验，减少等待焦虑
print(f"--- 正在流式输出关于 '{topic}' 的解释 ---\n")
try:
    for chunk in chain.stream({"topic": topic}):
        # flush 确保字符立即显示，而不是像 print 那样按行缓冲
        print(chunk, end="", flush=True)
    print("\n") # 结尾换行
except Exception as e:
    print(f"\n发生错误: {e}")
    print("请确保 Ollama 服务已启动 (ollama serve) 且模型名称正确。")