import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. 准备数据 (模拟你的本地知识库) ---
# 我们创建一个包含"私有知识"的文件。Qwen 72B 原本绝对不知道这件事。
file_path = "echo_vid_internal_doc.txt"
if not os.path.exists(file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("""
        项目名称: EchoVid
        核心功能: 这是一个基于 Docker 部署的视频处理系统。
        内部代号: Project-Banana
        管理员密码: admin_8888 (切勿泄露)
        已知Bug: 处理超过 1GB 的视频时会导致显存溢出，临时解决方案是重启容器。
        开发负责人: 你的名字
        """)
    print(f"已创建测试文件: {file_path}")

# --- 2. 加载与切分 (ETL) ---
# 加载
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

# 切分: 把长文章切成小块，方便检索
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.split_documents(docs)
print(f"文档已切分为 {len(splits)} 个片段")

# --- 3. 向量化与存储 (Indexing) ---
print("正在将文本向量化并存入 ChromaDB (可能需要几秒钟)...")
# 使用专门的 embedding 模型
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 创建向量数据库 (数据存在内存中，重启后消失。若要持久化需要指定 persist_directory)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 生成检索器: 告诉它每次只找最相关的 1 个片段
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# --- 4. 构建 RAG 链 (The Chain) ---
llm = ChatOllama(model="qwen2.5:72b-instruct-q4_K_M")

# 定义 Prompt: 关键在于 {context} 占位符
template = """你是一个智能助手。请严格基于下面的【上下文】回答问题。
如果上下文中没有答案，就说你不知道，不要编造。

【上下文】:
{context}

问题: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 辅助函数: 把检索到的文档列表拼成一个字符串
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# LCEL 组装:
# 1. RunnablePassthrough() 传递用户的问题
# 2. retriever | format_docs 用问题去检索文档，并把结果拼成字符串赋值给 context
# 3. 把 question 和 context 塞进 prompt
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. 提问测试 ---
question = "EchoVid 项目里遇到显存溢出怎么办？"
print(f"\n--- 提问: {question} ---")

# 流式输出结果
print("--- 回答 ---")
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)
print("\n")

# 测试它是否真的读了文件（问一个外部大模型不可能知道的细节）
question2 = "这个项目的内部代号是什么？"
print(f"--- 提问: {question2} ---")
print(rag_chain.invoke(question2))