import sys
from typing import Annotated, Literal, TypedDict

# 导入必要的库
# 如果这里报错，请确保 pip install langgraph langchain-ollama
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ==========================================
# 1. 定义工具 (Tools)
# ==========================================
@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    print(f"    [工具日志] 正在计算: {a} + {b}")
    return a + b


@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的天气。
    注意：如果用户输入的是中文城市名，请务必将其转换为对应的汉语拼音 (Pinyin) 再调用此工具。
    例如：'上海' -> 'shanghai'。
    """
    # 同时匹配中文和拼音
    print(f"    [工具日志] 正在查询 {city} 的天气...")
    if "shanghai" in city.lower():
        return "晴天, 25°C"
    elif "beijing" in city.lower():
        return "多云, 18°C"
    else:
        return "未知天气"


tools = [add, get_weather]

# ==========================================
# 2. 初始化模型 (Model)
# ==========================================
# 使用你本地的 Qwen 模型
llm = ChatOllama(model="qwen2.5:72b-instruct-q3_K_M", temperature=0)

# 关键步骤：绑定工具
llm_with_tools = llm.bind_tools(tools)


# ==========================================
# 3. 定义状态 (State)
# ==========================================
class AgentState(TypedDict):
    # add_messages: 当有新消息时，追加到列表，而不是覆盖
    messages: Annotated[list, add_messages]


# ==========================================
# 4. 定义节点 (Nodes)
# ==========================================

# 节点 A: 思考 (Call Model)
def call_model(state: AgentState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# 节点 B: 执行工具 (Tool Node)
tool_node = ToolNode(tools)


# ==========================================
# 5. 定义逻辑跳转 (Edges)
# ==========================================

# 判断逻辑：决定 LLM 思考完之后去哪里
def should_continue(state: AgentState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]

    # 2. 定义转换规则 (如果是红灯，就停；如果是绿灯，就走)
    if last_message.tool_calls:
        return "tools"  # 规则 A: AI 想调工具 ->以此状态跳转到 Tools 节点

    # 否则 -> 结束
    return END


# ==========================================
# 6. 组装图 (Graph)
# ==========================================
workflow = StateGraph(AgentState)

# 6.1 添加节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 6.2 设置入口
workflow.add_edge(START, "agent")

# 6.3 添加条件边 (Agent 跑完后去哪？)
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# 6.4 添加普通边 (工具跑完后，必须回 Agent)
workflow.add_edge("tools", "agent")

# 6.5 编译
app = workflow.compile()

# ==========================================
# 7. 运行测试
# ==========================================
if __name__ == "__main__":
    print("--- 手动组装的 Graph Agent 启动 ---")

    # 一个需要调用两个工具的复杂问题
    query = "上海天气怎么样？如果是晴天，帮我算一下 10 加 20 是多少。"
    print(f"用户提问: {query}\n")

    inputs = {"messages": [HumanMessage(content=query)]}
    i = 0
    try:
        # stream_mode="values" 会打印每一步的状态
        """
        messages = [
                        HumanMessage(...),   # 0: 用户提问
                        AIMessage(...),      # 1: AI 决定调两个工具
                        ToolMessage(天气),   # 2: 天气工具的结果 (倒数第二个)
                        ToolMessage(加法)    # 3: 加法工具的结果 (倒数第一个) [-1]
                    ]
        """
        for event in app.stream(inputs, stream_mode="values"):
            i += 1
            last_msg = event["messages"][-1]

            # 为了输出清晰，只打印最新的一步
            step_type = type(last_msg).__name__

            print(f"\n--- 第 {i} 步: {step_type}, last_msg:{last_msg} ---")

            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                print(f"🤖 [AI 决策] 计划调用工具: {last_msg.tool_calls}")
            elif step_type == "ToolMessage":
                print(f"🔧 [工具结果] {last_msg.content}")
            elif hasattr(last_msg, 'content') and last_msg.content:
                print(f"📝 [AI 回复] {last_msg.content}")

        print("\n✅ 流程结束")

    except Exception as e:
        print(f"❌ 运行出错: {e}")
