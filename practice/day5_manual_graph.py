import sys
from typing import Annotated, Literal, TypedDict

# 导入必要的库
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ==========================================
# 1. 定义工具 (Tools) - Agent 的能力
# ==========================================
@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    print(f"    [工具日志] 正在计算: {a} + {b}")
    return a + b


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    print(f"    [工具日志] 正在查询 {city} 的天气...")
    # 模拟 API 返回
    if "shanghai" in city.lower():
        return "晴天, 25°C"
    elif "beijing" in city.lower():
        return "多云, 18°C"
    else:
        return "未知天气"


tools = [add, get_weather]

# ==========================================
# 2. 初始化模型 (Model) - Agent 的大脑
# ==========================================
# 使用你本地的 Qwen 模型
llm = ChatOllama(model="qwen2.5:72b-instruct-q3_K_M", temperature=0)

# 关键步骤：告诉模型有哪些工具可用
llm_with_tools = llm.bind_tools(tools)


# ==========================================
# 3. 定义状态 (State) - Agent 的记忆
# ==========================================
# 这是 LangGraph 最核心的概念。
# 我们定义一个字典，里面包含一个 'messages' 列表。
# Annotated[list, add_messages] 的意思是：
# 当节点返回新的 message 时，LangGraph 会自动把它 append (追加) 到列表末尾，而不是覆盖旧数据。
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ==========================================
# 4. 定义节点 (Nodes) - 流程中的步骤
# ==========================================

# 节点 A: 思考 (Call Model)
# 它的工作是：看一眼历史记录(state)，然后调用 LLM 生成下一步计划
def call_model(state: AgentState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    # 返回的内容会被 add_messages 自动追加到 state['messages'] 中
    return {"messages": [response]}


# 节点 B: 执行工具 (Tool Node)
# LangGraph 提供了一个现成的节点，它会自动扫描上一步 LLM 生成的 tool_calls 并执行
tool_node = ToolNode(tools)


# ==========================================
# 5. 定义边 (Edges) - 流程的逻辑跳转
# ==========================================

# 条件判断逻辑：决定 LLM 思考完之后去哪里
def should_continue(state: AgentState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]

    # 如果 LLM 决定调用工具 (tool_calls 列表不为空) -> 跳转到 'tools' 节点
    if last_message.tool_calls:
        return "tools"

    # 否则 (LLM 认为任务结束了，输出了最终文本) -> 结束流程
    return END


# ==========================================
# 6. 组装图 (Graph Construction)
# ==========================================
workflow = StateGraph(AgentState)

# 6.1 添加节点
workflow.add_node("agent", call_model)  # 这里的名字 'agent' 可以随便取
workflow.add_node("tools", tool_node)

# 6.2 设置入口点 (Start)
workflow.add_edge(START, "agent")

# 6.3 添加条件边 (Conditional Edge)
# 从 'agent' 节点出来后，根据 should_continue 的返回值决定去哪
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# 6.4 添加普通边 (Normal Edge)
# 工具执行完后，必须跳回 'agent'，让 LLM 看看工具的结果，然后决定下一步
workflow.add_edge("tools", "agent")

# 6.5 编译图 (Compile)
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

    # stream_mode="values" 会打印每次状态更新后的完整 message 列表
    try:
        for event in app.stream(inputs, stream_mode="values"):
            # event["messages"] 包含了当前所有的对话历史
            last_msg = event["messages"][-1]

            # 为了输出清晰，我们只打印最新的一步
            step_type = type(last_msg).__name__
            print(f"--- 状态更新 ({step_type}) ---")

            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                print(f"🤖 AI 计划: 准备调用工具 {last_msg.tool_calls}")
            elif hasattr(last_msg, 'content') and last_msg.content:
                print(f"📝 AI 回复: {last_msg.content}")
            elif step_type == "ToolMessage":
                print(f"🔧 工具结果: {last_msg.content}")

        print("\n✅ 流程结束")

    except Exception as e:
        print(f"❌ 运行出错: {e}")