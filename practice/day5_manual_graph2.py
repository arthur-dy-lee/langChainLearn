import sys
from typing import Annotated, Literal, TypedDict

# 【核心概念解释】
# ChatOllama: 负责和本地大模型（大脑）对话
# @tool: 把普通 Python 函数包装成大模型能看懂的“工具说明书”
# HumanMessage, AIMessage, ToolMessage: 消息的三种基本类型（人说的、AI说的、工具返回的）
# StateGraph: 状态图，用来编排整个工作流的蓝图
# add_messages: 一个神奇的合并函数，专门用来处理消息列表的追加
# ToolNode: LangGraph 自带的一个节点，专门用来执行工具
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ==========================================
# 1. 定义工具 (Tools) —— 提供给大模型的“手脚”
# ==========================================
@tool
def add(a: int, b: int) -> int:
    # 这里的 """docstring""" 非常重要！
    # 大模型看不到函数的具体代码，它完全是通过这里的文字注释来知道这个工具是干嘛的。
    """计算两个整数的和。"""
    print(f"\n    [🔧 工具底层执行] 正在计算: {a} + {b}")
    return a + b


@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的天气。
    注意：如果用户输入的是中文城市名，请务必将其转换为对应的汉语拼音 (Pinyin) 再调用此工具。
    例如：'上海' -> 'shanghai'。
    """
    print(f"\n    [🔧 工具底层执行] 正在查询 {city} 的天气...")
    if "shanghai" in city.lower():
        return "晴天, 25°C"
    elif "beijing" in city.lower():
        return "多云, 18°C"
    else:
        return "未知天气"


# 把所有可用工具打包成一个列表，待会儿要交接给大模型
tools = [add, get_weather]

# ==========================================
# 2. 初始化模型 (Model) —— 聪明的大脑
# ==========================================
# temperature=0 表示让大模型的回答尽量严谨、确定，不乱发散（适合做逻辑处理和调用工具）
llm = ChatOllama(model="qwen2.5:72b-instruct-q3_K_M", temperature=0)

# 【核心步骤】：绑定工具
# 这一步并不会执行工具！
# 它的底层逻辑是：把 tools 列表里的函数名、参数要求、文档注释（docstring），
# 转换成 JSON Schema 格式，附带在每次发给大模型的 Prompt 里面。
# 这样大模型就知道：“哦，我除了会聊天，还会用这几个技能。”
llm_with_tools = llm.bind_tools(tools)


# ==========================================
# 3. 定义状态 (State) —— 贯穿全图的“共享记事本”
# ==========================================
class AgentState(TypedDict):
    # state 就是一个字典，目前里面只有一个键叫 "messages"
    # Annotated 和 add_messages 的组合非常巧妙：
    # 它规定了每次节点往 "messages" 里写数据时，不是“替换”旧内容，而是把新消息“追加(append)”到旧列表后面。
    # 这样大模型才能记住完整的聊天上下文！
    messages: Annotated[list, add_messages]


# ==========================================
# 4. 定义节点 (Nodes) —— 流程中的“工作车间”
# ==========================================

# 工作车间 A: 思考节点 (调用大模型)
# 每次执行到这里，都会把当前“记事本(state)”里的所有消息发给大模型
def call_model(state: AgentState):
    # 1. 取出聊天记录
    messages = state['messages']

    # 2. 发给大模型思考 (如果模型觉得需要调用工具，它的 response 里会包含 tool_calls 属性)
    response = llm_with_tools.invoke(messages)

    # 3. 返回**新产生**的消息。
    # 因为我们在 AgentState 中设置了 add_messages，所以这句 return 的实际效果是：
    # state['messages'].append(response)
    return {"messages": [response]}


# 工作车间 B: 工具节点 (执行实际的 Python 函数)
# LangGraph 的 ToolNode 已经帮我们写好了底层逻辑：
# 它会自动去查看最新一条消息里有没有大模型发出的 "tool_calls" 请求，
# 如果有，它会自动解析参数，运行对应的 Python 函数(add 或 get_weather)，
# 最后把函数的返回值包装成一个 "ToolMessage" 返回，追加到记事本里。
tool_node = ToolNode(tools)


# ==========================================
# 5. 定义逻辑跳转 (Edges) —— 决定去哪里的“红绿灯”
# ==========================================

# 这是一个路由器函数：当“思考节点 (Agent)”跑完后，到底该去哪？
def should_continue(state: AgentState) -> Literal["tools", END]:
    # 拿到最新的那条消息（大模型刚刚生成的）
    messages = state['messages']
    last_message = messages[-1]

    # 【核心判断逻辑】：
    # 大模型如果想用工具，它生成的 AIMessage 对象里，会自动带上一个叫 `tool_calls` 的列表。
    # 如果这个列表存在且不为空，说明大模型遇到了靠自己解决不了的问题，要求助工具了。
    if last_message.tool_calls:
        return "tools"  # 亮绿灯 -> 走向工具车间 (ToolNode)

    # 如果 `tool_calls` 为空，说明大模型觉得任务已经完成，直接给出了文本回答。
    # 亮红灯 -> 走向终点 (END)
    return END


# ==========================================
# 6. 组装图 (Graph) —— 把车间和马路连起来
# ==========================================
workflow = StateGraph(AgentState)

# 6.1 建房子：把刚才定义的两个“车间”放进图里，并给它们起个名字
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 6.2 修马路 (无条件边)：只要流程启动(START)，必定无条件进入 "agent" 车间
workflow.add_edge(START, "agent")

# 6.3 设岗亭 (条件边)："agent" 车间工作完出来，遇到分叉路口，去哪由 should_continue 函数决定
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# 6.4 修马路 (无条件边)："tools" 车间工作完，拿到结果后，必须无条件把结果送回给 "agent" 车间进行下一步判断
workflow.add_edge("tools", "agent")

# 6.5 竣工验收：把设计图编译成一个可以运行的应用
app = workflow.compile()

# ==========================================
# 7. 运行测试
# ==========================================
if __name__ == "__main__":
    print("=== 手动组装的 Graph Agent 启动 ===")

    query = "上海天气怎么样？如果是晴天，帮我算一下 10 加 20 是多少。"
    print(f"👨 提问: {query}\n")

    # 构造初始状态：记事本里只有用户说的一句话 (HumanMessage)
    inputs = {"messages": [HumanMessage(content=query)]}
    i = 0

    try:
        print(">>> 开始流式追踪运行状态 (stream_mode='values') >>>")
        # stream_mode="values" 会在每次图中的节点工作完、更新完 state 后，把完整的 state 抛出来
        for event in app.stream(inputs, stream_mode="values"):
            i += 1
            # 取出当前记事本里最后一条消息
            last_msg = event["messages"][-1]
            step_type = type(last_msg).__name__

            print(f"\n--- 步骤 {i} 完成 ---")

            # 判断刚刚产生的这条消息是什么类型
            if step_type == "HumanMessage":
                print(f"👤 [人类消息存入] {last_msg.content}")

            elif step_type == "AIMessage":
                # AI 消息有两种情况：1. 想调工具 2. 直接回复
                if last_msg.tool_calls:
                    print(f"🤖 [大脑思考 (Agent)] 遇到问题了，计划调用工具: ")
                    for tool_call in last_msg.tool_calls:
                        print(f"   -> 工具名称: {tool_call['name']}, 参数: {tool_call['args']}")
                else:
                    print(f"📝 [大脑思考 (Agent)] 认为任务完成，输出最终回复:\n{last_msg.content}")

            elif step_type == "ToolMessage":
                print(f"📦 [工具执行完成 (Tools)] 将结果返回给 AI: {last_msg.content}")

        print("\n✅ 流程完全结束")

    except Exception as e:
        print(f"❌ 运行出错: {e}")