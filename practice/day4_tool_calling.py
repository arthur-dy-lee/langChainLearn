from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


# --- 1. 定义工具 (Tools) ---
# 关键点：
# 1. 使用 @tool 装饰器
# 2. 函数的文档字符串 (Docstring) 必须写清楚！大模型是靠读这个来决定是否调用的。
# 3. 参数类型提示 (Type Hinting) 必须写清楚！大模型是靠这个来生成正确参数的。

@tool
def add(a: int, b: int) -> int:
    """将两个整数相加。

    Args:
        a: 第一个加数
        b: 第二个加数
    """
    print(f"\n[工具日志] 正在调用 add 工具: {a} + {b} ...")
    return a + b


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。

    Args:
        city: 城市名称 (例如: 'Beijing', 'Shanghai')
    """
    print(f"\n[工具日志] 正在查询 {city} 的天气 ...")
    # 模拟 API 返回
    if "shanghai" in city.lower():
        return "晴天, 25°C"
    elif "beijing" in city.lower():
        return "多云, 20°C"
    else:
        return "未知天气, 数据缺失"


# --- 2. 绑定工具 (Bind Tools) ---
# 初始化模型
llm = ChatOllama(model="qwen2.5:72b-instruct-q4_K_M", temperature=0)

# 将工具列表“挂载”给 LLM
# 这行代码会让 Qwen 知道它有了这两个技能
tools = [add, get_weather]
llm_with_tools = llm.bind_tools(tools)

# --- 3. 手动测试工具调用 (The Hard Way - 为了理解原理) ---
print("--- 测试 1: 简单的数学题 ---")
query1 = "请帮我计算 123 加 456 等于多少？"
messages1 = [HumanMessage(content=query1)]

# 调用模型
ai_msg1 = llm_with_tools.invoke(messages1)

# 观察结果
print(f"用户提问: {query1}")
print(f"模型回复类型: {type(ai_msg1)}")
print(f"工具调用请求 (tool_calls): {ai_msg1.tool_calls}")

# 如果模型决定调用工具，我们需要手动执行它
if ai_msg1.tool_calls:
    for tool_call in ai_msg1.tool_calls:
        # 1. 获取工具名和参数
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        # 2. 查找对应的函数并执行
        selected_tool = {"add": add, "get_weather": get_weather}[tool_name]
        tool_result = selected_tool.invoke(tool_args)

        print(f"工具执行结果: {tool_result}")

        # 3. 将结果构造成 ToolMessage 回传给模型 (闭环)
        # 这样模型才知道工具运行完了，可以生成最终回复了
        tool_msg = ToolMessage(
            tool_call_id=tool_id,
            content=str(tool_result),
            name=tool_name
        )
        messages1.append(ai_msg1)  # 把模型刚才的请求加进去
        messages1.append(tool_msg)  # 把工具的结果加进去

        # 4. 再次调用模型，让它生成最终的自然语言回复
        final_response = llm_with_tools.invoke(messages1)
        print(f"最终回复: {final_response.content}")

print("\n" + "=" * 30 + "\n")

# --- 4. 自动测试 (The Easy Way - 常用做法) ---
# 在实际开发中，我们通常不会像上面那样手写循环，而是用 langgraph 的 prebuilt 节点
# 但今天我们先手动跑通流程，理解原理最重要。

print("--- 测试 2: 多步推理 ---")
# 试着问一个需要调用两个工具的问题 (虽然现在的简单 Loop 可能处理不了多步，但看看 Qwen 的反应)
query2 = "上海现在的天气怎么样？另外算一下 10 加 20 是多少。"
print(f"用户提问: {query2}")
ai_msg2 = llm_with_tools.invoke([HumanMessage(content=query2)])
print(f"模型想要调用的工具: {ai_msg2.tool_calls}")