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
llm = ChatOllama(model="qwen2.5:72b-instruct-q3_K_M", temperature=0)

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

print("--- 测试 2: 多步/多工具推理 (完整手动闭环) ---")

query2 = "上海现在的天气怎么样？另外算一下 10 加 20 是多少。"
messages2 = [HumanMessage(content=query2)]

# 1. 第一轮调用：模型规划 (Planning)
print(f"用户提问: {query2}")
ai_msg2 = llm_with_tools.invoke(messages2)

# 把模型生成的“我想调用工具”这条消息，必须先加到历史记录里！
# 否则后面你直接塞 ToolMessage，模型会因上下文断裂而报错：“我没让你调工具啊，你给我结果干嘛？”
messages2.append(ai_msg2)

print(f"模型规划结果 (tool_calls): {ai_msg2.tool_calls}")

# 2. 中间处理：执行所有工具 (Execution Loop)
if ai_msg2.tool_calls:
    print("--- 开始执行工具列表 ---")

    # 定义可用工具映射表 (方便查找)
    available_tools = {
        "add": add,
        "get_weather": get_weather
    }

    # 遍历所有工具调用 (Qwen 可能会一次性返回两个：查天气 和 算加法)
    for tool_call in ai_msg2.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        # 找到对应的函数
        action_function = available_tools.get(tool_name)

        if action_function:
            # 执行函数
            print(f"正在执行: {tool_name} 参数: {tool_args}")
            tool_result = action_function.invoke(tool_args)

            # 构造 ToolMessage
            tool_msg = ToolMessage(
                tool_call_id=tool_id,  # 必须对应 ID，不然模型不知道这是哪个工具的结果
                content=str(tool_result),
                name=tool_name
            )

            # 将结果加入历史记录
            messages2.append(tool_msg)
            print(f"结果已回填: {tool_result}")

# 3. 最终轮调用：生成回答 (Final Generation)
print("--- 工具执行完毕，请求最终回复 ---")
# 现在的 messages2 里包含了：[用户问题, AI规划(ToolCall), 工具结果1, 工具结果2]
final_response = llm_with_tools.invoke(messages2)

print(f"\n最终回复: {final_response.content}")