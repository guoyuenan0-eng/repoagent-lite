"""记录 provider 的 function calling 行为。这是 D2 要用的全部信息。"""
import json, os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "读取文件内容",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

def probe(name, base_url, api_key, model):
    print(f"\n{'='*60}\n{name} / {model}\n{'='*60}")
    c = OpenAI(base_url=base_url, api_key=api_key)

    r = c.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "读一下 README.md"}],
        tools=[TOOL],
    )
    m = r.choices[0].message
    print("finish_reason:", r.choices[0].finish_reason)
    print("usage:", r.usage)
    print("tool_calls raw:", json.dumps(
        [tc.model_dump() for tc in (m.tool_calls or [])], ensure_ascii=False, indent=2))
    print("arguments 类型:", type(m.tool_calls[0].function.arguments).__name__ if m.tool_calls else "N/A")

    # 回传工具结果，看第二轮是否正常
    if m.tool_calls:
        msgs = [
            {"role": "user", "content": "读一下 README.md"},
            m,
            {"role": "tool", "tool_call_id": m.tool_calls[0].id,
             "content": "# repoagent-lite\n空文件"},
        ]
        r2 = c.chat.completions.create(model=model, messages=msgs, tools=[TOOL])
        print("第二轮 finish_reason:", r2.choices[0].finish_reason)
        print("第二轮内容:", (r2.choices[0].message.content or "")[:120])

    # 流式下 tool_call 如何分片 —— 最大的坑
    print("\n--- streaming ---")
    stream = c.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "读一下 README.md"}],
        tools=[TOOL], stream=True,
    )
    chunks = 0
    for chunk in stream:
        d = chunk.choices[0].delta
        if getattr(d, "tool_calls", None):
            print("delta.tool_calls:", d.tool_calls[0].model_dump())
            chunks += 1
        # if chunks >= 3:
        #     break
    print("（注意 index / id / arguments 是否分多次到达）")

if __name__ == "__main__":
    for tag in ("A", "B"):
        probe(f"provider {tag}",
              os.getenv(f"PROVIDER_{tag}_BASE_URL"),
              os.getenv(f"PROVIDER_{tag}_API_KEY"),
              os.getenv(f"PROVIDER_{tag}_MODEL"))