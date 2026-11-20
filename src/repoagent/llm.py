import json
import os
import logging

import openai
from openai import OpenAI

from repoagent.state import ToolCall

log = logging.getLogger(__name__)

class TruncatedToolCallError(Exception):
    pass

def to_api_messages(messages) -> list[dict]:
    """内部 Message -> SDK 格式。dict 转回 JSON 字符串只在这里做。"""
    out: list[dict] = []
    for m in messages:
        if m.role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id,
                "content": m.content,
            })
        elif m.role == "assistant" and m.tool_calls:
            out.append({
                "role": "assistant",
                "content": m.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.args, ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ],
            })
        else:
            out.append({"role": m.role, "content": m.content})
    return out

class LLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 provider_model: str | None = None):
        self.base_url = base_url or os.getenv(f"PROVIDER_A_BASE_URL")
        self.api_key = api_key or os.getenv(f"PROVIDER_A_API_KEY")
        self.provider_model = provider_model or os.getenv(f"PROVIDER_A_MODEL")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.calls: list[ToolCall] = []


    def call_model(self, messages, tools):
        try:
            resp = self.client.chat.completions.create(
                model=self.provider_model,
                messages=to_api_messages(messages),
                tools=tools or None,
                temperature=0
            )

            if resp.choices[0].finish_reason == "length":
                raise TruncatedToolCallError(
                    f"Tool call truncated (finish_reason=length). "
                    f"Increase max_tokens or split the tool arguments."
                )

            return resp.choices[0].message, resp.usage

        except openai.RateLimitError as e:
            # SDK 已经重试过，到这里说明配额/频率真的有问题
            # 记录日志，向上抛出自定义异常，让 Agent 层决定停止或降级
            log.error("rate limited: %s", e)
            raise

        except openai.APIConnectionError as e:
            # 连接失败或超时（SDK 也已重试过）
            log.error("connection error（常见原因：连接超时）: %s", e)
            raise

        except openai.BadRequestError as e:
            # 400：请求本身有问题（如上下文超长、参数非法），重试无用，直接抛出
            log.error("bad request（常见原因：上下文超长 / 参数非法）: %s", e)
            raise

    @staticmethod
    def _loads_lenient(arguments) -> dict | None:
        raw = arguments.strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)

        except json.JSONDecodeError:
            log.warning(f"Invalid JSON in %s", arguments[:200])
            return None

    def parse_output(self, message):
        text = message.content or ""
        calls : list[ToolCall] = []
        for tool_call in (message.tool_calls or []):
            args = self._loads_lenient(tool_call.function.arguments)
            if args is None:
                continue
            calls.append(ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                args=args
            ))
        return text, calls




