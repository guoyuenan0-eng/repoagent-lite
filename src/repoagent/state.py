from enum import Enum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]

class ToolSpec(BaseModel):
    name: str = ""
    description: str = ""
    params_schema: dict = Field(default_factory=dict)


class ToolCall(BaseModel):
    id: str = ""
    name: str = ""
    args: dict = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str = ""


class ToolResult(BaseModel):
    tool_call_id: str
    ok: bool
    content: str = ""
    error: str | None = None


class StopReason(str, Enum):
    NONE = "none"
    MAX_STEPS = "max_steps"
    BUDGET_EXCEEDED = "budget_exceeded"
    ARTIFACT_VALID = "artifact_valid"
    NO_PROGRESS = "no_progress"
    USER_ACTION = "user_action"
    ERROR = "error"


class Status(str, Enum):
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    DONE = "done"
    FAILED = "failed"


class Budget(BaseModel):
    """预算"""
    max_steps: int = 20
    max_tokens: int = 200_000
    max_usd: float = 0.5

    steps_used: int = 0
    tokens_used: int = 0
    usd_used: float = 0.0

    def spend(self, tokens: int, usd: float) -> None:
        self.steps_used += 1
        self.usd_used += usd
        self.tokens_used += tokens

    def exhausted(self) -> StopReason | None:
        if self.steps_used >= self.max_steps:
            return StopReason.MAX_STEPS
        if self.tokens_used >= self.max_tokens or self.usd_used >= self.max_usd:
            return StopReason.BUDGET_EXCEEDED
        return None


class ArtifactMeta(BaseModel):
    """产物契约：模型产出的东西必须满足约定结构，下游才能依赖。"""
    path: str
    sha256: str = ""
    size: int = 0
    validated: bool = False

    @staticmethod
    def check_artifact(path: str, sha256: str, size: int, validated: bool) -> bool:
        p = PurePosixPath(path)
        if p.is_absolute() or path.startswith(("/", "\\")) or ".." in p.parts:
            return False
        if sha256 != "":
            return False
        if size < 0:
            return False
        if not validated:
            return False
        return True

    def check(self) -> StopReason | None:
        if not self.check_artifact(self.path, self.sha256, self.size, self.validated):
            return StopReason.ARTIFACT_VALID
        return None


class Step(BaseModel):
    index: int
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    assistant_text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    latency_ms: int = 0
    stop_reason: StopReason | None = None


class SessionState(BaseModel):
    session_id: str
    status: Status = Status.RUNNING    # running / waiting_user / done / failed
    messages: list[Message] = Field(default_factory=list)         # 模型视角的上下文
    steps: list[Step] = Field(default_factory=list)         # 审计视角的事实（两者不要混）
    budget: Budget = Field(default_factory=Budget)
    artifacts: dict[str, ArtifactMeta] = Field(default_factory=dict)   # 产物契约（你的强项，直接用上)


if __name__ == '__main__':
    s = SessionState(session_id="t1")
    s.messages.append(Message(role="user", content="读一下 README.md"))
    s.steps.append(Step(index=0, tokens_in=12, tokens_out=8, usd=0.0001))
    s.budget.spend(tokens=20, usd=0.0001)
    s.artifacts["README.md"] = ArtifactMeta(path="../../README.md", size=20, validated=True)
    js = s.model_dump_json(indent=2)
    print(js)
    # 关键验收：存下去能原样恢复，且恢复后能继续跑
    back = SessionState.model_validate_json(js)
    assert back.session_id == "t1"
    assert back.messages[0].content == "读一下 README.md"
    assert back.artifacts["README.md"].validated is True
    assert back.budget.steps_used == 1
    print("\n[ok] 序列化 + 恢复通过")
