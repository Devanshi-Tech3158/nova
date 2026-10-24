"""LLM client interface and testing doubles."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for language model completion providers."""

    def complete(self, prompt: str, *, json: bool = False) -> str:
        """Generate a response completion for the prompt."""
        ...


class FakeLLM:
    """Scripted LLM implementation for offline testing."""

    def __init__(
        self,
        responses: list[str] | dict[str, str] | str | None = None,
        default_response: str = "",
    ) -> None:
        """Initialize FakeLLM with scripted responses."""
        self.default_response = default_response
        self.calls: list[dict[str, Any]] = []
        if isinstance(responses, str):
            self._responses_list = [responses]
            self._responses_dict: dict[str, str] = {}
        elif isinstance(responses, list):
            self._responses_list = list(responses)
            self._responses_dict = {}
        elif isinstance(responses, dict):
            self._responses_list = []
            self._responses_dict = dict(responses)
        else:
            self._responses_list = []
            self._responses_dict = {}

    def complete(self, prompt: str, *, json: bool = False) -> str:
        """Return scripted response matching the prompt or next in list."""
        self.calls.append({"prompt": prompt, "json": json})
        if prompt in self._responses_dict:
            return self._responses_dict[prompt]
        if self._responses_list:
            return self._responses_list.pop(0)
        return self.default_response
