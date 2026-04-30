"""
LLM Backend - Abstraction layer for different LLM providers.
Supports OpenAI (GPT-4), Anthropic (Claude), and DeepSeek.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    latency_seconds: float = 0.0


class LLMBackend:
    """
    Unified LLM backend supporting multiple providers.
    Handles rate limiting, retries, and token counting.
    """

    def __init__(self, config: Dict[str, Any]):
        # Default to DeepSeek; OpenAI and Anthropic remain supported via config.
        self.provider = config.get("provider", "deepseek")
        self.model = config.get("model", "deepseek-chat")
        self.seed = config.get("seed", None)
        self.default_temperature = config.get("temperature", 0.7)
        self.default_max_tokens = config.get("max_tokens", 2000)
        self.top_p = config.get("top_p", 0.95)
        self.requests_per_minute = config.get("requests_per_minute", 20)
        self.retry_max = config.get("retry_max", 3)
        self.retry_delay = config.get("retry_delay", 5)
        self._seed_unsupported_logged = False

        self._last_request_time = 0.0
        if self.requests_per_minute > 0:
            self._min_interval = 60.0 / self.requests_per_minute
        else:
            self._min_interval = 0.0
            logger.warning(
                "Invalid requests_per_minute=%s; disabling rate limit interval.",
                self.requests_per_minute,
            )

        # Track input and output tokens separately for accurate cost estimation
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_requests = 0

        # Initialize client
        self._init_client(config)

    def _init_client(self, config: Dict[str, Any]):
        """Initialize the appropriate LLM client."""
        api_key_env = config.get("api_key_env", "DEEPSEEK_API_KEY")
        api_key = os.environ.get(api_key_env)

        if not api_key:
            raise ValueError(
                f"API key not found. Set the {api_key_env} environment variable."
            )

        if self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("Install openai: pip install openai")

        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")

        elif self.provider == "deepseek":
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com",
                )
            except ImportError:
                raise ImportError("Install openai: pip install openai")

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            sleep_time = self._min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            stop: Optional stop sequences.

        Returns:
            LLMResponse object.
        """
        temp = temperature if temperature is not None else self.default_temperature
        max_tok = max_tokens if max_tokens is not None else self.default_max_tokens

        for attempt in range(self.retry_max):
            try:
                self._rate_limit()
                start_time = time.time()

                if self.provider in ("openai", "deepseek"):
                    response = self._call_openai(messages, temp, max_tok, stop)
                elif self.provider == "anthropic":
                    response = self._call_anthropic(messages, temp, max_tok, stop)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")

                response.latency_seconds = time.time() - start_time

                # Track tokens separately
                self._total_prompt_tokens += response.usage.get("prompt_tokens", 0)
                self._total_completion_tokens += response.usage.get("completion_tokens", 0)
                self._total_requests += 1

                logger.debug(
                    f"LLM response: {len(response.content)} chars, "
                    f"{response.usage}, {response.latency_seconds:.2f}s"
                )
                return response

            except Exception as e:
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.retry_max}): {e}"
                )
                if attempt < self.retry_max - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def _call_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> LLMResponse:
        """Call OpenAI-compatible API (also used for DeepSeek)."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": self.top_p,
        }
        if stop:
            kwargs["stop"] = stop
        if self.seed is not None:
            kwargs["seed"] = self.seed

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        return LLMResponse(
            content=choice.message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            finish_reason=choice.finish_reason,
        )

    def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> LLMResponse:
        """
        Call Anthropic API.

        FIX for Issue #1: Concatenate ALL system messages into a single
        system string instead of only keeping the last one.  This ensures
        the agent identity prompt is never lost when memory context is
        appended as a second system message.
        """
        system_parts: List[str] = []
        chat_messages: List[Dict[str, str]] = []

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                chat_messages.append(msg)

        # Join all system messages with a clear separator
        system_msg = "\n\n---\n\n".join(system_parts) if system_parts else ""

        if self.seed is not None and not self._seed_unsupported_logged:
            logger.info(
                "Anthropic does not support llm.seed; ignoring configured seed."
            )
            self._seed_unsupported_logged = True

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if stop:
            kwargs["stop_sequences"] = stop

        response = self.client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics with separate input/output token counts."""
        return {
            "total_requests": self._total_requests,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens_used": self._total_prompt_tokens + self._total_completion_tokens,
            "provider": self.provider,
            "model": self.model,
        }
