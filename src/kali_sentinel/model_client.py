"""Client for the hosted KALI Sentinel Model API.

Integrates with the live hosted KALI model at:
https://unportentous-vanna-unpretentiously.ngrok-free.dev

Handles:
- POST /generate with {"message": "<prompt>"}
- Header injection (ngrok-skip-browser-warning)
- Output parsing and clean extraction of reasoning and response blocks
- Safe offline fallback when API is unreachable
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Optional

from kali_sentinel.prompts import KALI_SENTINEL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_KALI_API_URL = os.getenv(
    "KALI_API_URL", "https://unportentous-vanna-unpretentiously.ngrok-free.dev"
)

_CLIENT_INSTANCE: Optional[KaliModelClient] = None


class KaliModelClient:
    """Client for generating security responses via the hosted KALI Sentinel Model API.
    
    Supports both:
    1. Native KALI / ngrok /generate endpoint
    2. Standard OpenAI-compatible /v1/chat/completions (vLLM, Ollama, LM Studio, llama.cpp)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 35.0,
    ) -> None:
        raw_url = base_url or os.getenv("KALI_API_URL") or DEFAULT_KALI_API_URL
        self.base_url = raw_url.rstrip("/")
        self.model_name = model_name or os.getenv("KALI_MODEL_NAME") or "Qwen3-8B"
        self.api_key = os.getenv("KALI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout

    def health(self) -> bool:
        """Check if the KALI Model API is reachable."""
        probe_paths = ["/health", "/v1/models", ""]
        headers = {
            "ngrok-skip-browser-warning": "true",
            "User-Agent": "curl/7.68.0",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for path in probe_paths:
            try:
                url = f"{self.base_url}{path}" if path else self.base_url
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=3) as r:
                    if r.status in (200, 204):
                        return True
            except Exception:
                continue
        return False

    def generate(self, message: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Generate text from KALI model using /generate or /v1/chat/completions."""
        if not message or not message.strip():
            return None

        clean_message = message.strip()
        system_text = (system_prompt or KALI_SENTINEL_SYSTEM_PROMPT).strip()

        # Format prompt
        prompt_formatted = (
            f"<|im_start|>system\n{system_text}<|im_end|>\n"
            f"<|im_start|>user\n{clean_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        payload_data = {
            "model": self.model_name,
            "message": clean_message,
            "system": system_text,
            "prompt": prompt_formatted,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": clean_message},
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
        }
        payload = json.dumps(payload_data).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
            "User-Agent": "curl/7.68.0",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Determine target endpoint URLs based on base_url structure
        if self.base_url.endswith("/v1"):
            target_urls = [f"{self.base_url}/chat/completions", f"{self.base_url}/generate"]
        elif self.base_url.endswith("/chat/completions") or self.base_url.endswith("/generate"):
            target_urls = [self.base_url]
        else:
            target_urls = [
                f"{self.base_url}/generate",
                f"{self.base_url}/v1/chat/completions",
                f"{self.base_url}/chat/completions",
                f"{self.base_url}/api/chat",
            ]

        for url in target_urls:
            try:
                req = urllib.request.Request(url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    if r.status != 200:
                        continue
                    data = json.loads(r.read().decode("utf-8", errors="ignore"))

                    # Parse OpenAI, Ollama, or native KALI format
                    raw_response = ""
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        if isinstance(choice, dict):
                            msg = choice.get("message")
                            if isinstance(msg, dict):
                                raw_response = msg.get("content", "")
                            else:
                                raw_response = choice.get("text", "")
                    elif "response" in data:
                        raw_response = data.get("response", "")
                    elif "message" in data and isinstance(data["message"], dict):
                        raw_response = data["message"].get("content", "")
                    elif "text" in data:
                        raw_response = data.get("text", "")

                    cleaned = self.clean_output(raw_response)
                    if cleaned:
                        return cleaned
            except Exception as exc:
                logger.debug("API invocation to %s failed: %s", url, exc)
                continue

        logger.info("KALI Model API invocation failed; falling back to local Astra-6 knowledge base")
        return None

    @staticmethod
    def clean_output(raw: str) -> str:
        """Format and clean output from KALI model, removing or formatting thinking tags."""
        if not raw:
            return ""
        text = raw.strip()
        # If model generated a reasoning block followed by answer:
        if "</think>" in text:
            parts = text.split("</think>", 1)
            ans = parts[1].strip()
            if ans:
                return ans
            # If nothing after </think>, take what was inside
            inside = parts[0].replace("<think>", "").strip()
            if inside:
                return inside

        # If generation stopped within <think>, strip the opening tag so thought is readable
        if text.startswith("<think>"):
            text = text[len("<think>"):].strip()

        return text


def get_kali_client(base_url: Optional[str] = None) -> KaliModelClient:
    """Retrieve or create singleton KALI Model Client."""
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is None or (base_url and base_url != _CLIENT_INSTANCE.base_url):
        _CLIENT_INSTANCE = KaliModelClient(base_url=base_url)
    return _CLIENT_INSTANCE
