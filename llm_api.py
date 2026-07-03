"""LLM client. Uses an optional cloud endpoint only when configured."""
import requests
import os
from config import (
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
)


LLM_MAX_TOKENS = int(os.getenv("LOCAL_RAG_LLM_MAX_TOKENS", "512"))


def _use_deepseek():
    return bool(DEEPSEEK_API_KEY)


def chat(prompt: str, system_prompt: str = "", temperature: float = 0.3) -> dict:
    """Unified LLM chat interface."""
    if _use_deepseek():
        return _call_deepseek(prompt, system_prompt, temperature)
    return _call_lmstudio(prompt, system_prompt, temperature)


def _call_lmstudio(prompt: str, system_prompt: str, temperature: float) -> dict:
    """Call the local LM Studio OpenAI-compatible endpoint."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # Qwen3 may put content into reasoning_content. /no_think keeps responses concise.
    if not prompt.lstrip().startswith("/no_think"):
        prompt = "/no_think\n" + prompt
    messages.append({"role": "user", "content": prompt})

    try:
        r = requests.post(
            f"{LMSTUDIO_BASE_URL}/v1/chat/completions",
            json={
                "model": LMSTUDIO_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": LLM_MAX_TOKENS,
            },
            timeout=180,
        )
        if r.ok:
            message = r.json()["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content") or ""
            return {"ok": True, "content": content}
        return {"ok": False, "error": f"LM Studio returned HTTP {r.status_code}"}
    except requests.ConnectionError:
        return {"ok": False, "error": "Cannot connect to LM Studio. Confirm it is running and models are loaded."}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _call_deepseek(prompt: str, system_prompt: str, temperature: float) -> dict:
    """Call the optional OpenAI-compatible cloud endpoint."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        r = requests.post(
            f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": LLM_MAX_TOKENS,
            },
            timeout=120,
        )
        if r.ok:
            content = r.json()["choices"][0]["message"]["content"]
            return {"ok": True, "content": content}
        return {"ok": False, "error": f"Cloud API returned HTTP {r.status_code}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
