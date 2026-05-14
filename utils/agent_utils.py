"""
utils/agent_utils.py — Shared agent utilities (v3)

Improvements in v3:
  #2  _agent_error flag check helper for pipeline routing
  #4  LLM retry with exponential backoff (no external dep — pure stdlib)
  #12 call_llm_json / make_llm unchanged but now retried automatically
"""
from __future__ import annotations

import functools
import time
import traceback
from typing import Callable, Any
from utils.serialization import sanitize_for_msgpack


# ─────────────────────────────────────────────────────────────────────────────
# Fix #4: Exponential backoff (no tenacity dependency — pure stdlib)
# ─────────────────────────────────────────────────────────────────────────────

def _with_backoff(fn, *args, max_attempts: int = 3, base_delay: float = 1.0, **kwargs):
    """
    Call fn(*args, **kwargs) up to max_attempts times with exponential backoff.
    Retries on any exception. Returns (result, None) on success or (None, last_exc).

    Delays: 1s, 2s, 4s, … (base_delay * 2^attempt)
    Only retryable transient errors (rate-limits, network) benefit from this;
    logic errors will still fail on all attempts but we still return gracefully.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs), None
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
    return None, last_exc


# ─────────────────────────────────────────────────────────────────────────────
# Fix #4 / #9: Central error handling — agents never crash the pipeline
# ─────────────────────────────────────────────────────────────────────────────

def agent_error_handler(agent_name: str) -> Callable:
    """
    Decorator factory. Wraps an agent function so that any unhandled exception
    is caught and returned as a structured error message instead of propagating
    up and breaking the LangGraph pipeline.

    On failure the graph state receives:
      {
        "agent_messages": ["[EDA Agent] ❌ Unexpected error: <details>"],
        "_agent_error":   True,
        "_agent_error_source": "EDA Agent",
      }
    This allows the orchestrator and UI to detect failures and surface them
    without crashing the entire session.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: Any, *args, **kwargs) -> dict:
            try:
                result = fn(state, *args, **kwargs)
                return sanitize_for_msgpack(result) if isinstance(result, dict) else result
            except Exception as exc:
                tb = traceback.format_exc()
                msg = (
                    f"[{agent_name}] ❌ Unexpected error: {exc}\n"
                    f"Traceback (most recent call last):\n{tb}"
                )
                import sys
                print(msg, file=sys.stderr)
                return sanitize_for_msgpack({
                    "agent_messages":       [f"[{agent_name}] ❌ Unexpected error: {exc}"],
                    "_agent_error":         True,
                    "_agent_error_source":  agent_name,
                })
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Fix #2: Helper to check if a state has a pending agent error
# ─────────────────────────────────────────────────────────────────────────────

def has_agent_error(state: dict) -> bool:
    """
    Return True if any agent in this run set _agent_error on the state.
    Used by pipeline.py conditional edges to route to an error terminal node
    instead of proceeding with incomplete/corrupt state.
    """
    return bool(state.get("_agent_error"))


# ─────────────────────────────────────────────────────────────────────────────
# Fix #14: Trim large prompt payloads to reduce LLM cost + latency
# ─────────────────────────────────────────────────────────────────────────────

def truncate_prompt_metadata(
    col_stats: list[dict],
    max_cols: int = 40,
    max_top_values: int = 5,
) -> list[dict]:
    """
    Trim column-stats lists before serialising into LLM prompts.

    - Caps the number of columns at *max_cols*.
    - Truncates 'top_values' lists to *max_top_values* entries.
    - Removes redundant / low-value keys.

    Returns the trimmed list (original is not mutated).
    """
    trimmed = []
    for stat in col_stats[:max_cols]:
        s = {k: v for k, v in stat.items()
             if k not in ("_raw_values", "_histogram")}
        if "top_values" in s and isinstance(s["top_values"], list):
            s["top_values"] = s["top_values"][:max_top_values]
        trimmed.append(s)
    return trimmed


# ─────────────────────────────────────────────────────────────────────────────
# Fix #12 + #4: Shared LLM invocation helper with automatic retry
# ─────────────────────────────────────────────────────────────────────────────

def call_llm_json(
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    request_timeout: float | None = None,
) -> tuple[dict | None, Exception | None]:
    """
    Centralised LangChain JSON-output LLM call with automatic exponential backoff.

    Retries up to max_attempts times on transient errors (rate limits, timeouts).
    Delays between retries: base_delay * 2^attempt  (1s, 2s, 4s by default).

    Returns
    -------
    (result_dict, None)   on success
    (None, exception)     on final failure — caller applies its own fallback
    """
    def _invoke():
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = make_llm(model_name=model_name, api_key=api_key,
                       temperature=temperature, max_tokens=max_tokens,
                       request_timeout=request_timeout)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        return (prompt | llm | JsonOutputParser()).invoke({})

    return _with_backoff(_invoke, max_attempts=max_attempts, base_delay=base_delay)


# ─────────────────────────────────────────────────────────────────────────────
# LLM provider factory — swap providers via AUTOML_LLM_PROVIDER env var
# ─────────────────────────────────────────────────────────────────────────────

def make_llm(
    model_name: str,
    api_key: str | None,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    request_timeout: float | None = None,
):
    """
    Return a LangChain chat model for the configured provider.

    Provider is selected via the AUTOML_LLM_PROVIDER env var:
      openai    (default) — requires OPENAI_API_KEY
      anthropic           — requires ANTHROPIC_API_KEY
      google              — requires GOOGLE_API_KEY
    """
    import os
    provider = os.getenv("AUTOML_LLM_PROVIDER", "openai").lower()

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model_name,
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=request_timeout,
            )
        except ImportError as e:
            raise ImportError(
                "langchain-anthropic is required for AUTOML_LLM_PROVIDER=anthropic. "
                "Install it with: pip install langchain-anthropic"
            ) from e

    elif provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
                temperature=temperature,
                max_output_tokens=max_tokens,
                timeout=request_timeout,
            )
        except ImportError as e:
            raise ImportError(
                "langchain-google-genai is required for AUTOML_LLM_PROVIDER=google. "
                "Install it with: pip install langchain-google-genai"
            ) from e

    else:  # default: openai
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=request_timeout,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fix #10: Streaming LLM call — yields text chunks for live UI display
# ─────────────────────────────────────────────────────────────────────────────

def stream_llm_text(
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: int = 1000,
):
    """
    Generator that streams LLM text tokens one chunk at a time.

    Usage in Streamlit:
        import streamlit as st
        with st.empty() as placeholder:
            text = ""
            for chunk in stream_llm_text(...):
                text += chunk
                placeholder.markdown(text + "▌")
            placeholder.markdown(text)

    Falls back to a single non-streamed call if the provider/model
    does not support streaming — the generator still yields a single item.
    """
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = make_llm(model_name=model_name, api_key=api_key,
                       temperature=temperature, max_tokens=max_tokens)
        msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        for chunk in llm.stream(msgs):
            text = getattr(chunk, "content", "") or ""
            if text:
                yield text
    except Exception as exc:
        # Graceful fallback: try a single blocking call
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            llm = make_llm(model_name=model_name, api_key=api_key,
                           temperature=temperature, max_tokens=max_tokens)
            msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
            result = llm.invoke(msgs)
            yield getattr(result, "content", str(result))
        except Exception:
            yield f"[Report generation failed: {exc}]"
