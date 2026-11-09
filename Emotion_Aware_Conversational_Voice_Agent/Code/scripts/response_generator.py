"""
response_generator.py
Turns (tone mode, transcript) into a short spoken reply.
- build_system_prompt(): MoodManager mode -> tone; memory facts -> content (Task B)
- generate_reply_template(): deterministic offline reply (demo fallback + paper comparison)
- _call_llm(): the ONLY function that talks to Gemini (google-genai); tries GEMINI_MODELS in order
- generate_reply(): LLM first, template on any failure. Never raises.
Key: GEMINI_API_KEY from Code/.env (see .env.example).
Models: GEMINI_MODEL (else DEFAULT_MODEL), then GEMINI_FALLBACK_MODELS (else DEFAULT_FALLBACKS).
"""
from pathlib import Path
import os
import time

from dotenv import load_dotenv

# Config
CODE_DIR = Path(__file__).resolve().parent.parent   # .../Code
load_dotenv(CODE_DIR / ".env")

USE_LLM = True
DEFAULT_MODEL = "gemini-3.6-flash"   # stable, free tier (ai.google.dev/gemini-api/docs/models)
# gemini-3.8-flash dropped 2026-10-31 for persistent 504s; re-enable via GEMINI_MODEL in .env
# Stable, free-tier, open to new users; all answered 3/3 in ~0.6-1.2 s when 3.8-flash timed out
DEFAULT_FALLBACKS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
SERVER_MIN_DEADLINE_S = 10.0   # Gemini rejects request deadlines under 10 s (400 INVALID_ARGUMENT)
PER_MODEL_TIMEOUT_S = 10.0     # healthy calls take ~0.6-1.2 s; an overloaded model gives up here instead of at ~29 s
TOTAL_BUDGET_S = 20.0          # whole turn: at most two full timeouts; fast errors (503/429) leave room for more models
MAX_HISTORY_TURNS = 4

# Lowest thinking level each model accepts (ai.google.dev/gemini-api/docs/thinking).
# Models not listed get no thinking_config (model default).
MIN_THINKING = {
    "gemini-3.8-flash": "LOW",          # no MINIMAL on this model
    "gemini-3.6-flash": "MINIMAL",
    "gemini-3.5-flash-lite": "MINIMAL",
    "gemini-3.1-flash-lite": "MINIMAL",
}
RETRYABLE_CODES = {404, 429, 500, 502, 503, 504}   # 404 = model name not found -> try the next model


def _env_list(name):
    return [m.strip() for m in os.environ.get(name, "").split(",") if m.strip()]


primary = os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
fallbacks = _env_list("GEMINI_FALLBACK_MODELS") or DEFAULT_FALLBACKS
GEMINI_MODELS = list(dict.fromkeys([primary, *fallbacks]))   # ordered, no duplicates

TONE_PROFILES = {
    "DEFAULT": (
        "Be warm, natural and friendly, like a close friend. "
        "Keep it concise and conversational."
    ),
    "EMPATHETIC": (
        "Acknowledge and validate their feeling first. Be soft and gentle. "
        "No jokes. Don't rush into advice. Ask at most one gentle question."
    ),
}


def build_system_prompt(mode, perceived_emotion=None, memory_facts=None) -> str:
    tone = TONE_PROFILES.get(mode, TONE_PROFILES["DEFAULT"])
    parts = [
        "You are a supportive voice companion having a spoken conversation. "
        "You are not a therapist and do not diagnose or give clinical advice.",
        f"Tone ({mode}): {tone}",
    ]
    if perceived_emotion:
        parts.append(
            f"Context: from their voice, they seem {perceived_emotion}. "
            "Let this shape how you respond, but don't announce it or say you detected it."
        )
    if memory_facts:
        facts = "\n".join(f"- {f}" for f in memory_facts)
        parts.append(f"Things you remember about them (use only if relevant):\n{facts}")
    parts.append(
        "Output rules: reply in 1 to 3 short sentences meant to be spoken aloud. "
        "Plain speech only: no markdown, no lists, no emoji, no stage directions."
    )
    parts.append(
        "Safety: if they mention self-harm or being in danger, respond with care and "
        "encourage them to reach out to someone they trust or a local crisis line."
    )
    return "\n\n".join(parts)


TEMPLATES = {
    "DEFAULT": "Thanks for telling me. How are you feeling about it?",
    "EMPATHETIC": "That sounds really hard, and it makes sense to feel this way. I'm here with you.",
}


def generate_reply_template(mode, transcript) -> str:
    """Deterministic offline reply. The transcript is accepted for interface parity but not parsed."""
    return TEMPLATES.get(mode, TEMPLATES["DEFAULT"])


class _NextModel(Exception):
    """This model failed in a way another model might not (overload, timeout, ...)."""
    def __init__(self, outcome, detail=""):
        super().__init__(f"{outcome} {detail}".strip())
        self.outcome = outcome   # short label for the attempts log, e.g. "429", "timeout"


def _call_llm(system_prompt, user_message, history=None, api_key=None, models=None, attempts=None):
    """Try each model in order. Returns (text, model). Raises if every model fails or a fatal error occurs.
    If `attempts` is a list, one {model, outcome, seconds} entry is appended per model call."""
    attempts = [] if attempts is None else attempts
    import httpx
    from google import genai
    from google.genai import types, errors

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")

    # Pass the key explicitly: the SDK otherwise prefers GOOGLE_API_KEY if that is also set.
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
    )

    contents = []
    for user_text, agent_text in (history or [])[-MAX_HISTORY_TURNS:]:
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))
        contents.append(types.Content(role="model", parts=[types.Part.from_text(text=agent_text)]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

    def attempt(model, timeout_s, thinking):
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),   # milliseconds
            thinking_config=types.ThinkingConfig(thinking_level=thinking) if thinking else None,
        )
        t0 = time.monotonic()

        def log(outcome):
            attempts.append({"model": model, "outcome": outcome, "seconds": round(time.monotonic() - t0, 2)})

        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as e:
            log(str(e.code))
            if e.code in RETRYABLE_CODES:
                raise _NextModel(str(e.code), e.status) from e
            raise                                    # 400 bad key / 403 / bad request: fatal
        except httpx.TimeoutException as e:
            log("timeout")
            raise _NextModel("timeout", type(e).__name__) from e
        except httpx.TransportError as e:
            log("connection")
            raise _NextModel("connection", type(e).__name__) from e
        text = (response.text or "").strip()
        if not text:
            log("empty")
            raise _NextModel("empty")
        log("ok")
        return text

    deadline = time.monotonic() + TOTAL_BUDGET_S
    failures = []
    def timeout_left():
        return min(PER_MODEL_TIMEOUT_S, deadline - time.monotonic())

    for model in models or GEMINI_MODELS:
        timeout_s = timeout_left()
        if timeout_s < SERVER_MIN_DEADLINE_S:
            failures.append(f"{model}: skipped (budget)")
            attempts.append({"model": model, "outcome": "skipped (budget)", "seconds": 0.0})
            break
        thinking = MIN_THINKING.get(model)
        try:
            try:
                return attempt(model, timeout_s, thinking), model
            except errors.ClientError as e:
                # Model rejected the thinking setting: retry this model once without it.
                if not (thinking and e.code == 400 and "thinking" in str(e.message).lower()):
                    raise
                timeout_s = timeout_left()
                if timeout_s < SERVER_MIN_DEADLINE_S:
                    raise _NextModel("thinking rejected, no budget left")
                return attempt(model, timeout_s, None), model
        except _NextModel as e:
            failures.append(f"{model}: {e}")
    raise RuntimeError("all models failed: " + "; ".join(failures))


def generate_reply(mode, transcript, perceived_emotion=None, memory_facts=None,
                   history=None, use_llm=USE_LLM, api_key=None) -> dict:
    """Returns {"text", "source": "llm"|"template", "model": str|None, "error": None|str,
    "attempts": [{model, outcome, seconds}, ...]}. Never raises."""
    error = None
    attempts = []
    if use_llm:
        try:
            prompt = build_system_prompt(mode, perceived_emotion, memory_facts)
            text, model = _call_llm(prompt, transcript, history, api_key, attempts=attempts)
            return {"text": text, "source": "llm", "model": model, "error": None, "attempts": attempts}
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:300]}"
    try:
        text = generate_reply_template(mode, transcript)
    except Exception as e:
        text, error = TEMPLATES["DEFAULT"], f"template failed: {type(e).__name__}"
    return {"text": text, "source": "template", "model": None, "error": error, "attempts": attempts}
