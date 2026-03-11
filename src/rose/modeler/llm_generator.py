"""
LLM-powered YAML model generator for ROSE.

Takes a plain-text description and uses a language model to produce
a valid ROSE YAML model file.  The generated YAML is validated
before being returned.

Requires the ``llm`` optional dependency group::

    pip install rose[llm]

Environment variable ``OPENAI_API_KEY`` must be set (or whichever
key the chosen model provider requires).
"""

from __future__ import annotations

import logging

from rose.modeler.prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def generate_model_yaml(
    description: str,
    model_name: str = "gpt-4o",
    temperature: float = 0.2,
    max_retries: int = 2,
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 4096,
) -> str:
    """Generate a ROSE YAML model from a plain-text description using an LLM.

    Args:
        description: Plain-text sample description and hypothesis.
        model_name: LLM model identifier (default ``"gpt-4o"``).
        temperature: Sampling temperature (default 0.2 for determinism).
        max_retries: Number of retry attempts on validation failure.
        api_key: Optional API key override (default reads from env).
        base_url: Optional base URL for OpenAI-compatible endpoints
            (ALCF, Ollama, vLLM, etc.).
        max_tokens: Maximum output tokens per LLM call (default 4096).

    Returns:
        YAML string of the generated model.

    Raises:
        ImportError: If ``langchain-openai`` is not installed.
        ValueError: If generation fails after all retries.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "LLM features require the 'llm' extras. Install with: pip install rose[llm]"
        ) from None

    from rose.modeler.validator import validate_model_yaml

    kwargs: dict = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    llm = ChatOpenAI(**kwargs)

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(description)

    logger.debug("System prompt length: %d chars", len(system_prompt))
    logger.debug("User prompt:\n%s", user_prompt)

    last_error: str = ""
    for attempt in range(1, max_retries + 2):  # +2 because range is exclusive
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if last_error:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The previous YAML was invalid:\n{last_error}\n\n"
                        "Please fix the YAML and output a corrected version. "
                        "Output only valid YAML, nothing else."
                    ),
                }
            )

        logger.info("LLM generation attempt %d/%d", attempt, max_retries + 1)
        response = llm.invoke(messages)
        raw = response.content

        # Strip markdown fences if the LLM wraps them
        yaml_text = _strip_markdown_fences(raw)

        errors = validate_model_yaml(yaml_text)
        if not errors:
            logger.info("Generated valid YAML on attempt %d", attempt)
            return yaml_text

        last_error = "; ".join(errors)
        logger.warning("Attempt %d validation errors: %s", attempt, last_error)

    raise ValueError(
        f"LLM failed to produce valid YAML after {max_retries + 1} attempts. "
        f"Last errors: {last_error}"
    )


def _strip_markdown_fences(text: str) -> str:
    """Remove ```yaml ... ``` wrapping if present."""
    text = text.strip()
    if text.startswith("```"):
        # Remove first line (```yaml or ```)
        lines = text.split("\n")
        lines = lines[1:]
        # Remove last ``` line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
