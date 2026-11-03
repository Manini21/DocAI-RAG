"""
LLM module.

Handles communication with the OpenAI chat completion API.
"""

from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE

_client = OpenAI(api_key=OPENAI_API_KEY)


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    """
    Generate an answer using OpenAI.
    """
    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

    except Exception as error:
        raise RuntimeError(
            f"Failed to generate answer from LLM: {error}"
        ) from error

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError("The LLM returned an empty response.")

    return answer.strip()