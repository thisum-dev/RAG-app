"""
rag/generator.py - LLM response generation via Groq API.

Responsibility:
  - Build the structured RAG prompt (system + context + history + question).
  - Send the prompt to the Groq LLaMA 3 model.
  - Return the model's text response.
"""

import logging
import os
from groq import Groq
from dotenv import load_dotenv
from config import LLM_MODEL_NAME

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# ─── Initialise the Groq client once ─────────────────────────────────────────
# The Groq() constructor automatically reads GROQ_API_KEY from the environment.
_client: Groq | None = None


def get_groq_client() -> Groq:
    """Return a singleton Groq client, creating it the first time."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. Please set it in your .env file."
            )
        _client = Groq(api_key=api_key)
        logger.info("Groq client initialised.")
    return _client


def build_prompt(
    question: str,
    context_chunks: list[str],
    chat_history: list[dict],
) -> list[dict]:
    """
    Assemble the message list that will be sent to the Groq chat completion API.

    The final message list follows the OpenAI / Groq chat format:
      [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": "..."},   # from history
        {"role": "assistant", "content": "..."},  # from history
        ...
        {"role": "user",   "content": "question + context"},
      ]

    Args:
        question:       Current user question.
        context_chunks: Retrieved document chunks to form the RAG context.
        chat_history:   Previous turns as list of {"role": ..., "content": ...}.

    Returns:
        List of message dicts ready for client.chat.completions.create().
    """
    # System instruction: strict grounding to context
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Answer strictly using the provided context. "
            "If the answer is not in the context, say 'I don't know based on the provided documents.' "
            "Be concise and precise."
        ),
    }

    # Format the retrieved chunks as readable context
    context_text = "\n\n---\n\n".join(context_chunks)

    # Final user turn: context + question together
    user_message = {
        "role": "user",
        "content": (
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        ),
    }

    # Combine: system + prior history + current question
    messages = [system_message] + chat_history + [user_message]
    return messages


def generate_answer(
    question: str,
    context_chunks: list[str],
    chat_history: list[dict],
) -> str:
    """
    Generate an answer using Groq LLaMA 3 given the question, context, and history.

    Args:
        question:       The user's current question.
        context_chunks: Relevant document chunks retrieved from FAISS.
        chat_history:   Previous conversation turns.

    Returns:
        The model's answer as a string.
    """
    client = get_groq_client()
    messages = build_prompt(question, context_chunks, chat_history)

    logger.info(f"Calling Groq model: {LLM_MODEL_NAME}")
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=messages,
        temperature=0.2,       # Low temperature = more factual, less creative
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()
    logger.info("Answer received from Groq.")
    return answer
