from __future__ import annotations
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import time
from colorama import Fore
from autogpt.config import Config
from abc import ABCMeta, abstractmethod
from typing import Any, Literal, Optional
import json
import os
from importlib.resources import files
import functools

from autogpt.models.command_registry import CommandRegistry
from google import genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from openai import OpenAI

from autogpt.logs import logger

def retry(max_attempts=3, backoff_base=1.5, exceptions_to_catch=(ResourceExhausted, ServiceUnavailable)):
    """
    A decorator to retry a function if an exception occurs.

    :param max_attempts: Maximum number of times to attempt the function.
    :param backoff_base: Factor by which the delay increases each time (e.g., 2 for exponential).
    :param exceptions_to_catch: A tuple of exception types to catch and retry on.
                                Defaults to all exceptions.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            backoff_msg = f"{Fore.RED}Rate Limit Reached. Waiting {{backoff}} seconds...{Fore.RESET}"
            error_msg = f"{Fore.RED}Unknown Error: {{err}}. Waiting {{backoff}} seconds...{Fore.RESET}"
            for attempt in range(1, max_attempts + 1):
                backoff = round(backoff_base ** (attempt), 2)
                try:
                    return func(*args, **kwargs)
                except exceptions_to_catch as e:
                    logger.warn(backoff_msg.format(backoff=backoff))
                    if attempt >= max_attempts:
                        raise
                except Exception as e:  # Catch-all for other potential error
                    logger.warn(error_msg.format(err=e, backoff=backoff))
                    if attempt >= max_attempts:
                        raise
                time.sleep(backoff)
        return wrapper
    return decorator


def ask_chatgpt(prompt):
    """
    Asks a question to either OpenAI's ChatGPT or Google's Gemini models.

    Args:
        query (str): The question to ask the model.
        system_message (str): The system message to guide the model's response.
        model (str, optional): The model to use.

    Returns:
        str: The content of the assistant's response.
    """
    # Set up the OpenAI API key
    api_key = os.getenv("API_KEY", default="")
    # Update base url for different API providers
    base_url = os.getenv("BASE_URL", default="")
    llm_model = os.getenv("LLM_MODEL", default="")
    if "google" in base_url: # Gemini version
        client = genai.Client(api_key=api_key)
    else:
        client = OpenAI(api_key=api_key)
    return create_chat_completion(client=client, model=llm_model, prompt=prompt)

# Overly simple abstraction until we create something better
def create_chat_completion(
    client,
    prompt,
    model,
) -> str:
    if type(client) is genai.Client:
        return create_chat_completion_gemini(client, model, prompt)
    elif type(client) is OpenAI:
        return create_chat_completion_gpt(client, model, prompt)
    return "ERROR: Client not supported."


@retry()
def create_chat_completion_gemini(
    client: genai.Client,
    model,
    prompt
) -> str:
    """Create a chat completion with Gemini."""
    response = client.models.generate_content(
        model=model, contents=prompt
    )
    return response.text

@retry()
def create_chat_completion_gpt(
    client: OpenAI,
    model,
    prompt,
) -> str:
    """Create a chat completion with GPT."""
    response = client.chat.completions.create(
        model=model, messages=[
        {
        "role": "user",
        "content": prompt
        },
        ],
    )
    return response.choices[0].message.content
    