"""Gemini API client wrapper for clinical report generation.

Provides a clean interface to the Google Generative AI API
with proper error handling and configuration.
"""

import os
from typing import Any

from google import genai


class GeminiClient:
    """Client wrapper for Gemini API interactions.

    Handles API key configuration, model selection, and basic
    request/response flow for clinical report generation.
    """

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize Gemini client.

        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model_name: Model to use. Defaults to gemini-2.0-flash.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key parameter."
            )

        self.model_name = model_name or self.DEFAULT_MODEL

        # Initialize the client with API key
        self._client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to send to Gemini.
            **kwargs: Additional generation config options.

        Returns:
            Generated text response.
        """
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            **kwargs,
        )
        return response.text
