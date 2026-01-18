from typing import Optional
import os
from dotenv import load_dotenv


class LLMAdapter:
    """
    Minimal LLM transport layer.
    Stateless. No parsing. No validation.
    """

    def __init__(self, provider: str = "mock"):
        """
        provider:
          - "mock"
          - "openai"
          - "gemini"
        """
        self.provider = provider.lower()

        # Load .env if present
        load_dotenv()

        if self.provider == "openai":
            self._init_openai()

        elif self.provider == "gemini":
            self._init_gemini()

        elif self.provider == "mock":
            pass

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Sends prompt to LLM and returns raw text output.
        """
        if self.provider == "mock":
            return self._mock_response(prompt)

        if self.provider == "openai":
            return self._openai_response(prompt)

        if self.provider == "gemini":
            return self._gemini_response(prompt)

        raise RuntimeError("Invalid LLM provider state")

    # --------------------------------------------------
    # Provider initializers
    # --------------------------------------------------

    def _init_openai(self):
        try:
            import openai
        except ImportError:
            raise RuntimeError("Run: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        self.openai = openai
        self.openai.api_key = api_key

    def _init_gemini(self):
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError("Run: pip install google-generativeai")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        genai.configure(api_key=api_key)
        self.genai = genai
        self.gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

    # --------------------------------------------------
    # Providers
    # --------------------------------------------------

    def _mock_response(self, prompt: str) -> str:
        return (
            "FILE: components/Navbar.tsx\n"
            "<nav class='h-20 bg-blue-600 text-white'>\n"
            "  Mock LLM Output\n"
            "</nav>"
        )

    def _openai_response(self, prompt: str) -> str:
        response = self.openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a stateless code generator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message["content"]

    def _gemini_response(self, prompt: str) -> str:
        response = self.gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": 4096,
            },
        )
        return response.text
