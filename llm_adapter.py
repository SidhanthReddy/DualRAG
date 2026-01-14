from typing import Optional
import os


class LLMAdapter:
    """
    Minimal LLM transport layer.
    Stateless. No parsing. No validation.
    """

    def __init__(self, provider: str = "mock"):
        """
        provider:
          - "mock"     → deterministic fake output (default)
          - "openai"   → OpenAI Chat API (optional)
        """
        self.provider = provider

        if provider == "openai":
            try:
                import openai
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. Run: pip install openai"
                )

            self.openai = openai
            self.openai.api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai.api_key:
                raise RuntimeError("OPENAI_API_KEY not set")

    def generate(self, prompt: str) -> str:
        """
        Sends prompt to LLM and returns raw text output.
        """
        if self.provider == "mock":
            return self._mock_response(prompt)

        if self.provider == "openai":
            return self._openai_response(prompt)

        raise ValueError(f"Unknown provider: {self.provider}")

    # -----------------------------
    # Providers
    # -----------------------------

    def _mock_response(self, prompt: str) -> str:
        """
        Deterministic mock output for testing.
        """
        return (
            "FILE: components/Navbar.tsx\n"
            "<nav class='h-20 bg-blue-600 text-white'>\n"
            "  Mock LLM Output\n"
            "</nav>"
        )

    def _openai_response(self, prompt: str) -> str:
        """
        OpenAI ChatCompletion (single-shot).
        """
        response = self.openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a stateless code generator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        return response.choices[0].message["content"]
