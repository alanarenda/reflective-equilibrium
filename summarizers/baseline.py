"""Baseline summarization methods"""

from .base import BaseSummarizer
from .utils import extract_summary


class BaselineSummarizer(BaseSummarizer):
    """Standard summarization with optional thinking"""

    def __init__(self, enable_thinking: bool = False, **kwargs):
        """
        Initialize baseline summarizer.

        Args:
            enable_thinking: Whether to enable model thinking mode
            **kwargs: Additional configuration passed to parent
        """
        super().__init__(**kwargs)
        self.enable_thinking = enable_thinking

    def summarize(self, document: str, prompt: str) -> str:
        """
        Generate a summary using baseline method.

        Args:
            document: The full document text
            prompt: The summarization instruction/prompt

        Returns:
            The generated summary
        """
        full_prompt = f"{prompt}\n\nDocument: {document}"

        client = self.get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": full_prompt,
                "enable_thinking": self.enable_thinking
            }]
        )

        content = response.choices[0].message.content
        return extract_summary(content) if self.enable_thinking else content

    def name(self) -> str:
        """Return unique identifier for this method"""
        thinking_str = "thinking" if self.enable_thinking else "no_thinking"
        return f"baseline_{thinking_str}"


class ThinkingSummarizer(BaselineSummarizer):
    """Convenience class for baseline with thinking enabled"""

    def __init__(self, **kwargs):
        """Initialize with thinking enabled by default"""
        super().__init__(enable_thinking=True, **kwargs)
