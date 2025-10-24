"""Abstract base class for summarization methods"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import threading
from together import Together


class BaseSummarizer(ABC):
    """Abstract base class for all summarization methods"""

    def __init__(self, model: str = "Qwen/Qwen3-235B-A22B-fp8-tput", **kwargs):
        """
        Initialize the summarizer.

        Args:
            model: The model identifier to use for summarization
            **kwargs: Additional configuration parameters
        """
        self.model = model
        self.config = kwargs
        self.thread_local = threading.local()

    def get_client(self) -> Together:
        """
        Get or create a Together client for the current thread.

        Thread-safe client management for parallel processing.

        Returns:
            Together API client instance
        """
        if not hasattr(self.thread_local, 'client'):
            self.thread_local.client = Together()
        return self.thread_local.client

    @abstractmethod
    def summarize(self, document: str, prompt: str) -> str:
        """
        Generate a summary for the given document and prompt.

        Args:
            document: The full document text
            prompt: The summarization instruction/prompt

        Returns:
            The generated summary
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Return a unique identifier for this summarization method.

        This name is used for output files and logging.

        Returns:
            Unique method name
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model}, config={self.config})"
