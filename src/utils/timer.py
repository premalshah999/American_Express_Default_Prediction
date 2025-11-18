"""Timer utility for measuring execution time."""

import time
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing code execution."""

    def __init__(self, name: Optional[str] = None, logger_func=None):
        """
        Initialize timer.

        Args:
            name: Name of the operation being timed
            logger_func: Logging function to use (default: logger.info)
        """
        self.name = name or "Operation"
        self.logger_func = logger_func or logger.info
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start the timer."""
        self.start_time = time.time()
        self.logger_func(f"{self.name} started...")
        return self

    def __exit__(self, *args):
        """Stop the timer and log the elapsed time."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time

        if elapsed < 60:
            time_str = f"{elapsed:.2f} seconds"
        elif elapsed < 3600:
            time_str = f"{elapsed/60:.2f} minutes"
        else:
            time_str = f"{elapsed/3600:.2f} hours"

        self.logger_func(f"{self.name} completed in {time_str}")

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
