"""Utility functions for tmam.guard"""

from pydantic import BaseModel


class JsonOutput(BaseModel):
    """
    A model representing the structure of JSON output for prompt injection detection.

    Attributes:
        score (float): The score of the harmful prompt likelihood.
        verdict (str): Verdict if detection is harmful or not.
        guard (str): The type of guardrail.
        classification (str): The classification of prompt detected.
        explanation (str): A detailed explanation of the detection.
    """

    score: float
    verdict: str
    guard: str
    classification: str
    explanation: str
