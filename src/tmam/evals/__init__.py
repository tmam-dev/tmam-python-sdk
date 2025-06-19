"""
tmam.evals

This module provides a set of classes for analyzing text for various types of
content-based vulnerabilities,
such as Hallucination, Bias, and Toxicity detection.
"""

from tmam.evals.hallucination import Hallucination
from tmam.evals.bias_detection import BiasDetector
from tmam.evals.toxicity import ToxicityDetector
from tmam.evals.all import All
