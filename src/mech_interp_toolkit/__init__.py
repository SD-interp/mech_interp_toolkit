"""
Mech-Interp-Toolkit: A Comprehensive Library for Mechanistic Interpretability

This library provides a suite of tools for researchers and developers to dissect and understand the internal workings of large language models (LLMs).
"""

from importlib.metadata import version

__version__ = version("mech-interp-toolkit")
__author__ = "Shantanu Darveshi"
__license__ = "MIT"


from . import (
    direct_logit_attribution,
    gradient_based_attribution,
    linear_probes,
    misc,
    tokenizer,
    utils,
)

__all__ = [
    "tokenizer",
    "utils",
    "direct_logit_attribution",
    "gradient_based_attribution",
    "linear_probes",
    "misc",
]
