# 🌟 tmam‑python‑sdk

[![PyPI version](https://img.shields.io/pypi/v/tmam-python-sdk.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

**tmam‑python‑sdk** is an OpenTelemetry-native auto-instrumentation library for GenAI applications—providing seamless observability for Large Language Models (LLMs), GPU workloads, vector databases, and agent-based frameworks. With one line of integration, you can add powerful telemetry to your AI/ML stack.

---

## 🚀 Key Features

- ✅ **Zero-code auto instrumentation** for LLMs, GPUs, vector DBs, and agents
- 📊 **Traces, metrics, and logs** for LLM prompts, latency, token usage, GPU utilization, and more
- 🔌 **Simple integration** using a hosted TMAM collector endpoint
- 📦 **OpenTelemetry-native**, vendor-agnostic support for downstream observability tools
- 🔍 **Supports popular frameworks** like OpenAI, Hugging Face, LangChain, PyTorch, NVIDIA CUDA, and more

---

## 🧪 Installation

```bash
pip install tmam



Quickstart

Add observability to your GenAI application with one line:

import tmam

tmam.init(
    url="http://api.tmam.ai/api/sdk/v1",
    public_key="pk-tmam-0edeba2a-f6f3-4efd-982c-412adbb03046",
    secret_key="sk-tmam-b320dda9-e36d-4eac-8ac5-4793fd38e002",
)

# Your LLM or agent code here
from openai import OpenAI
# ...
Once initialized, tmam will auto-instrument supported components and begin sending traces and metrics to TMAM’s backend.
