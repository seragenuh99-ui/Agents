# Multi-Agent Collaboration System
# Target: openEuler 24.03 LTS-SP3
#
# Quick start (offline, no network needed at runtime):
#   docker build -t multi-agent-system .
#   docker run --rm multi-agent-system demo
#
# Full comparison experiment (mock LLM, no API key needed):
#   docker run --rm multi-agent-system experiment --mock
#
# With real LLM API (DeepSeek):
#   docker run --rm -e DEEPSEEK_API_KEY=sk-xxx multi-agent-system experiment --provider deepseek
#
# With OpenAI:
#   docker run --rm -e OPENAI_API_KEY=sk-xxx multi-agent-system experiment --provider openai
#
# With custom OpenAI-compatible API:
#   docker run --rm -e OPENAI_API_KEY=sk-xxx -e OPENAI_BASE_URL=https://your-api.com/v1 multi-agent-system experiment
#
# With real embedding model (requires network to download from HuggingFace):
#   docker run --rm multi-agent-system demo   # auto-downloads all-MiniLM-L6-v2 on first run

FROM openeuler/openeuler:24.03

LABEL description="Multi-Agent Collaboration System - Structured Communication + Non-Text State Passing + Shared Memory"
LABEL target_os="openEuler 24.03 LTS-SP3"

# Install system dependencies
# faiss-cpu comes as prebuilt manylinux wheel, no compilation needed
# python3-devel and gcc are for optional sentence-transformers compilation
RUN dnf install -y \
    python3-pip \
    python3-devel \
    gcc \
    gcc-c++ \
    && dnf clean all

WORKDIR /app

# Copy and install Python dependencies (numpy, msgpack, faiss-cpu)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Optional: install sentence-transformers for real embeddings (adds ~2GB)
# RUN pip3 install --no-cache-dir sentence-transformers

# Optional: install requests for real LLM API calls
RUN pip3 install --no-cache-dir requests

# Copy project files
COPY . .

# Default entry point
ENTRYPOINT ["python3", "main.py"]
CMD ["demo"]
