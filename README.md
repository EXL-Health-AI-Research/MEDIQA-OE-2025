# MEDIQA-OE-2025

**Medical Order Extraction using MedGemma 27B + vLLM on RunPod**

This repository demonstrates multiple approaches for medical order extraction from clinical conversations as part of **MEDIQA-OE-2025**. We deploy **MedGemma-27B** on **RunPod** using **vLLM**, expose an OpenAI-compatible API, and run experiments via structured notebooks.

---

## ✅ Key Features
- **RunPod + vLLM** for scalable MedGemma deployment
- **OpenAI-compatible API** for inference
- Multiple approaches:
  - **Few-Shot Prompting**
  - **ReAct Reasoning**
  - **Agent Prompting**
- Organized **notebooks** and **prompt templates**

## 📂 Repository Layout

MEDIQA-OE-2025/
│
├── src/mediqa_oe/                     # Python package for core logic
│   ├── __init__.py
│   ├── data.py               # API call logic to vLLM server
│   └── lm
│        ├──__init__.py
│        ├── base.py
│        └── model.py              
│
├── notebooks/                     # Organized by approach
│   ├── .env.template
│   ├── 01_zero_shot.ipynb         # Zero-shot approach
│   ├── 02_few_shot.ipynb          # Few-shot prompt experiments
│   ├── 03_react_agent.ipynb       # ReAct-based approach
│   
├── main.py
├── pyproject.toml                 # Dependency config
├── README.md                      # Main documentation
└── uv.lock

## 🔧 Installation

### **1. Clone Repository**
```bash
git clone https://github.com/EXL-Health-AI-Research/MEDIQA-OE-2025.git
cd MEDIQA-OE-2025
```

### **2. Install Dependices**
```bash
pip install .
```
## Deploying vLLM on RunPod with MedGemma 27B

### **1. Start the Pod**

* Sign in to RunPod
* Launch a Custom Pod with:
    - **Image**: runpod/vllm-openai:latest
    -**GPU**: 2 A10 GPUs 48 vRAM 9 vCPU
    -**Disk**: ≥ 80GB
* Attach your Hugging Face token for gated models


### **2. Run the vLLM API Server**
 
Inside the container, execute:
```bash
--host 0.0.0.0 --port 8000 --model google/medgemma-27b-text-it --dtype bfloat16 --gpu-memory-utilization 0.98 --api-key <API-KEY>--max-model-len 24768 --tensor-parallel-size 2 --seed 1337 --enforce-eager
```

## ✅ Using Notebooks
* notebooks/01_few_shot.ipynb → Few-shot Prompting
* notebooks/02_react_reasoning.ipynb → ReAct Reasoning 
* notebooks/03_agentic_approach.ipynb → Agentic Approach

## 📊 Results Summary

|   **Approach**       | **Average Score** |
|----------------------|-------------------|
| Few-Shot Prompting   |       0.5090      |
| ReAct Reasoning      |       0.3891      |
| Agentic Approach     |       0.4602      |

