<h1 align="center">SEALGuard Replication Package</h1>

<h4 align="center">This repository provides the replication package for our SEALGuard experiments on multilingual LLM safety alignment. </h4>

<p align="center">
  <img src="./imgs/sealguard_logo.png" alt="SEALGuard Logo" width="150"/>
</p>

## 🛡️ Multilingual Guardrail — SEALGuard

<p align="center">
  <img src="./imgs/sealguard_overview.png" alt="SEALGuard Overview" style="width:100%; max-width:1000px;"/>
</p>

Two models are publicly available on the Hugging Face Hub:

- [SEALGuard-7B](https://huggingface.co/MickyMike/SEALGuard-7B)
- [SEALGuard-1.5B](https://huggingface.co/MickyMike/SEALGuard-1.5B)

---

<p align="center">
  <img src="./imgs/sealsbench_logo.png" alt="SEALSBench Logo" width="150"/>
</p>

## 📊 Multilingual LLM Safety Alignment Benchmark — SEALSBench

<p align="center">
  <img src="./imgs/data_curation.png" alt="SEALSBench Data Curation" width="300"/>
</p>

Our benchmark dataset is also publicly available:

- [SEALSBench Dataset on Hugging Face](https://huggingface.co/datasets/MickyMike/SEALSBench)

---

## 📚 Table of Contents

1. [Environment Setup](#1-environment-setup)  
2. [Reproduce SEALGuard](#2-reproduce-sealguard)  
3. [Reproduce Baseline LlamaGuard](#3-reproduce-baseline-llamaguard)  
4. [Results CSV Files Available](#4-results-csv-files-available)  
5. [Citation](#5-citation)

---

## 1. Environment Setup

We recommend using **Python 3.12** for best compatibility and performance.

### Step 1: Install Python Requirements

To install all necessary dependencies, run:

```bash
pip install -r requirements.txt
```

### Step 2: Install PyTorch with CUDA

If you’re using an NVIDIA GPU, we highly recommend installing PyTorch with CUDA support to accelerate training and inference.
Follow the official installation guide from PyTorch:
👉 https://pytorch.org/get-started/locally

---

## 2. Reproduce SEALGuard

To run SEALGuard evaluation:

```bash
cd ./sealguard/
python main_seallm.py
```
⚙️ Required Variables
Before running, make sure to set the following variables inside main_seallm.py:
- hf_token → Your Hugging Face API key
- model_id → Choose one of the following models:
  - "MickyMike/SEALGuard-7B"
  - "MickyMike/SEALGuard-1.5B"

🔁 To Retrain SEALGuard using LoRA tuning:
```bash
cd ./sealguard/
sh train.sh
```
The LoRA-tuned model will be saved to the local directory: ./SEALGuard-7B/

You can adjust the model name and training config inside train.sh as needed.

---

## 3. Reproduce Baseline LlamaGuard

To evaluate the baseline LlamaGuard model, run the following commands:

```bash
cd ./llamaguard
python main.py
```
⚙️ Required Variables
Before running, make sure to set the following variables inside main_seallm.py:
- hf_token → Your Hugging Face API key
- model_id → Choose one of the following models:
  - "meta-llama/Llama-Guard-3-8B"
  - "meta-llama/Llama-Guard-3-1B"

---

## 4. Results CSV Files Available

All result files are available in the `./results` folder.  
Each CSV file contains model predictions along with the original input prompts for further analysis.

---

## 5. Citation

If you use SEALGuard or SEALSBench in your work, please consider citing our paper:

```bibtex
under review
```
