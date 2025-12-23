# Medical QLoRA Fine-Tuning using Unsloth

This project demonstrates efficient medical domain fine-tuning of a large language model using **QLoRA** and **Unsloth** on Google Colab. The workflow focuses on parameter-efficient fine-tuning (PEFT) to adapt a pretrained model to medical question answering tasks while using minimal GPU memory.

---

## Project Overview

- Base Model: Meta LLaMA 3.1 8B (4-bit quantized)
- Fine-Tuning Method: QLoRA (Low-Rank Adaptation with quantization)
- Framework: Unsloth
- Dataset: Medical question–answer dataset (ChatDoctor / Medical QA)
- Platform: Google Colab (Tesla T4 GPU)

Only LoRA adapter weights are trained and saved, making the approach memory efficient and practical for limited-resource environments.

---

## Objectives

- Learn QLoRA-based fine-tuning for large language models
- Apply PEFT techniques to a medical domain dataset
- Reduce GPU memory usage using 4-bit quantization
- Train and evaluate a medical question answering model
- Save and reuse fine-tuned LoRA adapters

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Unsloth
- TRL (SFTTrainer)
- BitsAndBytes (4-bit quantization)
- Google Colab GPU

---

## Training Workflow

1. Set up Google Colab with GPU support
2. Install Unsloth and required dependencies
3. Load a 4-bit quantized pretrained LLaMA model
4. Apply LoRA adapters using Unsloth PEFT utilities
5. Load and format a medical Q&A dataset
6. Perform supervised fine-tuning using SFTTrainer
7. Monitor training loss and GPU memory usage
8. Save the fine-tuned LoRA adapter
9. Test the model on unseen medical queries

---

## Example Inference

**Prompt**
