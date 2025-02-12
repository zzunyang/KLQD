# KLQD: Korean Legal Question-Answering Dataset

## Project Overview

**KLQD (Korean Legal Question-Answering Dataset)** is a large-scale dataset for Korean legal QA, containing **16,419 expert-verified QA pairs**. The dataset is sourced from **NAVER Knowledge iN, Korea Legal Aid Corporation (KLAC), and Easy to Find Practical Law (EFPL)**, covering **nine major legal domains**, including civil, criminal, and administrative law.

Existing Korean legal datasets primarily focus on terminology extraction or judicial decision prediction, making them less suitable for practical legal consultations. KLQD addresses this limitation by **structuring real-world legal inquiries from non-experts**, enabling AI models to better understand legal contexts and generate more relevant responses.

Additionally, we fine-tuned **seven AI models, including LLaMA-3 and Gemma-2**, on KLQD and evaluated their performance using multiple metrics (ROUGE, METEOR, legal-specific similarity measures, and GPT-4-based evaluations). The results demonstrate that fine-tuned models significantly improve in legal comprehension, contextual reasoning, and response accuracy compared to their baseline counterparts.

KLQD and the seven fine-tuned models are released under an **open license**, making them freely available for research and practical applications. Our goal is to advance legal AI research and foster the development of real-world AI-powered legal assistance tools.

---

## Dataset & Model Downloads

### KLQD Dataset
- [KLQD Dataset (Hugging Face)](#) *(Add link here)*

### KLQD-trained Models
- [zzunyang/KLQD_llama3.1](https://huggingface.co/zzunyang/KLQD_llama3.1)
- [zzunyang/KLQD_llama3.2](https://huggingface.co/zzunyang/KLQD_llama3.2)
- [zzunyang/KLQD_ko_gemma2](https://huggingface.co/zzunyang/KLQD_ko_gemma2)
- [zzunyang/KLQD_ko_llama3.1](https://huggingface.co/zzunyang/KLQD_ko_llama3.1)
- [zzunyang/KLQD_ko_sft](#) *(Add link here)*
- [zzunyang/KLQD_law_gemma](https://huggingface.co/zzunyang/KLQD_law_gemma)
- [zzunyang/KLQD_law_llama3](https://huggingface.co/zzunyang/KLQD_law_llama3)

---

## Usage Guide

### 1. Load the Dataset
```python
from datasets import load_dataset

dataset = load_dataset("zzunyang/KLQD")
print(dataset["train"][0])
```

### 2. Load and Test a Fine-Tuned Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "zzunyang/KLQD_llama3.1"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_answer(question):
    inputs = tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

question = "How can I terminate a real estate contract?"
print(generate_answer(question))
```

---

## License

This project is provided under an **open license**, allowing free use for research and non-commercial purposes.

---

## Contribution

If you'd like to contribute, please submit an issue or a pull request on GitHub!

---

## Contact

For questions or suggestions, feel free to reach out via email or GitHub Issues.

📧 Email: **your_email@example.com** *(Add email here)*

📌 GitHub Issues: [Issues](https://github.com/zzunyang/KLQD/issues)
