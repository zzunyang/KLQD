import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import find_all_linear_names, print_trainable_parameters

# 기본 설정
output_dir = "./results_model_name"
model_name = "model_name"

# 데이터셋 로드
dataset = load_dataset("KLQD_dataset", split="train")

# 모델 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 모델 준비
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

# 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LORA 설정
peft_config = LoraConfig(
    r=128,
    lora_alpha=16,
    target_modules=find_all_linear_names(base_model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
base_model = get_peft_model(base_model, peft_config)
print_trainable_parameters(base_model)

# 데이터셋 전처리
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"아래는 상황을 설명하는 사용자의 질문입니다. 사용자의 질문에 대해 적절한 법적 답변이 포함된 응답을 작성하세요. 단, 답변은 대한민국 법을 바탕으로 작성되어야 합니다. \n###질문: {example['question'][i]}\n###응답: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

def preprocess_data(example):
    output_texts = formatting_prompts_func(example)
    tokenized = tokenizer(
        output_texts,
        truncation=True,
        padding="max_length",
        max_length=2048,  # 시퀀스 최대 길이
        return_tensors="pt"
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"],
    }

tokenized_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset.column_names)

# 학습 설정
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    num_train_epochs=3, 
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=3,
    logging_steps=500,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
)

# SFTTrainer 초기화 및 학습
trainer = SFTTrainer(
    model=base_model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    formatting_func=None,  # 이미 데이터셋에서 전처리됨
    args=training_args
)

trainer.train()
trainer.save_model(output_dir)
