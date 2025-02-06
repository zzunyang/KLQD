from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
import torch
import re
import time
import json
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from evaluate import load

# 평가 지표 로드
meteor_metric = load("meteor")
sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# 데이터셋 로드
dataset_name = "lega_inpretation_dataset"
dataset = load_dataset(dataset_name)
examples = dataset["train"]

# 모델과 토크나이저 로드
model_name = "model_name"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

# 프롬프트 템플릿
prompt_template = """
아래는 {category}에 관한 법령 질의입니다. 질의 요지에 대해 적절한 법령 해석이 포함된 응답을 작성하세요. 단, 답변은 대한민국 법을 바탕으로 작성되어야 합니다.

### 질의요지:
{question}
### 회답:
"""

# 텍스트 정규화 함수
def normalize_text(text):
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# 모델을 이용한 답변 생성 함수
def get_prediction_with_prompt(question, category="법률"):
    prompt = prompt_template.format(question=question, category=category)
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=1028,
            num_beams=2,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text[len(prompt):].strip()
    return answer

# 평가 실행
results = []
times = []
for i, example in enumerate(examples):
    question = example["question"]
    answer = example["answer"]
    category = example.get("category", "N/A")  # 카테고리 기본값 설정
    
    # 답변 생성
    start_time = time.time()
    prediction = get_prediction_with_prompt(question, category)
    end_time = time.time()
    response_time = end_time - start_time
    times.append(response_time)
    
    # BLEU, ROUGE, METEOR 계산
    bleu = sentence_bleu([answer.split()], prediction.split())
    rouge_scores = rouge_scorer_obj.score(answer, prediction)
    meteor_score = meteor_metric.compute(predictions=[prediction], references=[answer])["meteor"]
    
    # KR-SBERT 유사도 계산
    pred_embedding = sbert_model.encode(prediction)
    ref_embedding = sbert_model.encode(answer)
    sbert_similarity = util.cos_sim(pred_embedding, ref_embedding).item()
    
    # Classification Metrics (Precision, Recall, F1-Score, Accuracy)
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(answer).split()
    common_tokens = set(pred_tokens) & set(truth_tokens)
    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
    recall = len(common_tokens) / len(truth_tokens) if truth_tokens else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = 1 if prediction == answer else 0
    
    # 결과 저장
    results.append({
        "question": question,
        "answer": answer,
        "prediction": prediction,
        "category": category,
        "BLEU": bleu,
        "ROUGE-1": rouge_scores["rouge1"].fmeasure,
        "ROUGE-2": rouge_scores["rouge2"].fmeasure,
        "ROUGE-L": rouge_scores["rougeL"].fmeasure,
        "METEOR": meteor_score,
        "SBERT Similarity": sbert_similarity,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "Response Time": response_time,
    })
    
    # 진행 상황 출력
    if (i + 1) % 50 == 0:
        avg_time = sum(times) / len(times)
        print(f"진행 상황: {i + 1}/{len(examples)}")
        print(f"현재 평균 응답 생성 시간: {avg_time:.2f}초")

# 결과 저장
with open("./evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# 평가 요약 출력
print("\n평가 결과 요약")
print("=" * 40)
for metric in ["BLEU", "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L", "SBERT Similarity", "Precision", "Recall", "F1", "Accuracy"]:
    scores = [res[metric] for res in results]
    print(f"평균 {metric}: {sum(scores) / len(scores):.4f}")
print(f"평균 응답 생성 시간: {sum(times) / len(times):.2f}초")
print("=" * 40)
