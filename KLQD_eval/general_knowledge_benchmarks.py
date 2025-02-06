from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from evaluate import load
from sentence_transformers import SentenceTransformer, util
import torch
import re
import time
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 데이터셋 및 모델 로드
dataset = load_dataset("klue", "mrc", split="validation")
model_name = "model_name"                                                              #model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# KR-SBERT 모델 로드
sbert_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# 평가 메트릭 초기화
meteor_metric = load("meteor")
rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# 텍스트 정규화 함수
def normalize_text(text):
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# 모델 예측 함수
def get_prediction_with_context(question, context):
    prompt = f"질문: {question}\n문맥: {context}\n답변:"
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            num_beams=2,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text[len(prompt):].strip()
    return answer

# 평가 실행
results = []
times = []
for i, example in enumerate(dataset):
    question = example["question"]
    context = example["context"]
    answer = example["answers"]["text"][0]  # 첫 번째 정답 사용
    category = example.get("category", "N/A")  # 카테고리가 없으면 "N/A"로 설정
    start_time = time.time()
    prediction = get_prediction_with_context(question, context)
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
        "context": context,
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
    if (i + 1) % 50 == 0:
        avg_time = sum(times) / len(times)
        print(f"진행 상황: {i + 1}/{len(dataset)}")
        print(f"현재 평균 응답 생성 시간: {avg_time:.2f}초")
        
# 결과 저장                                                                                                             #save as
with open("./evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
    
# 평가 요약 출력
bleu_scores = [res["BLEU"] for res in results]
meteor_scores = [res["METEOR"] for res in results]
sbert_similarities = [res["SBERT Similarity"] for res in results]
rouge1_scores = [res["ROUGE-1"] for res in results]
rouge2_scores = [res["ROUGE-2"] for res in results]
rougeL_scores = [res["ROUGE-L"] for res in results]
precision_scores = [res["Precision"] for res in results]
recall_scores = [res["Recall"] for res in results]
f1_scores = [res["F1"] for res in results]
accuracy_scores = [res["Accuracy"] for res in results]
print("\n평가 결과 요약")
print("=" * 40)
print(f"평균 BLEU: {sum(bleu_scores) / len(bleu_scores):.4f}")
print(f"평균 METEOR: {sum(meteor_scores) / len(meteor_scores):.4f}")
print(f"평균 ROUGE-1: {sum(rouge1_scores) / len(rouge1_scores):.4f}")
print(f"평균 ROUGE-2: {sum(rouge2_scores) / len(rouge2_scores):.4f}")
print(f"평균 ROUGE-L: {sum(rougeL_scores) / len(rougeL_scores):.4f}")
print(f"평균 SBERT Similarity: {sum(sbert_similarities) / len(sbert_similarities):.4f}")
print(f"평균 Precision: {sum(precision_scores) / len(precision_scores):.4f}")
print(f"평균 Recall: {sum(recall_scores) / len(recall_scores):.4f}")
print(f"평균 F1-Score: {sum(f1_scores) / len(f1_scores):.4f}")
print(f"평균 Accuracy: {sum(accuracy_scores) / len(accuracy_scores):.4f}")
print(f"평균 응답 생성 시간: {sum(times) / len(times):.2f}초")
print("=" * 40)
