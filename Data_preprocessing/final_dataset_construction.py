import json

# 개별 데이터셋 로드
with open("naver_kin_data.json", "r", encoding="utf-8") as f:
    naver_data = json.load(f)

with open("klac_data.json", "r", encoding="utf-8") as f:
    klac_data = json.load(f)

# 데이터 합치기
final_dataset = naver_data + klac_data

# 최종 데이터셋 저장
with open("final_dataset.json", "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=4)
