import pandas as pd
from hanspell import spell_checker

# JSON 파일 로드
df = pd.read_json("naver_kin_data.json")

# 특수 문자 제거 (한글, 영어, 숫자, 공백만 유지)
df["content"] = df["content"].str.replace(r"[^가-힣a-zA-Z0-9\s]", "", regex=True)

# 중복 데이터 제거
df = df.drop_duplicates()

# 맞춤법 및 띄어쓰기 교정
def correct_spell(text):
    corrected = spell_checker.check(text).as_dict()["checked"]
    return corrected

df["content"] = df["content"].apply(correct_spell)

# 전처리된 데이터 저장
df.to_json("processed_data.json", orient="records", force_ascii=False)
