import pandas as pd
from langchain import LLMChain
from langchain.prompts import PromptTemplate
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI


os.environ["API_KEY"] = ""


# 데이터 로드
file_path = './data/emergency_hospital_data_final.csv'
df = pd.read_csv(file_path)

# LangChain Prompt Template 정의
prompt_template = PromptTemplate(
    input_variables=["medical_field", "district", "recommendations"],
    template=(
        "사용자가 '{medical_field}' 진료과목으로 '{district}' 지역의 응급실을 찾고 있습니다. "
        "다음은 추천되는 병원 목록입니다:\n\n{recommendations}\n\n"
        "이 정보를 바탕으로 사용자에게 응급실 정보를 제공하세요."
    ),
)

# OpenAI 모델 초기화
model = ChatOpenAI(model='gpt-4')

# 진료과목 및 구역을 기반으로 병원 추천하는 함수
def recommend_emergency_room(df, medical_field_input, user_district):
    """
    진료과목과 사용자 지역을 기반으로 병원 정보를 추천합니다.
    """
    # 진료과목 열에서 입력된 진료과목과 일치하는 병원 필터링
    relevant_hospitals = df[
        df.loc[:, '진료과목1':'진료과목36']
        .apply(lambda row: row.str.contains(medical_field_input, case=False, na=False).any(), axis=1)
    ]

    # 지역 조건 추가 필터링
    relevant_hospitals = relevant_hospitals[
        relevant_hospitals['지역'].str.contains(user_district, case=False, na=False)
    ]

    # 결과 반환
    if not relevant_hospitals.empty:
        recommendations = [
            f"{row['병원']}, 주소: {row['주소']}, 전화번호: {row['전화번호']}"
            for _, row in relevant_hospitals.iterrows()
        ]
        return "\n".join(recommendations)
    else:
        return "해당 진료과목 및 지역에 맞는 응급실 정보가 없습니다."

# LangChain Prompt Template 정의
prompt_template = PromptTemplate(
    input_variables=["medical_field", "district", "recommendations"],
    template=(
        "사용자가 '{medical_field}' 진료과목으로 '{district}' 지역의 응급실을 찾고 있습니다. "
        "다음은 추천되는 병원 목록입니다:\n\n{recommendations}\n\n"
        "이 정보를 바탕으로 사용자에게 응급실 정보를 제공하세요."
    ),
)

# 응급실 추천 정보를 생성하는 함수
def generate_emergency_response(medical_field_input, user_district):
    """
    LLM을 활용해 사용자에게 전달할 응급실 추천 메시지를 생성합니다.
    """
    # 추천 병원 정보 가져오기
    recommendations = recommend_emergency_room(df, medical_field_input, user_district)

    # 추천 병원이 없으면 해당 메시지 반환
    if recommendations == "해당 진료과목 및 지역에 맞는 응급실 정보가 없습니다.":
        return recommendations

    # LLM 사용하여 응급실 추천 메시지 생성
    prompt = prompt_template.format(
        medical_field=medical_field_input,
        district=user_district,
        recommendations=recommendations
    )
    return model(prompt).strip()

# 예시 사용
if __name__ == "__main__":
    # 사용자가 입력한 증상 및 지역 정보 (emergency_treatment.py에서 받아온 데이터로 가정)
    symptom_input = "복통"  # 예시 증상
    user_district = "서울"  # 예시 지역
    related_field = "내과"  # 예시 진료과목 (emergency_treatment.py에서 받아온 결과로 대체)

    # 응급실 추천 메시지 생성 및 출력
    if related_field:
        emergency_response = generate_emergency_response(related_field, user_district)
        print(emergency_response)
    else:
        print("해당 증상에 대한 적절한 진료과목 정보를 찾을 수 없습니다.")

