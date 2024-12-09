import streamlit as st
import pandas as pd
from rank_bm25 import BM25Okapi
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema import HumanMessage
import os

# OpenAI API 키 설정
os.environ["API_KEY"] = "" 

# 모델 및 데이터 초기화 함수
@st.cache_resource
def initialize_models():
    # CSV 데이터 로드
    file_path = './data/응급처치_증상_data_final.csv'
    df = pd.read_csv(file_path)
    df['심각도2'] = df['심각도2'].fillna('없음')  # NaN 처리
    
    # 병원 데이터 로드
    hospital_file_path = r'C:\Users\user\langchain\study_langchain\emergency_streamlit\data\emergency_hospital_data_final.csv'  # 병원 데이터 파일 경로 예시
    hospital_df = pd.read_csv(hospital_file_path)
    
    # BM25 초기화
    tokenized_corpus = [symptom.split() for symptom in df['증상']]
    bm25 = BM25Okapi(tokenized_corpus)

    # FAISS 인덱스 로드
    embeddings = OpenAIEmbeddings()
    index = FAISS.load_local('./data/symptom_index/', embeddings, allow_dangerous_deserialization=True)

    return df, bm25, embeddings, index, hospital_df  # 병원 데이터도 함께 반환

# 응답 생성 함수
def generate_response(symptom_input, df, bm25, embeddings, index, hospital_df):
    if not symptom_input.strip():
        return "증상을 입력해주세요."

    # BM25 기반 검색
    symptom_tokens = symptom_input.split()
    scores = bm25.get_scores(symptom_tokens)
    top_indices_bm25 = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    matched_bm25_index = [i for i in top_indices_bm25 if scores[i] > 0]
    bm25_row = df.iloc[matched_bm25_index[0]] if matched_bm25_index else None

    # FAISS 기반 검색
    query_embedding = embeddings.embed_query(symptom_input.lower().strip())
    search_results = index.similarity_search_by_vector(query_embedding, k=5)

    matched_llm_row = None
    highest_similarity_score = -1
    for result in search_results:
        matched_text = result.page_content
        symptom_in_data = matched_text.split("\n")[0].replace("증상: ", "").lower().strip()
        similarity_score = result.score if hasattr(result, 'score') else 0

        if similarity_score > highest_similarity_score:
            highest_similarity_score = similarity_score
            matched_llm_row = df[df['증상'].str.lower().str.strip() == symptom_in_data].iloc[0]

    # BM25와 FAISS 결합 결과 선택
    best_row = bm25_row if bm25_row is not None else matched_llm_row

    if best_row is not None:
        symptom = best_row['증상']
        emergency_action = best_row['응급처치']
        severity1 = best_row['심각도1']
        severity2 = best_row['심각도2']
        related_field = best_row['진료과목']  # 진료과목을 가져옴

        # 응급처치 및 병원 추천 안내
        if severity1 == '중증' and severity2 == '없음':
            return (
                f"조급한 마음은 실수를 만들 수 있으니 접어두고.\n\n"
                f"천천히 이 다음을 따라와주세요 \n{emergency_action}\n\n"
                f"이 진료 관련 응급실을 빠르게 알려드릴게요! 현재 위치를 알려주세요.", related_field
            )
        elif severity1 == '경증' and severity2 == '없음':
            return (
                f"휴 다행이도 위험한 증상은 아니네요.\n\n"
                f"제가 이제 응급처치를 도와드릴게요. \n{emergency_action}\n\n"
                f"증상이 악화되거나 중증 증상이 나타나면 즉시 응급실을 방문하세요. "
                f"참고로, 응급실 방문 시 경증으로 분류되면 의료비 부담이 높아질 수 있습니다. "
                f"추가적인 질문이나 보상 관련 정보가 필요하시면 말씀해주세요!", related_field
            )
        elif severity1 == '경증' and severity2 == '중증':
            return (
                f"응급실가시기 전 우선 급한대로 다음을 따라와주세요. 빠른 호전이 가능할 수 있어요.\n\n"
                f"우선 \n{emergency_action}\n\n 그래도 안되면 응급실을 생각해 볼 수도 있을 것 같아요."
                f"증상이 생각보다 심하고 추가 증상이 있으면 알려주세요. "
                f"괜찮아지실 거예요! 추가적으로 도움이 필요하시면 언제든 다시 문의해주세요!", related_field
            )
    else:
        return "현재 입력된 증상에 대한 정보를 찾을 수 없습니다. 더 구체적인 증상을 말씀해주시면 도움을 드릴 수 있을 것 같아요.", None

# OpenAI 모델 초기화
model = ChatOpenAI(model='gpt-4')

# LangChain Prompt Template 정의
prompt_template = PromptTemplate(
    input_variables=["medical_field", "district", "recommendations"],
    template=(
        "사용자가 '{medical_field}' 진료과목으로 '{district}' 구역의 응급실을 찾고 있습니다. "
        "다음은 추천되는 병원 목록입니다:\n\n{recommendations}\n\n"
        "이 정보를 바탕으로 사용자에게 응급실 정보를 알려주세요."
    ),
)

# 응급실 추천 정보를 생성하는 함수
def generate_emergency_response(medical_field_input, user_district, df, hospital_df):
    # 추천 병원 정보 가져오기
    recommendations = recommend_emergency_room(df, medical_field_input, user_district, hospital_df)
    
    # LangChain을 통한 응답 생성
    chain = LLMChain(llm=model, prompt=prompt_template)
    
    # HumanMessage를 통해 모델에게 정확한 요청 전달
    human_message = HumanMessage(content=recommendations)
    
    # 응답 생성
    response = chain.run({
        "medical_field": medical_field_input,
        "district": user_district,
        "recommendations": human_message.content  # 응급실 추천 정보를 포함시킴
    })
    
    return response

# 응급실 안내 모델 수정
def recommend_emergency_room(df, medical_field_input, user_district, hospital_df):
    """
    진료과목과 사용자 지역을 기반으로 병원 정보를 추천합니다.
    """
    relevant_hospitals = hospital_df[ 
        hospital_df.loc[:, '진료과목1':'진료과목36']
        .apply(lambda row: row.str.contains(medical_field_input, case=False, na=False).any(), axis=1)
    ]
    relevant_hospitals = relevant_hospitals[
        relevant_hospitals['구역'].str.contains(user_district, case=False, na=False)
    ]

    if not relevant_hospitals.empty:
        recommendations = [
            f"{row['병원']}, 주소: {row['주소']}, 전화번호: {row['전화번호']}"
            for _, row in relevant_hospitals.iterrows()
        ]
        return "\n".join(recommendations)
    else:
        return "해당 진료과목 및 지역에 맞는 응급실 정보가 없습니다."

def display_chatbot_page():
    # 모델 초기화
    df, bm25, embeddings, index, hospital_df = initialize_models()  # 모든 반환값을 받아야 함

    st.title("🤖 스마트 응급처치 챗봇")
    st.write("응급처치 정보를 얻고 싶은 증상을 입력해주세요.")

    symptom_input = st.text_input("💬 증상을 입력하세요:", "")
    
    if symptom_input:
        response, related_field = generate_response(symptom_input, df, bm25, embeddings, index, hospital_df)  # 모든 모델을 연결
        
        st.write(response)

        # 응급실 안내가 필요한 경우 지역 입력 받기
        if related_field:  # 진료과목이 있을 경우에만 응급실 안내
            st.write("📍 구역을 입력하세요. 예시: 강서구, 구로구")

            user_district = st.text_input("구역을 입력하세요:", "")
            if user_district:  # 구역이 입력되었을 때
                emergency_room_response = generate_emergency_response(related_field, user_district, df, hospital_df)
                st.write(emergency_room_response)

# 챗봇 페이지 실행
display_chatbot_page()









