import pandas as pd
import os
import streamlit as st
from rank_bm25 import BM25Okapi
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI


# 환경 설정
os.environ['API_KEY']=''

# 데이터 및 모델 초기화
@st.cache_resource
def initialize_models():
    # CSV 데이터 로드
    file_path = './data/응급처치_증상_data_final.csv'
    df = pd.read_csv(file_path)
    df['심각도2'] = df['심각도2'].fillna('없음')  # NaN 처리
    
    # BM25 초기화
    tokenized_corpus = [symptom.split() for symptom in df['증상']]
    bm25 = BM25Okapi(tokenized_corpus)

    # FAISS 인덱스 로드
    embeddings = OpenAIEmbeddings()
    index = FAISS.load_local(
        './data/symptom_index/', embeddings, allow_dangerous_deserialization=True
    )

    return df, bm25, embeddings, index

# 초기화 실행
df, bm25, embeddings, index = initialize_models()

# Streamlit 앱 구성
st.title("응급처치 스마트 챗봇")
st.write("증상을 입력하면 적합한 응급조치와 병원 정보를 안내해드립니다.")

# 사용자 입력
symptom_input = st.text_input("증상을 입력하세요:", "")

# 응답 생성 함수
def generate_response(symptom_input):
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
                f"이 진료 관련 응급실을 빠르게 알려드릴게요! 현재 위치를 알려주세요."
                
            )
        elif severity1 == '경증' and severity2 == '없음':
            return (
                f"휴 다행이도 위험한 증상은 아니네요.\n\n"
                f"제가 이제 응급처치를 도와드릴게요. \n{emergency_action}\n\n"
                f"증상이 악화되거나 중증 증상이 나타나면 즉시 응급실을 방문하세요. "
                f"참고로, 응급실 방문 시 경증으로 분류되면 의료비 부담이 높아질 수 있습니다. "
                f"추가적인 질문이나 보상 관련 정보가 필요하시면 말씀해주세요!"
            )
        elif severity1 == '경증' and severity2 == '중증':

            return (
                f"응급실가시기 전 우선 급한대로 다음을 따라와주세요. 빠른 호전이 가능할 수 있어요.\n\n"
                f"우선 \n{emergency_action}\n\n 그래도 안되면 응급실을 생각해 볼 수도 있을 것 같아요."
                f"증상이 생각보다 심하고 추가 증상이 있으면 알려주세요. "
                f"괜찮아지실 거예요!"
                f"추가적으로 도움이 필요하시면 언제든 다시 문의해주세요!"
            )
        else:
            return "현재 입력된 증상에 대한 정보를 찾을 수 없습니다. 더 구체적인 증상을 말씀해주시면 도움을 드릴 수 있을 것 같아요."
    

# 응답 출력
if symptom_input:
    response = generate_response(symptom_input)
    st.write(response)
