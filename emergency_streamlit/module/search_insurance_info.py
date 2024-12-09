from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# OpenAI API 키 설정 
os.environ["API_KEY"] = ""

# FAISS 인덱스 로드
embedding = OpenAIEmbeddings()
faiss_db = FAISS.load_local(
    'C:/Users/user/langchain/study_langchain/data/insurance_faiss_index/',embedding, allow_dangerous_deserialization=True)  # 이미 저장된 FAISS 인덱스를 로드

# 검색기 생성
retriever = faiss_db.as_retriever(search_kwargs={"k": 3})

# 보험 약관 검색 함수
def search_insurance_terms(symptom, retriever):
    """
    증상과 관련된 보험 약관 내용을 검색하고 답변을 생성합니다.

    Args:
        symptom (str): 사용자의 질문 또는 검색 키워드.
        retriever: 검색에 사용할 FAISS retriever 객체.
    """
    # 증상에 대한 관련 문서 검색
    results = retriever.get_relevant_documents(symptom)
    
    if results:
        # 검색된 결과 중 가장 관련성 높은 내용 선택
        best_result = results[0]
        summary = best_result.page_content[:500].replace('\n', ' ')

        # 자연스러운 대화식 답변 생성
        prompt_template = """
        주어진 보험 약관에 대한 정보를 바탕으로 아래와 같이 질문에 대해 답변을 해주세요:
        
        사용자가 궁금해하는 주제: {symptom}
        관련 내용: {summary}
        
        질문에 대한 자연스럽고 대화식으로 답변을 생성해주세요.
        """
        
        prompt = PromptTemplate(input_variables=["symptom", "summary"], template=prompt_template)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # LLMChain을 통해 답변 생성
        response = chain.run({"symptom": symptom, "summary": summary})
        return response
    else:
        return "관련된 내용이 없습니다."

# 테스트 실행
if __name__ == "__main__":
    symptom_query = "아나팔락시스 보험"  # 예시 질문
    response = search_insurance_terms(symptom_query, retriever)
    
    print(f"질문: {symptom_query}")
    print(f"답변: {response}")

