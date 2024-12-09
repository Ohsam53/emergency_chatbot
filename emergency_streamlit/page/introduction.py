import streamlit as st

def show_project_description():
    # 프로젝트 배경
    st.subheader("🌟 프로젝트 배경")
    st.markdown("""
    - 🏥 응급실 과밀화 문제로 경증 환자의 진료비 부담 증가 예상  
    - 🚑 연휴와 의사 파업 등으로 병원 이용의 어려움 발생  
    - 🤔 증상 구분 어려움으로 119 상담 의존 및 정보 부족 문제  
    - 💡 건강보험 약관 이해 부족으로 병원비 청구 어려움
    """)

    # 프로젝트 목표
    st.subheader("🎯 프로젝트 목표")
    st.markdown("""
    - 🚨 **응급실 과밀화 완화** 및 **중증 환자 진료 속도 향상**  
    - 🩺 경증 환자에게 가벼운 증상별 응급처치 제공  
    - 💬 보험 약관 요약 정보로 병원 진료 고민 해결  
    """)

    # 주요 기능
    st.subheader("✨ 주요 기능")
    st.markdown("""
    - 🤖 **응급처치 챗봇**: 가벼운 증상별 응급처치 가이드 제공  
    - 📜 **보험 약관 요약**: 건강보험 정보 간단히 설명  
    - 🏥 **응급실 추천**: 위치와 증상에 따른 병원 정보 제공  
    """)

    # 기술 스택
    st.subheader("🛠️ 기술 스택")
    st.markdown("""
    - **OpenAPI** + **LangChain**: 챗봇 대화 흐름 관리  
    - **FAISS**: 임베딩 기반 빠른 검색  
    - **Pandas**: 데이터 처리  
    - **Streamlit**: 사용자 친화적 웹 구현  
    """)

    # 향후 계획
    st.subheader("🔮 향후 계획")
    st.markdown("""
    - 🛠️ 챗봇 응답 정확도 개선  
    """)

show_project_description()


