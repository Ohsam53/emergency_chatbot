import streamlit as st
from page.introduction import show_project_description  # 프로젝트 소개 관련 함수
from page.hospital_data import display_hospital_data_page  # 병원 데이터 관련 함수
from page.smart_chatbot import display_chatbot_page  # 스마트 챗봇 관련 함수

# 사이드바에서 페이지 선택
with st.sidebar:
    st.write("### 페이지 선택")
    selected_page = st.radio(
        "원하는 페이지를 선택하세요:",
        ("메인 페이지", "프로젝트 소개", "응급실 데이터", "스마트 응급가이드 챗봇")
    )

# 메인 페이지
if selected_page == "메인 페이지":
    st.title("🚑 우리 모두 안심, 스마트 응급가이드 챗봇 🚑")
    st.markdown(
        """
        ### 🩺 경증·중증 자가 진단 및 응급처치 도움 서비스

        **💡 여러분의 증상에 대한 고민을 해결해드립니다!**  

        - 🔍 **프로젝트 소개**: 서비스의 목적과 주요 기능을 알아보세요.  
        - 📋 **응급실 데이터**: 서울시 내 응급실 정보를 검색하고 필요한 병원을 찾아보세요.  
        - 🤖 **스마트 응급가이드 챗봇**: 증상에 맞는 응급처치 및 병원 추천을 받아보세요.

        #### 🎯 **왼쪽 메뉴에서 원하는 페이지를 선택하세요!**  
        """
    )

# 프로젝트 소개 페이지
elif selected_page == "프로젝트 소개":
    st.title("프로젝트 소개")
    show_project_description()  # 프로젝트 소개 함수 호출

# 응급실 데이터 페이지
elif selected_page == "응급실 데이터":
    display_hospital_data_page()  # 병원 데이터 페이지 호출

# 스마트 챗봇 페이지
elif selected_page == "스마트 응급가이드 챗봇":
    display_chatbot_page()  # 스마트 챗봇 페이지 호출




