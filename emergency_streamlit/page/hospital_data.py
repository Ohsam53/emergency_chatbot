import streamlit as st
import pandas as pd
import os

def display_hospital_data_page():
    st.title("🏥 서울특별시 응급실 데이터 리스트")
    st.write("🔍 **서울특별시 응급실 데이터**를 조회할 수 있습니다. 지역구와 진료과목을 선택하여 원하는 병원을 찾아보세요!")

    # CSV 파일 경로 설정
    data_path = "./data/emergency_hospital_data_final.csv"
    
    # 파일 경로 존재 확인
    if not os.path.exists(data_path):
        st.error(f"⚠️ CSV 파일 경로에 문제가 있습니다: {data_path}")
        return
    
    # CSV 파일 읽기
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        st.error(f"🚨 CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
        return
    
    # 데이터 확인
    if df.empty:
        st.warning("⚠️ CSV 파일이 비어 있습니다.")
        return
    
    # 데이터 확인
    st.write("📋 **데이터 샘플 확인**")
    st.write(df.head())  # 데이터프레임의 앞부분을 확인
    
    # 진료과목 열 리스트 추출
    subject_cols = [col for col in df.columns if col.startswith("진료과목")]
    
    # 1. 구역 선택
    st.subheader("🌍 1. 구역 선택")
    unique_regions = sorted(df["구역"].dropna().unique())
    selected_region = st.selectbox("🌏 **구역을 선택하세요:**", ["전체"] + unique_regions)
    
    # 2. 진료과목 선택
    st.subheader("🩺 2. 진료과목 선택")
    unique_subjects = sorted(
        pd.concat([df[col].dropna() for col in subject_cols]).unique()
    )
    selected_subjects = st.multiselect("🩻 **진료과목을 선택하세요:**", unique_subjects)
    
    # 3. 조건에 따른 필터링
    st.subheader("🔎 3. 검색 결과")
    filtered_df = df.copy()

    # 구역 필터
    if selected_region != "전체":
        filtered_df = filtered_df[filtered_df["구역"] == selected_region]
    
    # 진료과목 필터
    if selected_subjects:
        filtered_df = filtered_df[filtered_df[subject_cols].apply(
            lambda row: any(subject in row.values for subject in selected_subjects),
            axis=1
        )]
    
    # 결과 출력
    if filtered_df.empty:
        st.warning("❌ 조건에 맞는 병원이 없습니다.")
    else:
        st.success(f"✅ 총 {len(filtered_df)}개의 병원을 찾았습니다!")
        # 병원, 주소, 전화번호 열만 출력
        st.dataframe(filtered_df[["병원", "주소", "전화번호"]])


