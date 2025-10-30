# app.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

st.set_page_config(page_title="지도학습 실습실", page_icon="🧠", layout="wide")


# --------------------------
# 유틸: GitHub URL → raw URL 변환
# --------------------------
def to_raw_url(url: str) -> str:
    if not url:
        return url
    if "raw.githubusercontent.com" in url:
        return url
    return url.replace("github.com/", "raw.githubusercontent.com/").replace("/blob/", "/")


def looks_like_year(col_name: str) -> bool:
    name = col_name.strip().lower()
    return name in ["year", "연도", "년도"]


# --------------------------
# 기본 데이터셋
# --------------------------
DATASET_DEFAULTS = {
    "데이터셋 1": "https://github.com/hyeon9997/2025informatics_MLpractice/blob/main/snow_incheon.csv",
    "데이터셋 2": "",
    "데이터셋 3": "",
}

# --------------------------
# 사이드바
# --------------------------
with st.sidebar:
    st.header("🔗 데이터셋 링크 설정(선택)")
    st.caption("GitHub CSV 링크를 넣으면 raw로 자동 변환됩니다.")
    ds1 = st.text_input("데이터셋 1(URL)", value=DATASET_DEFAULTS["데이터셋 1"])
    ds2 = st.text_input("데이터셋 2(URL)", value=DATASET_DEFAULTS["데이터셋 2"])
    ds3 = st.text_input("데이터셋 3(URL)", value=DATASET_DEFAULTS["데이터셋 3"])

# --------------------------
# 세션 초기화
# --------------------------
for k, v in {
    "pipeline": None,
    "problem_type": None,
    "features": [],
    "target": None,
    "X_test": None,
    "y_test": None,
    "test_indices": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🧠 지도학습(분류/회귀) 체험 웹앱")

# --------------------------
# ① 데이터 선택
# --------------------------
st.subheader("① 데이터 선택")
choice = st.radio(
    "사용할 데이터셋을 선택하세요",
    options=["데이터셋 1", "데이터셋 2", "데이터셋 3"],
    horizontal=True,
    index=0,
)

DATASET_URLS = {"데이터셋 1": ds1, "데이터셋 2": ds2, "데이터셋 3": ds3}
raw_url = to_raw_url(DATASET_URLS[choice])

df = None
if raw_url:
    try:
        df = pd.read_csv(raw_url, encoding="cp949")
