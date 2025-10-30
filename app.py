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
# 1️⃣ 데이터 유형 선택 + Git 링크 입력
# --------------------------
st.sidebar.header("📁 데이터 설정")
data_type = st.sidebar.radio("데이터 유형 선택", ["인문", "사회", "과학", "미디어"], horizontal=True)
git_url = st.sidebar.text_input(
    "데이터 GitHub Raw 링크 입력",
    placeholder="예: https://raw.githubusercontent.com/사용자명/저장소/main/data.csv"
)

st.title("🧠 지도학습(분류/회귀) 체험 웹앱")
st.markdown(f"**선택한 데이터 유형:** `{data_type}`")

# --------------------------
# 2️⃣ 데이터 불러오기
# --------------------------
df = None
if git_url:
    try:
        df = pd.read_csv(git_url)
        st.success("✅ 데이터가 성공적으로 불러와졌습니다!")
    except Exception as e:
        st.error(f"❌ 데이터 불러오기 오류: {e}")

# --------------------------
# 3️⃣ 데이터 미리보기
# --------------------------
st.subheader("① 데이터 미리보기 (상위 3행)")
if df is not None:
    st.dataframe(df.head(3), use_container_width=True)
else:
    st.info("GitHub의 Raw CSV 링크를 입력하면 데이터를 불러옵니다.")

# --------------------------
# 4️⃣ 문답지
# --------------------------
st.subheader("② 문답지 (스스로 생각해보기)")
with st.expander("문답지 열기/닫기", expanded=True):
    st.markdown("""
**2-1.** 여러분은 *기계학습 중 지도학습 방법*을 이용해 인공지능을 학습시킬 예정입니다.  
**2-2.** 지도학습은 **문제와 정답**이 같이 제공되는 학습 방식입니다.
""")
    q_features = st.text_input("2-3. 문제(예측을 위해 필요한 데이터)에 해당하는 속성은?", placeholder="예: 온도, 습도, 풍속")
    q_target = st.text_input("2-4. 정답(예측하고 싶은 값)은 무엇인가요?", placeholder="예: 적설량")
    q_kind = st.radio("2-5. 예측하고 싶은 값은 수치형인가요, 범주형인가요?", ["모름(자동판단)", "수치형(회귀)", "범주형(분류)"], horizontal=True)

# --------------------------
# 5️⃣ Feature / Target 설정
# --------------------------
st.subheader("③ Feature / Target 설정")

if df is not None:
    all_cols = list(df.columns)
    preset_feats = [c.strip() for c in q_features.split(",") if c.strip() in all_cols]
    preset_target = q_target.strip() if q_target.strip() in all_cols else None

    features = st.multiselect("Feature(입력 변수) 선택", options=all_cols, default=preset_feats)
    target = st.selectbox("Target(예측할 변수) 선택", options=["<선택>"] + all_cols,
                          index=(all_cols.index(preset_target) + 1) if preset_target in all_cols else 0)
    if target == "<선택>":
        target = None

    # 자동 문제 유형 판단
    problem_type = None
    if target:
        if q_kind.startswith("수치형"):
            problem_type = "regression"
        elif q_kind.startswith("범주형"):
            problem_type = "classification"
        else:
            if pd.api.types.is_numeric_dtype(df[target]):
                problem_type = "regression"
            else:
                problem_type = "classification"

    st.write("**선택 결과 요약**")
    st.write("- Features:", features if features else "없음")
    st.write("- Target:", target if target else "없음")
    st.write("- 문제 유형:", "회귀(수치형)" if problem_type == "regression" else ("분류(범주형)" if problem_type else "미정"))

# --------------------------
# 6️⃣ 모델 학습
# --------------------------
st.subheader("④ 모델 학습")

model = None
train_ok = False

if df is not None and target and features:
    X = df[features]
    y = df[target]

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])
    categorical_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols),
    ])

    test_size = st.slider("검증용 데이터 비율", 0.1, 0.5, 0.2, step=0.05)
    random_state = st.number_input("랜덤 시드", 0, 100, 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if problem_type == "classification" else None
    )

    if problem_type == "regression":
        model = LinearRegression()
    elif problem_type == "classification":
        k = st.slider("KNN의 이웃 수 (k)", 1, 25, 5)
        model = KNeighborsClassifier(n_neighbors=k)

    if model:
        if st.button("🚀 학습 시작", type="primary"):
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            st.success("✅ 학습 완료!")

            y_pred = pipeline.predict(X_test)
            if problem_type == "regression":
                st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
                st.write(f"R²: {r2_score(y_test, y_pred):.4f}")
            else:
                st.write(f"정확도: {accuracy_score(y_test, y_pred):.4f}")
                st.write(f"F1-score: {f1_score(y_test, y_pred, average='macro'):.4f}")
            train_ok = True

# --------------------------
# 7️⃣ 예측 테스트
# --------------------------
st.subheader("⑤ 예측 테스트 입력창")

if df is not None and target and features and train_ok:
    st.markdown("입력값을 입력하고 예측 버튼을 눌러보세요!")

    inputs = {}
    cols = st.columns(min(3, len(features)))
    for i, col in enumerate(features):
        with cols[i % len(cols)]:
            if pd.api.types.is_numeric_dtype(df[col]):
                val = float(df[col].median())
                inputs[col] = st.number_input(f"{col}", value=val)
            else:
                uniques = df[col].dropna().unique().tolist()[:20]
                inputs[col] = st.selectbox(f"{col}", uniques)

    if st.button("🔮 예측하기", type="primary"):
        pred_df = pd.DataFrame([inputs])
        y_pred = pipeline.predict(pred_df)
        st.success(f"예측 결과: **{y_pred[0]}**")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.caption("© 2025 지도학습 체험실 — pandas.read_csv(git 링크) 기반 자동 학습/예측 앱")
