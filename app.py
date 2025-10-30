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
# 유틸: GitHub 페이지 URL -> raw URL 자동 변환
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

# 기본 데이터셋 3개 자리(1번은 고정, 2~3번은 추후 채워넣기)
DATASET_DEFAULTS = {
    "데이터셋 1": "https://github.com/hyeon9997/2025informatics_MLpractice/blob/main/snow_incheon.csv",
    "데이터셋 2": "",  # TODO
    "데이터셋 3": "",  # TODO
}

# --------------------------
# 사이드바: (선택) URL 편집
# --------------------------
with st.sidebar:
    st.header("🔗 데이터셋 링크 설정(선택)")
    st.caption("일반 GitHub 페이지 링크여도 raw 링크로 자동 변환됩니다.")
    ds1 = st.text_input("데이터셋 1(URL)", value=DATASET_DEFAULTS["데이터셋 1"])
    ds2 = st.text_input("데이터셋 2(URL)", value=DATASET_DEFAULTS["데이터셋 2"])
    ds3 = st.text_input("데이터셋 3(URL)", value=DATASET_DEFAULTS["데이터셋 3"])

# 상태 초기화
for k, v in {
    "pipeline": None,
    "problem_type": None,
    "features": [],
    "target": None,
    "X_test": None,
    "y_test": None,
    "test_indices": None
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🧠 지도학습(분류/회귀) 체험 웹앱")

# --------------------------
# ① 데이터 선택 (기본: 데이터셋 1 자동 로드)
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
        df = pd.read_csv(raw_url, encoding='cp949')
        st.success(f"✅ {choice} 불러오기 성공")
    except Exception as e:
        st.error(f"❌ {choice} 불러오기 실패: {e}")
else:
    st.info(f"{choice}에 URL이 비어 있습니다. 사이드바에서 GitHub CSV 링크를 입력해주세요.")

# --------------------------
# ② 데이터 미리보기(3행)
# --------------------------
st.subheader("② 데이터 미리보기 (상위 3행)")
if df is not None:
    st.dataframe(df.head(3), use_container_width=True)

# --------------------------
# ③ 문답지 — 3-3 객관식(다중 선택)
# --------------------------
st.subheader("③ 문답지 (스스로 생각해보기)")
with st.expander("문답지 열기/닫기", expanded=True):
    st.markdown(
        """
**3-1.** 여러분은 *기계학습 중 지도학습 방법*을 이용해 인공지능을 학습시킬 예정입니다.  
**3-2.** 지도학습은 **문제와 정답**이 같이 제공되는 학습 방식입니다.
        """
    )
    if df is not None:
        q_features_multi = st.multiselect(
            "3-3. 문제(예측에 필요한 데이터) 속성은? (여러 개 선택 가능)",
            options=list(df.columns),
            help="체크박스처럼 여러 개 선택할 수 있어요."
        )
    else:
        q_features_multi = []
        st.info("데이터가 로드되면 3-3 문항에 열 목록이 보입니다.")

    q_target = st.text_input("3-4. 정답(예측하고 싶은 값)은 무엇인가요? (단일 열명)", placeholder="예: 적설량")
    q_kind = st.radio("3-5. 예측하고 싶은 값은?", ["모름(자동판단)", "수치형(회귀)", "범주형(분류)"], horizontal=True)

# --------------------------
# ④ Feature / Target 설정
# --------------------------
st.subheader("④ Feature / Target 설정")
problem_type = None
features = []
target = None

if df is not None:
    all_cols = list(df.columns)
    features = st.multiselect(
        "Feature(입력 변수) 선택",
        options=all_cols,
        default=[c for c in q_features_multi if c in all_cols],
        help="문답지(3-3)에서 고른 항목이 기본으로 반영됩니다."
    )

    preset_target = q_target.strip() if q_target.strip() in all_cols else None
    target = st.selectbox(
        "Target(예측할 변수) 선택",
        options=["<선택>"] + all_cols,
        index=(all_cols.index(preset_target) + 1) if preset_target in all_cols else 0
    )
    target = None if target == "<선택>" else target

    if target:
        if q_kind.startswith("수치형"):
            problem_type = "regression"
        elif q_kind.startswith("범주형"):
            problem_type = "classification"
        else:
            problem_type = "regression" if pd.api.types.is_numeric_dtype(df[target]) else "classification"

    with st.container(border=True):
        st.write("**선택 요약**")
        st.write("- Features:", features if features else "없음")
        st.write("- Target:", target if target else "없음")
        st.write("- 문제 유형:", "회귀(수치형)" if problem_type == "regression"
                 else ("분류(범주형)" if problem_type == "classification" else "미정"))

# --------------------------
# ⑤ 모델 학습 (회귀: 선형회귀 / 분류: KNN)
# --------------------------
st.subheader("⑤ 모델 학습")
if df is not None and target and features:
    X = df[features].copy()
    y = df[target].copy()

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

    colA, colB = st.columns(2)
    with colA:
        test_size = st.slider("검증용 데이터 비율", 0.1, 0.5, 0.2, step=0.05)
    with colB:
        random_state = st.number_input("랜덤 시드", min_value=0, value=42, step=1)

    # split 시 원본 인덱스 유지(비교를 위해)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if problem_type == "classification" else None
    )

    if problem_type == "regression":
        model = LinearRegression()
    elif problem_type == "classification":
        k = st.slider("KNN의 이웃 수 (k)", 1, 25, 5)
        model = KNeighborsClassifier(n_neighbors=k)
    else:
        model = None
        st.warning("문제 유형을 결정할 수 없습니다. Target을 확인하세요.")

    if model and st.button("🚀 학습하기", type="primary"):
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        st.session_state.pipeline = pipeline
        st.session_state.problem_type = problem_type
        st.session_state.features = features
        st.session_state.target = target
        st.session_state.X_test = X_test.copy()
        st.session_state.y_test = y_test.copy()
        st.session_state.test_indices = X_test.index.to_list()

        st.success("학습이 완료되었습니다!")

        y_pred = pipeline.predict(X_test)
        with st.container(border=True):
            st.markdown("**검증 성능**")
            if problem_type == "regression":
                st.write(f"- MAE: **{mean_absolute_error(y_test, y_pred):.4f}**")
                st.write(f"- R²: **{r2_score(y_test, y_pred):.4f}**")
            else:
                st.write(f"- 정확도: **{accuracy_score(y_test, y_pred):.4f}**")
                try:
                    st.write(f"- F1-macro: **{f1_score(y_test, y_pred, average='macro'):.4f}**")
                except Exception:
                    pass

# --------------------------
# ⑥ 검증 데이터에서 행 선택 → 예측 vs 실제 비교
# --------------------------
st.subheader("⑥ 검증 데이터로 모델 점검 (행 선택 → 예측 vs 실제)")
if (
    df is not None
    and st.session_state.pipeline is not None
    and st.session_state.X_test is not None
    and st.session_state.y_test is not None
    and len(st.session_state.test_indices) > 0
):
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    idx_options = st.session_state.test_indices

    # 간단한 미리보기 테이블
    st.caption("아래 표는 검증셋(X_test) 앞부분 일부 미리보기입니다.")
    st.dataframe(X_test.head(5), use_container_width=True)

    selected_idx = st.selectbox("예측할 행(원본 인덱스) 선택", options=idx_options)
    if st.button("🔍 이 행 예측하기", type="primary"):
        try:
            row_X = X_test.loc[[selected_idx]]  # DataFrame 형식 유지
            true_y = y_test.loc[selected_idx]
            pred_y = st.session_state.pipeline.predict(row_X)[0]

            with st.container(border=True):
                st.markdown("**예측 결과 vs 실제 정답**")
                st.write("- 선택한 행 인덱스:", selected_idx)
                st.write("- 입력 특징값(X):")
                st.dataframe(row_X, use_container_width=True)
                # 결과 비교
                if st.session_state.problem_type == "regression":
                    st.success(f"예측 값: **{float(pred_y):.4f}**   |   실제 값: **{float(true_y):.4f}**")
                else:
                    st.success(f"예측 라벨: **{str(pred_y)}**   |   실제 라벨: **{str(true_y)}**")
        except Exception as e:
            st.error(f"예측 중 오류: {e}")
else:
    st.info("모델을 학습하면 검증 데이터에서 행을 골라 예측/비교할 수 있습니다.")

# --------------------------
# ⑦ 2024, 2025년 예측 (수동 입력)
# --------------------------
st.subheader("⑦ 2024/2025년 값 입력 → 예측")
if (
    df is not None
    and st.session_state.pipeline is not None
    and st.session_state.features
):
    features = st.session_state.features
    problem_type = st.session_state.problem_type

    # 입력 UI 생성 함수
    def build_manual_inputs(default_year: int):
        cols = st.columns(min(3, len(features)))
        inputs = {}
        for i, col in enumerate(features):
            with cols[i % len(cols)]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # 기본값: 중앙값, 단 '연도'로 보이면 default_year
                    if looks_like_year(col):
                        default_val = float(default_year)
                    else:
                        default_val = float(pd.to_numeric(df[col], errors="coerce").median())
                    inputs[col] = st.number_input(f"{col} (수치)", value=default_val)
                else:
                    uniques = df[col].dropna().astype(str).unique().tolist()
                    uniques = uniques[:200] if len(uniques) > 0 else [""]
                    default_idx = 0
                    # 연도형이지만 문자열로 저장된 경우에 대비
                    if looks_like_year(col):
                        # 연도 후보가 있으면 근사 매칭
                        if str(default_year) in uniques:
                            default_idx = uniques.index(str(default_year))
                    inputs[col] = st.selectbox(f"{col} (범주)", options=uniques, index=default_idx)
        return inputs

    st.markdown("**A. 2024년 입력**")
    inputs_2024 = build_manual_inputs(2024)
    st.markdown("**B. 2025년 입력**")
    inputs_2025 = build_manual_inputs(2025)

    col_pred_2024, col_pred_2025 = st.columns(2)
    with col_pred_2024:
        if st.button("🔮 2024년 예측", type="primary"):
            try:
                pred_df = pd.DataFrame([inputs_2024], columns=features)
                pred = st.session_state.pipeline.predict(pred_df)[0]
                if problem_type == "regression":
                    st.success(f"2024 예측 결과(수치): **{float(pred):.4f}**")
                else:
                    st.success(f"2024 예측 결과(범주): **{str(pred)}**")
            except Exception as e:
                st.error(f"2024 예측 중 오류: {e}")

    with col_pred_2025:
        if st.button("🔮 2025년 예측", type="primary"):
            try:
                pred_df = pd.DataFrame([inputs_2025], columns=features)
                pred = st.session_state.pipeline.predict(pred_df)[0]
                if problem_type == "regression":
                    st.success(f"2025 예측 결과(수치): **{float(pred):.4f}**")
                else:
                    st.success(f"2025 예측 결과(범주): **{str(pred)}**")
            except Exception as e:
                st.error(f"2025 예측 중 오류: {e}")
else:
    st.info("모델을 학습하면 2024/2025년 수동 입력 예측 기능을 사용할 수 있습니다.")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.caption("© 2025 지도학습 실습실 • GitHub CSV 로드 + 선형회귀/KNN • 검증행 비교 & 2024/2025 수동 예측")
