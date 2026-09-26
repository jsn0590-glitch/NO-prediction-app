import streamlit as st
import pandas as pd
import joblib
from PIL import Image


# =========================================================
# Page settings
# =========================================================

app_icon = Image.open("NO predictor_PNG.png")

st.set_page_config(
    page_title="NO Predictor",
    page_icon=app_icon,
    layout="centered"
)


# =========================================================
# Load model
# =========================================================

model_package = joblib.load(
    "NO_prediction_control_model.pkl"
)

model = model_package["model"]
features = model_package["features"]


# =========================================================
# Header
# =========================================================

col1, col2 = st.columns([1, 5])

with col1:
    st.image(app_icon, width=85)

with col2:
    st.title("NO Predictor")
    st.caption("Control-referenced RGB-based NO prediction")


st.write(
    "Control과 Sample의 RGB 값을 입력하면 "
    "배지 색상 정보를 기반으로 NO 농도를 예측합니다."
)


# =========================================================
# Control RGB
# =========================================================

st.subheader("1. Control RGB")

c1, c2, c3 = st.columns(3)

with c1:
    control_r = st.number_input(
        "Control R",
        min_value=0.0,
        max_value=255.0,
        value=150.0,
        step=0.001
    )

with c2:
    control_g = st.number_input(
        "Control G",
        min_value=0.0,
        max_value=255.0,
        value=105.0,
        step=0.001
    )

with c3:
    control_b = st.number_input(
        "Control B",
        min_value=0.0,
        max_value=255.0,
        value=107.0,
        step=0.001
    )


# =========================================================
# Sample RGB
# =========================================================

st.subheader("2. Sample RGB")

s1, s2, s3 = st.columns(3)

with s1:
    sample_r = st.number_input(
        "Sample R",
        min_value=0.0,
        max_value=255.0,
        value=150.0,
        step=0.001
    )

with s2:
    sample_g = st.number_input(
        "Sample G",
        min_value=0.0,
        max_value=255.0,
        value=120.0,
        step=0.001
    )

with s3:
    sample_b = st.number_input(
        "Sample B",
        min_value=0.0,
        max_value=255.0,
        value=110.0,
        step=0.001
    )


# =========================================================
# Prediction
# =========================================================

if st.button(
    "NO 예측",
    type="primary",
    use_container_width=True
):

    control_total = (
        control_r +
        control_g +
        control_b
    )

    sample_total = (
        sample_r +
        sample_g +
        sample_b
    )

    if control_total == 0 or sample_total == 0:

        st.error(
            "RGB 값의 합은 0보다 커야 합니다."
        )

    else:

        # Control normalized RGB
        control_norm_r = control_r / control_total
        control_norm_g = control_g / control_total
        control_norm_b = control_b / control_total

        # Sample normalized RGB
        sample_norm_r = sample_r / sample_total
        sample_norm_g = sample_g / sample_total
        sample_norm_b = sample_b / sample_total

        # Model input
        input_data = pd.DataFrame(
            [[
                sample_norm_r,
                sample_norm_g,
                sample_norm_b,
                control_norm_r,
                control_norm_g,
                control_norm_b
            ]],
            columns=features
        )

        prediction = model.predict(
            input_data
        )[0]


        # =================================================
        # Result
        # =================================================

        st.divider()

        st.subheader("Prediction Result")

        st.metric(
            "Predicted NO concentration",
            f"{prediction:.2f} μM"
        )


        # =================================================
        # Calculated color information
        # =================================================

        sample_gb = (
            sample_g / sample_b
            if sample_b != 0
            else None
        )

        control_gb = (
            control_g / control_b
            if control_b != 0
            else None
        )

        if (
            sample_gb is not None
            and control_gb is not None
        ):

            delta_gb = (
                sample_gb -
                control_gb
            )

            st.write(
                f"Sample G/B: **{sample_gb:.4f}**"
            )

            st.write(
                f"Control G/B: **{control_gb:.4f}**"
            )

            st.write(
                f"ΔG/B: **{delta_gb:.4f}**"
            )


        # =================================================
        # Normalized RGB information
        # =================================================

        with st.expander(
            "Normalized RGB 확인"
        ):

            st.write("**Sample**")

            st.write(
                f"R: {sample_norm_r:.4f}"
            )

            st.write(
                f"G: {sample_norm_g:.4f}"
            )

            st.write(
                f"B: {sample_norm_b:.4f}"
            )

            st.write("**Control**")

            st.write(
                f"R: {control_norm_r:.4f}"
            )

            st.write(
                f"G: {control_norm_g:.4f}"
            )

            st.write(
                f"B: {control_norm_b:.4f}"
            )


        # =================================================
        # Notice
        # =================================================

        st.info(
            "본 모델은 RAW 264.7 세포 배양액의 "
            "색상 정보를 이용한 연구용 예비 스크리닝 모델입니다. "
            "Griess assay를 대체하는 정량 분석법이 아닙니다."
        )
