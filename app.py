
import streamlit as st
import pandas as pd
import joblib
import colorsys
from PIL import Image
app_icon = Image.open("NO predictor_PNG")

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="NO Predictor",
    page_icon=app_icon,
    layout="centered"
)

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("NO_prediction_model.pkl")

# -----------------------------
# Prediction function
# -----------------------------
def predict_no(R, G, B):

    # Normalized G
    total = R + G + B

    if total == 0:
        return None, None, None

    norm_G = G / total

    # RGB -> HSV -> Hue
    r = R / 255
    g = G / 255
    b = B / 255

    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    hue = h * 360

    # Model input
    input_data = pd.DataFrame(
        [[hue, norm_G]],
        columns=["Hue_deg", "norm_G"]
    )

    # Prediction
    raw_prediction = model.predict(input_data)[0]

    # Display value cannot be negative
    predicted_no = max(0, raw_prediction)

    return predicted_no, hue, norm_G


# -----------------------------
# App interface
# -----------------------------
col1, col2 = st.columns([1, 5])

with col1:
    st.image(app_icon, width=90)

with col2:
    st.title("NO Predictor")
    st.caption("RGB-based NO Prediction System")

st.write(
    "RAW 264.7 세포 배양액의 RGB 값을 입력하면 "
    "NO 농도를 예측합니다."
)

st.subheader("RGB 입력")

R = st.number_input(
    "Red (R)",
    min_value=0.0,
    max_value=255.0,
    value=170.0,
    step=0.001,
    format="%.3f"
)

G = st.number_input(
    "Green (G)",
    min_value=0.0,
    max_value=255.0,
    value=125.0,
    step=0.001,
    format="%.3f"
)

B = st.number_input(
    "Blue (B)",
    min_value=0.0,
    max_value=255.0,
    value=130.0,
    step=0.001,
    format="%.3f"
)

# -----------------------------
# Prediction button
# -----------------------------
if st.button("NO 예측", type="primary"):

    predicted_no, hue, norm_G = predict_no(R, G, B)

    if predicted_no is None:

        st.error("RGB 값의 합이 0일 수 없습니다.")

    else:

        st.divider()

        st.subheader("예측 결과")

        st.metric(
            label="Predicted NO concentration",
            value=f"{predicted_no:.2f} μM"
        )

        st.write(f"Hue: **{hue:.2f}°**")
        st.write(f"Normalized G: **{norm_G:.4f}**")

        # Training NO range
        if predicted_no > 17.08333:

            st.warning(
                "예측값이 모델 학습 NO 범위 "
                "(0.61–17.08 μM)를 초과했습니다. "
                "해석에 주의하세요."
            )

        elif predicted_no < 2:

            st.warning(
                "저농도 NO 영역에서는 현재 모델의 "
                "예측 오차가 상대적으로 큽니다."
            )

        else:

            st.success(
                "예측값이 모델의 주요 학습 범위 내에 있습니다."
            )

st.divider()

st.caption(
    "본 프로그램은 RAW 264.7 세포 배양액의 RGB 기반 "
    "NO 농도 예측을 위한 연구용 screening tool입니다. "
    "Griess assay를 대체하는 정량 분석법이 아닙니다."
)
