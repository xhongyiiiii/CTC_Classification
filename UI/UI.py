import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import io
import zipfile
import time
import pandas as pd
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from models.mbrnet import PVTv2_GuidedBiFormer

# -----------------------------
# UI 設定
# -----------------------------

st.set_page_config(
    page_title="CTC Classification",
    page_icon="🔬",
    layout="wide"
)

st.title("CTC Classification System")

# -----------------------------
# 類別
# -----------------------------

CLASSES = [
    "CTC single",
    "CTC cluster",
    "Epithelial cell debris"
]

# -----------------------------
# 載入模型
# -----------------------------

MODEL_PATH = "weights/best_model.pth"

@st.cache_resource
def load_model():

    model = PVTv2_GuidedBiFormer(num_classes=3)

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model

model = load_model()

# -----------------------------
# 上傳影像
# -----------------------------

st.subheader("Upload Images")

bright = st.file_uploader("Brightfield Image", type=["png","jpg","tif"])
fluor_y = st.file_uploader("Yellow Fluorescence", type=["png","jpg","tif"])
fluor_g = st.file_uploader("Green Fluorescence", type=["png","jpg","tif"])
fluor_b = st.file_uploader("Blue Fluorescence", type=["png","jpg","tif"])

# -----------------------------
# 顯示影像 (選擇後立即顯示)
# -----------------------------

st.subheader("Input Images")

col1, col2, col3, col4 = st.columns(4)

if bright:
    col1.image(Image.open(bright), caption="Brightfield", use_container_width=True)

if fluor_y:
    col2.image(Image.open(fluor_y), caption="Yellow Fluorescence", use_container_width=True)

if fluor_g:
    col3.image(Image.open(fluor_g), caption="Green Fluorescence", use_container_width=True)

if fluor_b:
    col4.image(Image.open(fluor_b), caption="Blue Fluorescence", use_container_width=True)


# -----------------------------
# Preprocess (模型輸入)
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def load_img(file):

    img = Image.open(file)

    # 模型需要灰階
    img = img.convert("L")

    img = transform(img)

    return img.unsqueeze(0)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Run Prediction"):
    if not all([bright, fluor_y, fluor_g, fluor_b]):
        st.warning("Please upload all images.")
    else:
        b = load_img(bright)
        fy = load_img(fluor_y)
        fg = load_img(fluor_g)
        fb = load_img(fluor_b)

        with torch.no_grad():
            output = model(b, fb, fg, fy)  # 注意模型順序仍是 b, fb, fg, fy
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()

        # 存入 session_state
        st.session_state["prediction"] = CLASSES[pred]
        st.session_state["prob"] = prob.cpu().numpy()[0]
        st.success(f"Prediction: **{st.session_state['prediction']}**")

if "prediction" in st.session_state:
    st.subheader("Prediction Result")
    st.markdown(f"**Class:** {st.session_state['prediction']}")

# -----------------------------
# 使用者評估
# -----------------------------

# if "prediction" in st.session_state:

#     st.divider()
#     st.subheader("User Evaluation")

#     col1,col2 = st.columns(2)

#     if col1.button("Correct"):
#         st.session_state["feedback"] = "correct"

#     if col2.button("Wrong"):
#         st.session_state["feedback"] = "wrong"

# -----------------------------
# 儲存影像
# -----------------------------

# if "prediction" in st.session_state:

#     st.divider()
#     st.subheader("Download CTC Images")

#     # 將四張影像原始檔案讀取為 bytes
#     bright_bytes = bright.getvalue()
#     yellow_bytes = fluor_y.getvalue()
#     green_bytes = fluor_g.getvalue()
#     blue_bytes = fluor_b.getvalue()

#     col1, col2, col3, col4 = st.columns(4)
#     col1.download_button(
#         label="Brightfield",
#         data=bright_bytes,
#         file_name=bright.name,
#         mime="image/png"
#     )
#     col2.download_button(
#         label="Yellow Fluorescence",
#         data=yellow_bytes,
#         file_name=fluor_y.name,
#         mime="image/png"
#     )
#     col3.download_button(
#         label="Green Fluorescence",
#         data=green_bytes,
#         file_name=fluor_g.name,
#         mime="image/png"
#     )
#     col4.download_button(
#         label="Blue Fluorescence",
#         data=blue_bytes,
#         file_name=fluor_b.name,
#         mime="image/png"
#     )

if "prediction" in st.session_state:
    st.divider()
    st.subheader("Download CTC Images")

    # 讓使用者輸入 zip 檔名
    if "zip_name" not in st.session_state:
        st.session_state["zip_name"] = f"{st.session_state['prediction']}_CTC"

    user_zip_name = st.text_input("Enter zip file name", st.session_state["zip_name"])

    # 只生成一次 zip，存入 session_state
    if "zip_buffer" not in st.session_state:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w") as zf:
            images = [bright, fluor_y, fluor_g, fluor_b]  # 順序：亮視野、黃、綠、藍
            for idx, img in enumerate(images, start=1):
                zf.writestr(img.name, img.getvalue())  # 原始檔名
        zip_buffer.seek(0)
        st.session_state["zip_buffer"] = zip_buffer

    # 單一 download_button
    st.download_button(
        label="Download All 4 Images",
        data=st.session_state["zip_buffer"],
        file_name=f"{user_zip_name}.zip",
        mime="application/zip",
        key="download_ctc_zip"
    )


