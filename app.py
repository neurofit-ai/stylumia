import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import ast
import torch
from torchvision.models import resnet50
from torchvision import transforms
import faiss
import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st

st.set_page_config("Stylumia Intelligence Technology")

st.header("Stylumia Intelligence Technology")
st.subheader(":dress:  Fashion Visual Search & Intelligent Styling Assistant")
st.divider()

# Load model
model = resnet50(weights=None)
model.fc = torch.nn.Identity()
model.load_state_dict(torch.load("models/resnet50_feature_extractor.pth"))
model.eval()

# Load FAISS indices and features
index_dresses = faiss.read_index("models/faiss_index_dresses.index")
index_jeans = faiss.read_index("models/faiss_index_jeans.index")
features_dresses = np.load("models/features_dresses.npy")
features_jeans = np.load("models/features_jeans.npy")

df_dresses = pd.read_csv("datasets/dresses_image_data.csv")
df_jeans = pd.read_csv("datasets/jeans_image_data.csv")

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        return model(input_tensor).squeeze().numpy()

# Visual search
def visual_search(query_image_path, k=5):
    query_features = extract_features(query_image_path).reshape(1, -1)
    _, I_dresses = index_dresses.search(query_features, k)
    _, I_jeans = index_jeans.search(query_features, k)

    # Display Dresses
    st.markdown("## Top Dress Matches:")
    dress_cols = st.columns(3)
    for i, idx in enumerate(I_dresses[0]):
        row = df_dresses.iloc[idx]
        image_path = ast.literal_eval(row["local_image_path"])[0]
        if image_path:
            with dress_cols[i % 3]:
                st.image(image_path, caption=row["product_name"], width=200)

    # Display Jeans
    st.markdown("## Top Jeans Matches:")
    jeans_cols = st.columns(3)
    for i, idx in enumerate(I_jeans[0]):
        row = df_jeans.iloc[idx]
        image_path = ast.literal_eval(row["local_image_path"])[0]
        if image_path:
            with jeans_cols[i % 3]:
                st.image(image_path, caption=row["product_name"], width=200)


# Outfit recommendation (e.g., pair dress with matching jeans)
def recommend_outfit(product_id, k=3):
    st.markdown("## Recommended Outfits:")

    if product_id in df_dresses["product_id"].values:
        idx = df_dresses[df_dresses["product_id"] == product_id].index[0]
        query_features = features_dresses[idx].reshape(1, -1)
        _, I = index_jeans.search(query_features, k)
        recommended = df_jeans.iloc[I[0]]
    elif product_id in df_jeans["product_id"].values:
        idx = df_jeans[df_jeans["product_id"] == product_id].index[0]
        query_features = features_jeans[idx].reshape(1, -1)
        _, I = index_dresses.search(query_features, k)
        recommended = df_dresses.iloc[I[0]]
    else:
        st.warning(f"Product ID {product_id} not found.")
        return

    # Display recommended images in 3 columns per row
    cols = st.columns(3)
    for i, (_, row) in enumerate(recommended.iterrows()):
        image_path = ast.literal_eval(row["local_image_path"])[0]
        if image_path:
            with cols[i % 3]:
                st.image(image_path, caption=row["product_name"], width=200)

col1, col2 = st.columns(2)

with col1:
    image = st.file_uploader("Choose an exiting photo", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
with col2:
    photo = st.camera_input("Take a beautiful picture")

if image:
    visual_search(image)
    recommend_outfit(df_dresses.iloc[0]["product_id"])