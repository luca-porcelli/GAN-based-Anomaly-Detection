import os
import io
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import torch

from models.ganomaly import GANomaly
from src.utils import load_json

st.set_page_config(page_title="GANomaly Anomaly Detection", layout="wide")

@st.cache_resource
def load_model(weights_path, img_size, z_dim, lambda_img, lambda_latent):
    #device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    model = GANomaly(in_c=3, z_dim=z_dim, lambda_img=lambda_img, lambda_latent=lambda_latent).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device

def to_tensor(im, img_size):
    im = im.convert("RGB").resize((img_size, img_size))
    x = torch.from_numpy(np.array(im).astype("float32")/255.0).permute(2,0,1).unsqueeze(0)
    x = x*2 - 1
    return x

def residual_map(model, x, device):
    with torch.no_grad():
        x = x.to(device)
        x_rec, _, _ = model(x)
    res = torch.abs(x - x_rec) * 0.5
    res = res.mean(dim=1, keepdim=True)
    return res, x_rec

st.title("🔎 Anomaly Detection")
st.markdown("Upload images to compute anomaly scores and residual maps.")

cfg = load_json("configs.json", {})
img_size = st.sidebar.number_input("Image size", min_value=64, max_value=1024, value=int(cfg.get("img_size", 256)), step=32)
z_dim = st.sidebar.number_input("Latent dim (z)", min_value=32, max_value=512, value=128, step=16)
lambda_img = st.sidebar.number_input("λ image", min_value=0.0, value=float(cfg.get("lambda_img", 50.0)), step=1.0)
lambda_latent = st.sidebar.number_input("λ latent", min_value=0.0, value=float(cfg.get("lambda_latent", 1.0)), step=0.1)

weights = st.sidebar.text_input("Weights path", value="runs/.../weights/best.pt")

uploaded = st.file_uploader("Upload one or more images", type=["png","jpg","jpeg","bmp","tif","tiff"], accept_multiple_files=True)

if weights and Path(weights).exists():
    model, device = load_model(weights, img_size, z_dim, lambda_img, lambda_latent)
    st.sidebar.success("Model loaded")
else:
    model = device = None
    st.sidebar.warning("Provide a valid weights path")

th_percentile = st.sidebar.slider("Threshold percentile (on uploaded batch)", min_value=10, max_value=100, value=int(cfg.get("threshold_percentile",95)))

if uploaded and model is not None:
    cols = st.columns(3)
    scores = []
    imgs = []
    recs = []
    heats = []
    for uf in uploaded:
        im = Image.open(io.BytesIO(uf.read()))
        x = to_tensor(im, img_size)
        res, x_rec = residual_map(model, x, device)
        score = (lambda_img * torch.mean(torch.abs(x - x_rec)) + lambda_latent * torch.mean(torch.abs(model.E(x.to(device)) - model.E2(x_rec)))).item()
        scores.append(score)
        imgs.append(im.resize((img_size, img_size)))
        recs.append(Image.fromarray(np.uint8(((x_rec[0].detach().cpu().numpy().transpose(1,2,0)+1)/2)*255)))
        heat = (res[0,0].detach().cpu().numpy())
        heat = (heat/ (heat.max()+1e-8)*255).astype("uint8")
        heats.append(Image.fromarray(heat))

    thr = np.percentile(scores, th_percentile)
    st.write(f"**Threshold (percentile {th_percentile})**: {thr:.5f}")

    for im, rec, heat, s in zip(imgs, recs, heats, scores):
        with st.container():
            c1, c2, c3 = st.columns(3)
            c1.image(im, caption=f"Input — score={s:.5f}")
            c2.image(rec, caption="Reconstruction")
            c3.image(heat, caption="Residual heatmap (0-255)")
            st.markdown(f"**Predicted:** {'ANOMALY' if s>thr else 'NORMAL'}")
else:
    st.info("Upload images and load a model to start.")
