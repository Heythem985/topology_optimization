import io
import os
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.unet import UNet


@st.cache_resource
def load_model(checkpoint_path='checkpoints/unet_best.pth', device='cpu'):
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = UNet(in_channels=4, out_channels=1)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model, device


def prepare_input_from_npz_bytes(npz_bytes, vol_override=None):
    f = io.BytesIO(npz_bytes)
    data = np.load(f)
    fx = data['force_x_elem'].astype(np.float32)
    fy = data['force_y_elem'].astype(np.float32)
    bc = data['bc_elem'].astype(np.float32)
    vol = None
    if 'volfrac' in data:
        try:
            vol = float(data['volfrac'])
        except Exception:
            vol = None
    if vol_override is not None:
        vol = float(vol_override)
    if vol is None:
        vol = 0.0
    vol_ch = np.full_like(fx, float(vol), dtype=np.float32)
    x = np.stack([fx, fy, bc, vol_ch], axis=0)
    return x, data


def run_inference(model, device, x_np, threshold=0.5):
    x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = (probs >= threshold).astype(np.uint8)
    return probs, pred


def viz_images(prob, pred, gt=None):
    fig, axes = plt.subplots(1, 3 if gt is not None else 2, figsize=(10, 4))
    axes[0].imshow(prob, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Prob')
    axes[0].axis('off')
    axes[1].imshow(pred, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Pred')
    axes[1].axis('off')
    if gt is not None:
        axes[2].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('GT')
        axes[2].axis('off')
    fig.tight_layout()
    return fig


def main():
    st.title('Topology UNet — Inference')

    st.sidebar.markdown('Model & input')
    ckpt = st.sidebar.text_input('Checkpoint path', 'checkpoints/unet_best.pth')
    device_choice = st.sidebar.selectbox('Device', ['cpu'])
    threshold = st.sidebar.slider('Threshold', 0.0, 1.0, 0.5)
    vol_override = st.sidebar.text_input('Volfrac (optional, override)', '')

    model, device = load_model(ckpt, device_choice)

    uploaded = st.file_uploader('Upload input .npz (force_x_elem, force_y_elem, bc_elem, optional volfrac)', type=['npz'])
    if uploaded is not None:
        input_bytes = uploaded.read()
        try:
            x_np, raw_data = prepare_input_from_npz_bytes(input_bytes, vol_override=vol_override or None)
        except Exception as e:
            st.error(f'Failed to read .npz: {e}')
            return

        col1, col2 = st.columns([1, 1])
        with col1:
            st.write('Input arrays shapes:')
            st.write('force_x_elem:', x_np[0].shape)
            st.write('force_y_elem:', x_np[1].shape)
            st.write('bc_elem:', x_np[2].shape)
            st.write('vol (used):', float(vol_override) if vol_override else float(raw_data['volfrac']) if 'volfrac' in raw_data else 0.0)

        probs, pred = run_inference(model, device, x_np, threshold=threshold)

        # Do not visualize ground truth in the app (gt ignored)
        fig = viz_images(probs, pred, None)
        st.pyplot(fig)

        # Download button for .npz
        out_buf = io.BytesIO()
        np.savez_compressed(out_buf, prob=probs, pred=pred)
        out_buf.seek(0)
        st.download_button('Download prediction (.npz)', out_buf, file_name=f'pred_{os.path.splitext(uploaded.name)[0]}.npz')

        st.success('Inference completed')


if __name__ == '__main__':
    main()
