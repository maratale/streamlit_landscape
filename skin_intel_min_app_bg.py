import io, os, time, json, base64, re
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from PIL import Image
import requests

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ======== CONSTANTS (edit here if пути другие) ========
# Background image (applies to all pages)
BG_IMAGE_PATH   = "/Users/maratalekberov/Desktop/Скрины/Screenshot 2025-09-12 at 17.30.48.png"

# Skin (ResNet50, 2 classes)
SKIN_CKPT_PATH  = "/Users/maratalekberov/Downloads/true.pth"
SKIN_CLASSES    = ["benign", "malignant"]
SKIN_VAL_DIR    = ""  # e.g. "data/skin/val"

# Intel (ResNet18, 6 classes)
INTEL_CKPT_PATH = "resnet18_full_model1.pth"
INTEL_CLASSES   = ["buildings","forest","glacier","mountain","sea","street"]
INTEL_VAL_DIR   = ""  # e.g. "/Users/.../archive/seg_test/seg_test"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ======== STYLES ========
def inject_background(image_path: str):
    p = Path(image_path)
    if not p.is_file():
        return
    data = p.read_bytes()
    b64  = base64.b64encode(data).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{b64}") no-repeat center center fixed;
        background-size: cover;
    }}
    .block-container {{
        background: rgba(0,0,0,0.35);
        border-radius: 16px;
        padding: 1.2rem 1.2rem 2rem 1.2rem;
    }}
    section[data-testid="stSidebar"] > div:first-child {{
        background: rgba(0,0,0,0.35);
    }}
    </style>
    """, unsafe_allow_html=True)


# ======== UTILS ========
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def tfm(size=224):
    from torchvision import transforms as T
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def _strip_prefix(sd, prefix):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}

def detect_resnet_arch(sd: dict) -> str:
    w = sd.get("layer1.0.conv1.weight")
    if w is not None and w.ndim == 4:
        kH, kW = w.shape[2], w.shape[3]
        if kH == 3 and kW == 3:  # basic block
            return "resnet18"
        if kH == 1 and kW == 1:  # bottleneck
            return "resnet50"
    fcw = sd.get("fc.weight")
    if fcw is not None and fcw.ndim == 2:
        in_f = fcw.shape[1]
        if in_f == 512:  return "resnet18"
        if in_f == 2048: return "resnet50"
    return "resnet18"

def load_model_resnet(ckpt_path: str, target_arch: str, out_dim: int, device):
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Checkpoint не найден: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    elif hasattr(obj, "state_dict"):
        sd = obj.state_dict()
    else:
        raise RuntimeError("Неподдерживаемый формат чекпоинта")

    for pref in ("module.", "model."):
        if any(k.startswith(pref) for k in sd):
            sd = _strip_prefix(sd, pref)

    arch_by_sd = detect_resnet_arch(sd)
    arch = target_arch or arch_by_sd

    if arch == "resnet50":
        net = models.resnet50(weights=None)
    else:
        net = models.resnet18(weights=None)

    net.fc = nn.Linear(net.fc.in_features, out_dim)
    net.load_state_dict(sd, strict=False)
    net.to(device).eval()
    return net

def extract_all_markdown(file_path: Path) -> str:
    """
    Возвращает полный Markdown из файла:
    - извлекает текст внутри тройных кавычек (три двойные или три одинарные)
    - если таких блоков нет, ищет переменные вида text, text1, text2 с многострочными/однострочными строками
    - если ничего не найдено — возвращает сырой текст файла
    """
    if not file_path.is_file():
        return ""
    raw = file_path.read_text(encoding="utf-8", errors="ignore")

    blocks = []
    # тройные кавычки
    for pat in (r'"""(.*?)"""', r"'''(.*?)'''"):
        blocks += re.findall(pat, raw, flags=re.DOTALL)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks if b.strip())

    # попытка вытащить text/text1/text2/... = """...""" или '...'
    for var in ("text", "text1", "text2", "text3"):
        m = re.search(var + r"\s*=\s*(?:[urUR]?)(['\"]{3})(.*?)\1", raw, flags=re.DOTALL)
        if m:
            return m.group(2).strip()
        m = re.search(var + r"\s*=\s*(['\"])(.*?)\1", raw, flags=re.DOTALL)
        if m:
            return m.group(2).strip()

    # иначе — весь сырой файл
    return raw.strip()


# ======== PAGES ========
def page_home():
    st.title("COMETS-2025")
    st.subheader("Классификация изображений: Skin disease and Landscape")
    st.markdown("—")
    st.markdown("### 👥 Участники проекта")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- **Надежда Ишпайкина**\n- **Дмитрий Кошелев**")
    with col2:
        st.markdown("- **Ярослав Пахомов**\n- **Марат Алекберов**")
    st.divider()
    st.markdown(
        "Используйте меню слева, чтобы открыть страницы **Skin Disease** (ResNet50), "
        "**Landscape** (ResNet18) или **📊 Отчёт и метрики**."
    )

def page_skin():
    st.header("🩺 Skin disease")
    device = get_device()
    st.caption(f"Device: **{device.type}**")

    try:
        model = load_model_resnet(SKIN_CKPT_PATH, target_arch="resnet50", out_dim=len(SKIN_CLASSES), device=device)
    except Exception as e:
        st.error(f"Не удалось загрузить модель: {e}")
        return

    size = st.slider("Input size", 160, 512, 224, 16)
    # компактные превью: слайдер ширины
    thumb_w = st.slider("Размер превью, px", 96, 512, 220, 8, key="thumb_w_skin")

    c1, c2 = st.columns(2)
    with c1:
        files = st.file_uploader("Upload images", type=["jpg","jpeg","png","bmp","webp","tif","tiff"], accept_multiple_files=True)
    with c2:
        urls_text = st.text_area("Or paste URLs (one per line)")

    if st.button("Classify", type="primary"):
        imgs: List[Image.Image] = []
        names: List[str] = []
        for f in files or []:
            try:
                imgs.append(Image.open(f).convert("RGB"))
                names.append(f.name)
            except Exception as e:
                st.warning(f"{f.name}: {e}")
        for line in (urls_text or "").splitlines():
            u = line.strip()
            if not u:
                continue
            try:
                imgs.append(load_image_from_url(u))
                names.append(Path(u).name)
            except Exception as e:
                st.warning(f"{u}: {e}")
        if not imgs:
            st.info("Добавьте изображения или URL.")
            return

        xb = torch.stack([tfm(size)(im) for im in imgs]).to(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        elapsed = time.perf_counter() - t0
        st.success(f"⏱ Processing time: {elapsed:.3f}s  (~{elapsed/len(imgs):.4f}s/img)")

        pred_idx = probs.argmax(axis=1)
        rows = [{"image": n, "pred": SKIN_CLASSES[p], "confidence": float(probs[i, p])}
                for i, (n, p) in enumerate(zip(names, pred_idx))]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        cols = st.columns(min(5, len(imgs)))
        for i, (im, n) in enumerate(zip(imgs, names)):
            label = SKIN_CLASSES[pred_idx[i]]
            with cols[i % len(cols)]:
                st.image(im, caption=f"{n} → {label} ({probs[i, pred_idx[i]]:.2%})", width=thumb_w)

def page_intel():
    st.header("🌍 Landscape")
    device = get_device()
    st.caption(f"Device: **{device.type}**")

    try:
        model = load_model_resnet(INTEL_CKPT_PATH, target_arch="resnet18", out_dim=len(INTEL_CLASSES), device=device)
    except Exception as e:
        st.error(f"Не удалось загрузить модель: {e}")
        return

    size = st.slider("Input size", 160, 512, 224, 16, key="intel_size")
    # компактные превью: слайдер ширины
    thumb_w = st.slider("Размер превью, px", 96, 512, 220, 8, key="thumb_w_intel")

    c1, c2 = st.columns(2)
    with c1:
        files = st.file_uploader("Upload images (multi)", type=["jpg","jpeg","png","bmp","webp","tif","tiff"], accept_multiple_files=True, key="intel_files")
    with c2:
        urls_text = st.text_area("Or paste URLs (one per line)", key="intel_urls")

    if st.button("Classify", type="primary", key="intel_run"):
        imgs: List[Image.Image] = []
        names: List[str] = []
        for f in files or []:
            try:
                imgs.append(Image.open(f).convert("RGB"))
                names.append(f.name)
            except Exception as e:
                st.warning(f"{f.name}: {e}")
        for line in (urls_text or "").splitlines():
            u = line.strip()
            if not u:
                continue
            try:
                imgs.append(load_image_from_url(u))
                names.append(Path(u).name)
            except Exception as e:
                st.warning(f"{u}: {e}")
        if not imgs:
            st.info("Добавьте изображения или URL.")
            return

        xb = torch.stack([tfm(size)(im) for im in imgs]).to(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        elapsed = time.perf_counter() - t0
        st.success(f"⏱ Processing time: {elapsed:.3f}s  (~{elapsed/len(imgs):.4f}s/img)")

        pred_idx = probs.argmax(axis=1)
        rows = [{"image": n, "pred": INTEL_CLASSES[p], "confidence": float(probs[i, p])}
                for i, (n, p) in enumerate(zip(names, pred_idx))]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        cols = st.columns(min(5, len(imgs)))
        for i, (im, n) in enumerate(zip(imgs, names)):
            label = INTEL_CLASSES[pred_idx[i]]
            with cols[i % len(cols)]:
                st.image(im, caption=f"{n} → {label} ({probs[i, pred_idx[i]]:.2%})", width=thumb_w)

def page_report():
    st.header("📊 Отчёт и метрики")

    base = Path(__file__).resolve().parent

    # 1) ТЕКСТ: подтянуть ВСЁ из main.py и main1.py
    text_blocks_found = False
    for name in ("main.py", "main1.py"):
        p = base / name
        md = extract_all_markdown(p)
        if md:
            text_blocks_found = True
            st.markdown(f"**Источник:** `{name}`")
            st.markdown(md)
            st.markdown("---")
    if not text_blocks_found:
        st.info("Не найдено текстовых блоков в main.py / main1.py. Проверь, что файлы лежат рядом и содержат Markdown в тройных кавычках.")

    # 2) ИЗОБРАЖЕНИЯ МЕТРИК: покажем всё, что есть (1 и 2)
    img_sets = [
        ("Metrics1.png", "Heatmap1.png", "Модель / Эксперимент 1"),
        ("Metrics2.png", "Heatmap2.png", "Модель / Эксперимент 2"),
    ]
    for m_img, h_img, title in img_sets:
        m_path = base / m_img
        h_path = base / h_img
        if m_path.exists() or h_path.exists():
            st.subheader(title)
            cols = st.columns(2)
            with cols[0]:
                if m_path.exists():
                    st.image(str(m_path), use_container_width=True, caption=f"{m_img} — кривые обучения/метрик")
                else:
                    st.warning(f"Не найден файл: {m_path}")
            with cols[1]:
                if h_path.exists():
                    st.image(str(h_path), use_container_width=True, caption=f"{h_img} — Confusion Matrix (heatmap)")
                else:
                    st.warning(f"Не найден файл: {h_path}")
            st.markdown("---")

    # 3) Дополнительные изображения .png в папке
    known = {n for pair in img_sets for n in pair[:2]}
    extras = [p for p in base.glob("*.png") if p.name not in known]
    if extras:
        st.subheader("Дополнительные изображения")
        for p in extras:
            st.image(str(p), use_container_width=True, caption=p.name)


# ======== MAIN ========
st.set_page_config(page_title="Skin & Intel Classifiers — Minimal (BG)", layout="wide")
inject_background(BG_IMAGE_PATH)

page = st.sidebar.radio("Page", ["🏠 Title", "Skin Disease", "Landscape", "📊 Отчёт и метрики"], index=0)

if page == "🏠 Title":
    page_home()
elif page == "Skin Disease":
    page_skin()
elif page == "Landscape":
    page_intel()
else:
    page_report()
