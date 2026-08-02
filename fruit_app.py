# -*- coding: utf-8 -*-
"""
Fruit Classification Web App  (Streamlit)
=========================================
Two tabs:
  * Predict     -- upload an image, the trained CNN identifies the fruit.
  * Performance -- per-epoch training/validation accuracy & loss graphs, and a
                   confusion matrix computed on datasets/test.

Run:   streamlit run fruit_app.py
Needs: pip install streamlit tensorflow pillow numpy matplotlib
"""

import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
from backbones import get as _get_backbone

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "fruit_model.keras")
LABELS_PATH  = os.path.join(BASE_DIR, "fruit_labels.json")
HISTORY_PATH = os.path.join(BASE_DIR, "training_history.json")
CONFIG_PATH  = os.path.join(BASE_DIR, "model_config.json")
TEST_DIR     = os.path.join(BASE_DIR, "datasets", "test")

# match the backbone used at training time (preprocessing + input size)
_backbone = "inceptionv3"
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH) as _f:
            _backbone = json.load(_f).get("backbone", "inceptionv3")
    except Exception:
        pass
_, preprocess_input, IMG_SIZE = _get_backbone(_backbone)

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH) as f:
        CLASSES = json.load(f)
else:
    CLASSES = ["apple", "banana", "kiwi", "unknown"]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def import_and_predict(image_data, model):
    """Resize -> RGB -> InceptionV3 preprocessing -> predict (matches training)."""
    image = ImageOps.fit(image_data, (IMG_SIZE, IMG_SIZE))
    image = image.convert("RGB")
    arr = preprocess_input(np.asarray(image).astype(np.float32))
    return model.predict(arr[np.newaxis, ...])[0]


st.title("Fruit Classification")

if not os.path.exists(MODEL_PATH):
    st.error("fruit_model.keras not found. Train it first:  python fruit_model.py")
    st.stop()

model = load_model()
tab_predict, tab_perf = st.tabs(["Predict", "Performance"])

# --------------------------------------------------------------------------
# PREDICT TAB
# --------------------------------------------------------------------------
with tab_predict:
    st.write("Give the model a photo and it will predict the fruit "
             "(apple / banana / kiwi), or say it's not a fruit (unknown).")

    # choose how to give it an image: upload a file OR take a webcam photo
    source = st.radio("Image source", ["Upload an image", "Take a photo (webcam)"],
                      horizontal=True)
    if source == "Take a photo (webcam)":
        img_file = st.camera_input("Take a photo of the fruit")
    else:
        img_file = st.file_uploader("Please upload an image file",
                                    type=["jpg", "jpeg", "png"])

    if img_file is None:
        st.text("Waiting for an image...")
    else:
        image = Image.open(img_file)
        st.image(image, use_container_width=True)
        pred = import_and_predict(image, model)
        idx = int(np.argmax(pred))
        label = CLASSES[idx] if idx < len(CLASSES) else "unknown"
        conf = float(np.max(pred)) * 100
        if label == "unknown":
            st.subheader(f"This does not look like a fruit  ({conf:.1f}% unknown)")
        else:
            st.subheader(f"It is a {label}!  ({conf:.1f}% confident)")
        st.text("Probabilities per class:")
        st.bar_chart({CLASSES[i]: float(pred[i]) for i in range(len(CLASSES))})

# --------------------------------------------------------------------------
# PERFORMANCE TAB
# --------------------------------------------------------------------------
with tab_perf:
    # ---- training history graphs ----
    st.header("Training history (per epoch)")
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            h = json.load(f)
        n_ep = len(h.get("accuracy", []))
        epochs = list(range(1, n_ep + 1))

        colA, colB = st.columns(2)
        colA.metric("Best validation accuracy",
                    f"{max(h.get('val_accuracy', [0]))*100:.1f}%")
        colB.metric("Final training accuracy",
                    f"{(h.get('accuracy', [0])[-1])*100:.1f}%")

        if HAVE_MPL:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            ax1.plot(epochs, h.get("accuracy", []), marker="o", label="train")
            if "val_accuracy" in h:
                ax1.plot(epochs, h["val_accuracy"], marker="o", label="validation")
            ax1.set_title("Accuracy"); ax1.set_xlabel("epoch"); ax1.set_ylabel("accuracy")
            ax1.set_ylim(0, 1.02); ax1.grid(True, alpha=0.3); ax1.legend()

            ax2.plot(epochs, h.get("loss", []), marker="o", label="train")
            if "val_loss" in h:
                ax2.plot(epochs, h["val_loss"], marker="o", label="validation")
            ax2.set_title("Loss"); ax2.set_xlabel("epoch"); ax2.set_ylabel("loss")
            ax2.grid(True, alpha=0.3); ax2.legend()
            st.pyplot(fig)
        else:
            # fallback without matplotlib
            st.line_chart({k: h[k] for k in ("accuracy", "val_accuracy") if k in h})
            st.line_chart({k: h[k] for k in ("loss", "val_loss") if k in h})
    else:
        st.info("No training_history.json yet. Train with fruit_model.py to create it.")

    # ---- confusion matrix on the test set ----
    st.header("Confusion matrix (test set)")
    st.caption("Runs every image in datasets/test through the model. "
               "May take a minute at 299x299.")
    if st.button("Run test-set evaluation"):
        if not os.path.isdir(TEST_DIR):
            st.error("datasets/test not found.")
        else:
            with st.spinner("Predicting on datasets/test ..."):
                gen = ImageDataGenerator(preprocessing_function=preprocess_input) \
                    .flow_from_directory(TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE),
                                         batch_size=16, class_mode="categorical",
                                         shuffle=False)
                preds = model.predict(gen)
                y_true = gen.classes
                y_pred = preds.argmax(axis=1)
                labels = [c for c, _ in sorted(gen.class_indices.items(),
                                               key=lambda kv: kv[1])]
                n = len(labels)
                cm = np.zeros((n, n), dtype=int)
                for t, p in zip(y_true, y_pred):
                    cm[t, p] += 1
                acc = float((y_true == y_pred).mean())

            st.metric("Test accuracy", f"{acc*100:.1f}%")

            if HAVE_MPL:
                fig2, ax = plt.subplots(figsize=(1.6 + n, 1.6 + n))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_yticks(range(n)); ax.set_yticklabels(labels)
                ax.set_xlabel("Predicted"); ax.set_ylabel("True (actual)")
                ax.set_title("Confusion matrix")
                thresh = cm.max() / 2 if cm.max() else 0
                for i in range(n):
                    for j in range(n):
                        ax.text(j, i, cm[i, j], ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")
                fig2.colorbar(im, fraction=0.046, pad=0.04)
                fig2.tight_layout()
                st.pyplot(fig2)
            else:
                st.write("Confusion matrix (rows = true, cols = predicted):")
                st.table({labels[i]: {labels[j]: int(cm[i, j]) for j in range(n)}
                          for i in range(n)})

            # per-class precision / recall
            rows = {}
            for i, lab in enumerate(labels):
                tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                rows[lab] = {"precision": round(prec, 2), "recall": round(rec, 2),
                             "images": int(cm[i].sum())}
            st.write("Per-class precision / recall:")
            st.table(rows)
