from __future__ import annotations

import os
import sys
from typing import Any

import cv2
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

MODEL_REPO = os.getenv(
    "PPE_MODEL_REPO", "keremberke/yolov8m-protective-equipment-detection"
)
MODEL_FILE = os.getenv("PPE_MODEL_FILE", "best.pt")
PERSON_MODEL_FILE = os.getenv("PERSON_MODEL_FILE", "yolo11n.pt")
CONFIDENCE = 0.35
IOU = 0.45

st.set_page_config(
    page_title="Edge Vision Safety Copilot",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --ink:#12221e; --mint:#c9f3df; --lime:#d7f56b; --paper:#f5f5ee; --coral:#ff7c61; }
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
    .stApp { background: var(--paper); }
    .hero { padding: 1.5rem 0 1rem; border-bottom: 1px solid #b8c5bd; }
    .eyebrow { font: 500 .75rem 'DM Mono', monospace; letter-spacing: .08em; text-transform: uppercase; color:#517068; }
    h1 { font-size: clamp(2.4rem, 5vw, 5.4rem) !important; line-height: .94 !important; letter-spacing: -.06em !important; max-width: 850px; margin: .45rem 0 1rem !important; }
    .lede { font-size: 1.05rem; max-width: 640px; color:#4c625b; }
    .metric { border-top: 2px solid var(--ink); padding: .7rem 0; }
    .metric-label { color:#62766f; font: .72rem 'DM Mono', monospace; text-transform:uppercase; }
    .metric-value { font-size: 1.65rem; font-weight: 600; margin-top: .15rem; }
    .status { padding: 1.2rem 1.4rem; border-radius: 4px; margin: .75rem 0 1.25rem; }
    .status.ok { background: var(--lime); }
    .status.alert { background: var(--coral); }
    .status-title { font-size: 1.6rem; font-weight: 700; }
    .status-copy { margin-top: .3rem; }
    [data-testid="stSidebar"] { background:#dcefe7; border-right:1px solid #b8c5bd; }
    .mono { font-family:'DM Mono', monospace; font-size:.8rem; color:#58726a; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Downloading and loading the vision models...")
def load_models() -> tuple[YOLO, YOLO]:
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return YOLO(model_path), YOLO(PERSON_MODEL_FILE)


def label_for(model: YOLO, class_id: int) -> str:
    names = model.names
    return str(names[class_id] if isinstance(names, dict) else names[class_id]).lower().replace("-", "_")


def is_person(label: str) -> bool:
    return label in {"person", "worker", "human"}


def is_helmet(label: str) -> bool:
    return any(token in label for token in ("helmet", "hardhat", "hard_hat", "hard hat")) and not any(
        token in label for token in ("no_", "without", "missing")
    )


def is_vest(label: str) -> bool:
    return any(token in label for token in ("vest", "jacket", "hi_vis", "high_vis", "safety_clothing")) and not any(
        token in label for token in ("no_", "without", "missing")
    )


def is_goggles(label: str) -> bool:
    return any(token in label for token in ("goggle", "eye_protection", "safety_glass")) and not any(
        token in label for token in ("no_", "without", "missing")
    )


def is_gloves(label: str) -> bool:
    return "glove" in label and not any(token in label for token in ("no_", "without", "missing"))


def is_mask(label: str) -> bool:
    return "mask" in label and not any(token in label for token in ("no_", "without", "missing"))


def is_shoes(label: str) -> bool:
    return any(token in label for token in ("shoe", "boot", "footwear")) and not any(
        token in label for token in ("no_", "without", "missing")
    )


def center(box: np.ndarray) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2)


def inside(point: tuple[float, float], box: np.ndarray) -> bool:
    return float(box[0]) <= point[0] <= float(box[2]) and float(box[1]) <= point[1] <= float(box[3])


def blur_faces(image: np.ndarray) -> tuple[np.ndarray, int]:
    if not hasattr(cv2, "CascadeClassifier"):
        return image, 0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(cascade_path)
    faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    output = image.copy()
    for x, y, width, height in faces:
        pad_x, pad_y = int(width * 0.15), int(height * 0.15)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(output.shape[1], x + width + pad_x), min(output.shape[0], y + height + pad_y)
        crop = output[y1:y2, x1:x2]
        if crop.size:
            output[y1:y2, x1:x2] = cv2.GaussianBlur(crop, (0, 0), sigmaX=15)
    return output, len(faces)


def run_inference(image: Image.Image, ppe_model: YOLO, person_model: YOLO) -> dict[str, Any]:
    rgb = np.array(image.convert("RGB"))
    ppe_result = ppe_model.predict(rgb, conf=CONFIDENCE, iou=IOU, verbose=False)[0]
    person_result = person_model.predict(rgb, classes=[0], conf=CONFIDENCE, iou=IOU, verbose=False)[0]
    detections: list[dict[str, Any]] = []
    if ppe_result.boxes is not None:
        for box, confidence, class_id in zip(ppe_result.boxes.xyxy.cpu().numpy(), ppe_result.boxes.conf.cpu().numpy(), ppe_result.boxes.cls.cpu().numpy()):
            detections.append({"box": box, "confidence": float(confidence), "label": label_for(ppe_model, int(class_id))})
    if person_result.boxes is not None:
        for box, confidence in zip(person_result.boxes.xyxy.cpu().numpy(), person_result.boxes.conf.cpu().numpy()):
            detections.append({"box": box, "confidence": float(confidence), "label": "person"})

    persons = [item for item in detections if is_person(item["label"])]
    equipment = [
        item
        for item in detections
        if any(
            check(item["label"])
            for check in (is_helmet, is_goggles, is_gloves, is_mask, is_shoes, is_vest)
        )
    ]
    person_reports = []
    for person in persons:
        person_box = person["box"]
        related = [item for item in equipment if inside(center(item["box"]), person_box)]
        has_helmet = any(is_helmet(item["label"]) for item in related)
        has_goggles = any(is_goggles(item["label"]) for item in related)
        has_gloves = any(is_gloves(item["label"]) for item in related)
        has_mask = any(is_mask(item["label"]) for item in related)
        has_shoes = any(is_shoes(item["label"]) for item in related)
        has_vest = any(is_vest(item["label"]) for item in related)
        person_reports.append(
            {
                "helmet": has_helmet,
                "goggles": has_goggles,
                "gloves": has_gloves,
                "mask": has_mask,
                "shoes": has_shoes,
                "vest": has_vest,
                "compliant": all((has_helmet, has_goggles, has_gloves, has_mask, has_shoes)),
            }
        )

    annotated = rgb.copy()
    colors = {
        "person": (18, 34, 30),
        "helmet": (30, 165, 255),
        "goggles": (44, 190, 105),
        "gloves": (205, 130, 35),
        "mask": (190, 70, 175),
        "shoes": (90, 110, 210),
        "vest": (44, 190, 105),
    }
    for item in detections:
        x1, y1, x2, y2 = [int(value) for value in item["box"]]
        label = item["label"]
        category = "person"
        for name, check in (
            ("helmet", is_helmet),
            ("goggles", is_goggles),
            ("gloves", is_gloves),
            ("mask", is_mask),
            ("shoes", is_shoes),
            ("vest", is_vest),
        ):
            if check(label):
                category = name
                break
        color = colors[category]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        caption = f"{label} {item['confidence']:.0%}"
        cv2.putText(annotated, caption, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .62, color, 2, cv2.LINE_AA)
    private_image, face_count = blur_faces(annotated)
    return {"image": private_image, "persons": person_reports, "detections": detections, "faces": face_count}


with st.sidebar:
    st.markdown("### Control room")
    st.markdown("Local-only PPE analysis for safety teams.")
    st.divider()
    st.markdown("**Privacy mode**")
    st.markdown("Images are processed in this Python process. Face regions are blurred before results are displayed.")
    st.divider()
    st.markdown(f"Model repository  \\`{MODEL_REPO}\\`")
    st.markdown("<span class='mono'>Inference target: local edge device</span>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Edge Vision / Safety Copilot</div>
      <h1>PPE compliance,<br>without the cloud.</h1>
      <div class="lede">Upload a site image or use your camera. The detector identifies protective equipment locally, then turns the result into a fast safety decision.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

input_mode = st.radio("Input", ["Upload image", "Camera snapshot"], horizontal=True, label_visibility="collapsed")
source = st.file_uploader("Choose a site image", type=["jpg", "jpeg", "png"], label_visibility="collapsed") if input_mode == "Upload image" else st.camera_input("Capture a site image")

if source is None:
    st.info("Add an image to start a private PPE check.")
    if not st.runtime.exists():
        print("Start this app with: streamlit run app.py")
        sys.exit(0)
    st.stop()

try:
    ppe_model, person_model = load_models()
    image = Image.open(source)
    with st.spinner("Running local inference..."):
        report = run_inference(image, ppe_model, person_model)
except Exception as error:
    st.error(f"The model could not run: {error}")
    st.stop()

persons = report["persons"]
non_compliant = sum(not person["compliant"] for person in persons)
compliant = bool(persons) and non_compliant == 0
status_class = "ok" if compliant else "alert"
status_title = "ALL CLEAR" if compliant else "ACTION REQUIRED"
status_copy = "Every detected worker has helmet, goggles, gloves, mask, and safety shoes." if compliant else "One or more detected workers may be missing required PPE. Verify on site."

left, right = st.columns([1.35, 1], gap="large")
with left:
    st.image(report["image"], use_container_width=True, caption="Private view: detected faces are blurred before display")
with right:
    st.markdown(f'<div class="status {status_class}"><div class="status-title">{status_title}</div><div class="status-copy">{status_copy}</div></div>', unsafe_allow_html=True)
    metric_a, metric_b = st.columns(2)
    with metric_a:
        st.markdown(f'<div class="metric"><div class="metric-label">Workers</div><div class="metric-value">{len(persons)}</div></div>', unsafe_allow_html=True)
    with metric_b:
        st.markdown(f'<div class="metric"><div class="metric-label">Needs review</div><div class="metric-value">{non_compliant}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric"><div class="metric-label">Detections</div><div class="metric-value">{len(report["detections"])}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric"><div class="metric-label">Faces blurred</div><div class="metric-value">{report["faces"]}</div></div>', unsafe_allow_html=True)
    if persons:
        st.markdown("#### Worker checks")
        for index, person in enumerate(persons, start=1):
            marker = "OK" if person["compliant"] else "REVIEW"
            checks = " · ".join(
                f"{label}" if person[key] else f"no {label}"
                for key, label in (
                    ("helmet", "helmet"),
                    ("goggles", "goggles"),
                    ("gloves", "gloves"),
                    ("mask", "mask"),
                    ("shoes", "safety shoes"),
                )
            )
            st.markdown(f"**Worker {index}** · `{marker}`  \\n+{checks}")
    else:
        st.warning("No worker was detected. Try a clearer, wider image.")

st.caption("This tool supports human safety checks; it does not replace a trained safety inspection.")
