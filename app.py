import math
import re
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "Hệ thống nhận diện và dự đoán nhóm máu"
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MAX_IMAGE_SIDE = 1200
MASK_NAMES = [
    "anti_a_blue",
    "anti_b_yellow",
    "blood_red",
    "foreground",
    "reaction_candidate",
]


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
)


def find_model_files(model_dir: Path = MODEL_DIR) -> list[Path]:
    """Return all saved model packages, newest first."""
    if not model_dir.exists():
        return []
    return sorted(model_dir.rglob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_resource(show_spinner=False)
def load_model_package(model_path: str) -> tuple[dict, Path]:
    path = Path(model_path)
    package = joblib.load(path)
    validate_model_package(package, path)
    patch_legacy_model_compatibility(package["model"])
    return package, path


def validate_model_package(package: dict, model_path: Path) -> None:
    required_keys = {"model", "feature_columns", "target"}
    missing = required_keys - set(package.keys())
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Model package {model_path.name} thiếu key: {missing_text}")
    if not package["feature_columns"]:
        raise ValueError(f"Model package {model_path.name} không có feature_columns.")

def patch_legacy_model_compatibility(model) -> None:
    """Patch legacy sklearn estimators loaded from older pickle formats."""
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            patch_legacy_model_compatibility(step)
        return

    if model.__class__.__name__ == "SimpleImputer" and not hasattr(model, "_fill_dtype"):
        # scikit-learn 1.8 expects this attribute on older pickles.
        model._fill_dtype = getattr(model, "_fit_dtype", None)


def entropy_from_values(values: np.ndarray, bins: int = 32, value_range=(0, 255)) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=value_range, density=False)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def normalized_histogram(values: np.ndarray, bins: int, value_range, prefix: str) -> dict:
    if values.size == 0:
        return {f"{prefix}_hist_{i}": 0.0 for i in range(bins)}

    hist, _ = np.histogram(values, bins=bins, range=value_range, density=False)
    total = hist.sum()
    hist = hist.astype(np.float64) / total if total else np.zeros_like(hist, dtype=np.float64)
    return {f"{prefix}_hist_{i}": hist[i] for i in range(bins)}


def lbp_image(gray: np.ndarray) -> np.ndarray:
    center = gray[1:-1, 1:-1]
    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]
    lbp = np.zeros_like(center, dtype=np.uint8)
    for i, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << i)
    return lbp


def lbp_features(gray: np.ndarray, mask: np.ndarray, prefix: str, bins: int = 16) -> dict:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        feats = {f"{prefix}_lbp_hist_{i}": 0.0 for i in range(bins)}
        feats[f"{prefix}_lbp_entropy"] = 0.0
        return feats

    lbp = lbp_image(gray)
    inner_mask = mask[1:-1, 1:-1]
    values = lbp[inner_mask]
    feats = normalized_histogram(values, bins=bins, value_range=(0, 256), prefix=f"{prefix}_lbp")
    feats[f"{prefix}_lbp_entropy"] = entropy_from_values(values, bins=bins, value_range=(0, 256))
    return feats


def glcm_features(gray: np.ndarray, mask: np.ndarray, prefix: str, levels: int = 16) -> dict:
    empty_result = {
        f"{prefix}_glcm_contrast": 0.0,
        f"{prefix}_glcm_homogeneity": 0.0,
        f"{prefix}_glcm_energy": 0.0,
        f"{prefix}_glcm_entropy": 0.0,
    }
    if gray.shape[0] < 2 or gray.shape[1] < 2 or mask.sum() < 2:
        return empty_result

    quantized = np.clip((gray.astype(np.float32) / 256 * levels).astype(np.int32), 0, levels - 1)
    valid = mask[:, :-1] & mask[:, 1:]
    left = quantized[:, :-1][valid]
    right = quantized[:, 1:][valid]
    matrix = np.zeros((levels, levels), dtype=np.float64)

    for a, b in zip(left, right):
        matrix[a, b] += 1
        matrix[b, a] += 1

    total = matrix.sum()
    if total == 0:
        return empty_result

    p = matrix / total
    i, j = np.indices(p.shape)
    nonzero = p[p > 0]
    return {
        f"{prefix}_glcm_contrast": float(((i - j) ** 2 * p).sum()),
        f"{prefix}_glcm_homogeneity": float((p / (1 + np.abs(i - j))).sum()),
        f"{prefix}_glcm_energy": float(np.sqrt((p ** 2).sum())),
        f"{prefix}_glcm_entropy": float(-(nonzero * np.log2(nonzero)).sum()),
    }


def connected_component_features(mask: np.ndarray) -> dict:
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    areas = [stats[label, cv2.CC_STAT_AREA] for label in range(1, num_labels)]
    if not areas:
        return {
            "component_count": 0,
            "largest_component_area": 0,
            "largest_component_ratio": 0.0,
        }

    largest = max(areas)
    return {
        "component_count": len(areas),
        "largest_component_area": largest,
        "largest_component_ratio": largest / mask.size,
    }


def contour_shape_features(mask: np.ndarray, prefix: str) -> dict:
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            f"{prefix}_contour_count": 0,
            f"{prefix}_max_contour_area_ratio": 0.0,
            f"{prefix}_max_contour_perimeter_ratio": 0.0,
            f"{prefix}_max_contour_circularity": 0.0,
        }

    areas = np.array([cv2.contourArea(c) for c in contours], dtype=np.float64)
    perimeters = np.array([cv2.arcLength(c, True) for c in contours], dtype=np.float64)
    idx = int(np.argmax(areas))
    area = areas[idx]
    perimeter = perimeters[idx]
    circularity = (4 * math.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
    h, w = mask.shape
    return {
        f"{prefix}_contour_count": len(contours),
        f"{prefix}_max_contour_area_ratio": area / mask.size,
        f"{prefix}_max_contour_perimeter_ratio": perimeter / (2 * (h + w)),
        f"{prefix}_max_contour_circularity": circularity,
    }


def clean_mask(mask: np.ndarray, kernel_size: int = 5, min_area: int = 80) -> np.ndarray:
    mask = mask.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


def segment_image(rgb: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    anti_a_blue = (h >= 80) & (h <= 135) & (s >= 35) & (v >= 35)
    anti_b_yellow = (h >= 15) & (h <= 45) & (s >= 35) & (v >= 45)
    blood_red = (((h <= 12) | (h >= 165)) & (s >= 30) & (v >= 30))
    foreground = ((s >= 38) & (v >= 30) & (v <= 250)) | ((s >= 18) & (v <= 150))

    masks = {
        "anti_a_blue": clean_mask(anti_a_blue, kernel_size=5, min_area=120),
        "anti_b_yellow": clean_mask(anti_b_yellow, kernel_size=5, min_area=120),
        "blood_red": clean_mask(blood_red, kernel_size=5, min_area=120),
        "foreground": clean_mask(foreground, kernel_size=5, min_area=200),
    }
    masks["reaction_candidate"] = masks["foreground"] & (
        masks["blood_red"] | masks["anti_a_blue"] | masks["anti_b_yellow"]
    )
    return masks


def features_for_mask(rgb: np.ndarray, mask: np.ndarray, prefix: str) -> dict:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    values_rgb = rgb[mask]
    values_hsv = hsv[mask]
    gray_values = gray[mask]
    edge_values = edges[mask]
    grad_values = grad_mag[mask]

    result = {
        f"{prefix}_mask_present": int(mask.any()),
        f"{prefix}_area_ratio": float(mask.mean()),
    }
    result.update({f"{prefix}_{k}": v for k, v in connected_component_features(mask).items()})
    result.update(contour_shape_features(mask, prefix))

    if values_rgb.size == 0:
        for name in [
            "r_mean",
            "g_mean",
            "b_mean",
            "r_std",
            "g_std",
            "b_std",
            "h_mean",
            "s_mean",
            "v_mean",
            "h_std",
            "s_std",
            "v_std",
            "gray_mean",
            "gray_std",
            "edge_density",
            "gradient_mean",
            "gradient_std",
            "entropy",
        ]:
            result[f"{prefix}_{name}"] = 0.0
        result.update(normalized_histogram(np.array([]), bins=12, value_range=(0, 180), prefix=f"{prefix}_hue"))
        result.update(normalized_histogram(np.array([]), bins=8, value_range=(0, 256), prefix=f"{prefix}_sat"))
        result.update(normalized_histogram(np.array([]), bins=8, value_range=(0, 256), prefix=f"{prefix}_val"))
        result.update(lbp_features(gray, mask, prefix))
        result.update(glcm_features(gray, mask, prefix))
        return result

    result.update(
        {
            f"{prefix}_r_mean": values_rgb[:, 0].mean(),
            f"{prefix}_g_mean": values_rgb[:, 1].mean(),
            f"{prefix}_b_mean": values_rgb[:, 2].mean(),
            f"{prefix}_r_std": values_rgb[:, 0].std(),
            f"{prefix}_g_std": values_rgb[:, 1].std(),
            f"{prefix}_b_std": values_rgb[:, 2].std(),
            f"{prefix}_h_mean": values_hsv[:, 0].mean(),
            f"{prefix}_s_mean": values_hsv[:, 1].mean(),
            f"{prefix}_v_mean": values_hsv[:, 2].mean(),
            f"{prefix}_h_std": values_hsv[:, 0].std(),
            f"{prefix}_s_std": values_hsv[:, 1].std(),
            f"{prefix}_v_std": values_hsv[:, 2].std(),
            f"{prefix}_gray_mean": gray_values.mean(),
            f"{prefix}_gray_std": gray_values.std(),
            f"{prefix}_edge_density": float((edge_values > 0).mean()),
            f"{prefix}_gradient_mean": grad_values.mean(),
            f"{prefix}_gradient_std": grad_values.std(),
            f"{prefix}_entropy": entropy_from_values(gray_values),
        }
    )
    result.update(normalized_histogram(values_hsv[:, 0], bins=12, value_range=(0, 180), prefix=f"{prefix}_hue"))
    result.update(normalized_histogram(values_hsv[:, 1], bins=8, value_range=(0, 256), prefix=f"{prefix}_sat"))
    result.update(normalized_histogram(values_hsv[:, 2], bins=8, value_range=(0, 256), prefix=f"{prefix}_val"))
    result.update(lbp_features(gray, mask, prefix))
    result.update(glcm_features(gray, mask, prefix))
    return result


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Không đọc được ảnh. Hãy thử file JPG, JPEG hoặc PNG khác.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_if_needed(rgb: np.ndarray, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = max(h, w) / max_side
    if scale <= 1:
        return rgb
    new_size = (int(w / scale), int(h / scale))
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)


def extract_features_from_image(rgb: np.ndarray, expected_features: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rgb = resize_if_needed(rgb)
    features = {
        "resized_width": rgb.shape[1],
        "resized_height": rgb.shape[0],
    }

    masks = segment_image(rgb)
    for mask_name, mask in masks.items():
        features.update(features_for_mask(rgb, mask, mask_name))

    df_features = pd.DataFrame([features])
    for col in expected_features:
        if col not in df_features.columns:
            df_features[col] = 0.0
    return df_features[expected_features], masks


def model_label(path: Path) -> str:
    parent = path.parent.name.replace("_", " ")
    return f"{path.name} ({parent})"


def extract_model_timestamp(model_path: Path) -> str | None:
    match = re.search(r"(\d{8}_\d{6})", model_path.stem)
    return match.group(1) if match else None


def infer_model_name(model_path: Path, timestamp: str | None) -> str:
    stem = model_path.stem
    if timestamp:
        stem = stem.removesuffix(f"_{timestamp}")
    return stem.removeprefix("color_segmentation_blood_group_")


def accuracy_from_metrics_file(model_path: Path, model_name: str) -> float | None:
    timestamp = extract_model_timestamp(model_path)
    if not timestamp:
        return None

    metrics_files = sorted(model_path.parent.glob(f"*metrics_{timestamp}.csv"))
    for metrics_file in metrics_files:
        try:
            metrics_df = pd.read_csv(metrics_file)
        except Exception:
            continue

        if "model" not in metrics_df.columns or "test_accuracy" not in metrics_df.columns:
            continue

        matched_rows = metrics_df[metrics_df["model"].astype(str) == model_name]
        if not matched_rows.empty:
            return float(matched_rows.iloc[0]["test_accuracy"])

    return None


def accuracy_from_report_file(model_path: Path) -> float | None:
    timestamp = extract_model_timestamp(model_path)
    if not timestamp:
        return None

    report_files = sorted(model_path.parent.glob(f"*classification_report_{timestamp}.txt"))
    for report_file in report_files:
        try:
            report_text = report_file.read_text(encoding="utf-8")
        except Exception:
            continue

        match = re.search(r"^\s*accuracy\s+([0-9.]+)", report_text, flags=re.MULTILINE)
        if match:
            return float(match.group(1))

    return None


def get_model_accuracy(package: dict, model_path: Path) -> float | None:
    for key in ("test_accuracy", "accuracy"):
        if key in package:
            return float(package[key])

    metrics = package.get("metrics")
    if isinstance(metrics, dict):
        for key in ("test_accuracy", "accuracy"):
            if key in metrics:
                return float(metrics[key])

    timestamp = extract_model_timestamp(model_path)
    model_name = str(package.get("model_name") or infer_model_name(model_path, timestamp))
    return accuracy_from_metrics_file(model_path, model_name) or accuracy_from_report_file(model_path)


def prediction_probabilities(model, input_df: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(input_df), dtype=np.float64)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(input_df), dtype=np.float64)
        if scores.ndim == 1:
            pos = 1.0 / (1.0 + np.exp(-scores))
            probabilities = np.column_stack([1.0 - pos, pos])
        else:
            shifted = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    else:
        return None

    if probabilities.ndim == 1:
        probabilities = probabilities[np.newaxis, :]
    return probabilities


def top_predictions(model, input_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame | None:
    probabilities = prediction_probabilities(model, input_df)
    if probabilities is None:
        return None

    probabilities = probabilities[0]
    classes = model.classes_
    order = np.argsort(probabilities)[::-1][:top_n]
    return pd.DataFrame(
        {
            "nhom_mau": classes[order],
            "do_tin_cay": probabilities[order],
        }
    )


def prediction_confidence(top_df: pd.DataFrame | None) -> float | None:
    if top_df is None or top_df.empty:
        return None
    return float(top_df.iloc[0]["do_tin_cay"])


def render_sidebar(model_files: list[Path]) -> tuple[Path, bool, bool]:
    st.sidebar.header("Cấu hình")
    selected_label = st.sidebar.selectbox(
        "Model sử dụng",
        options=[model_label(path) for path in model_files],
        index=0,
    )
    selected_path = model_files[[model_label(path) for path in model_files].index(selected_label)]
    show_masks = st.sidebar.checkbox("Hiển thị mask phân đoạn", value=True)
    show_features = st.sidebar.checkbox("Hiển thị bảng feature", value=False)
    return selected_path, show_masks, show_features


def render_model_info(package: dict, model_path: Path) -> None:
    feature_columns = package["feature_columns"]
    accuracy = get_model_accuracy(package, model_path)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Thông tin model")
    st.sidebar.write(f"Thuật toán: `{package.get('model_name', 'N/A')}`")
    st.sidebar.write(f"Target: `{package.get('target', 'N/A')}`")
    st.sidebar.write(f"Số feature: `{len(feature_columns)}`")
    if accuracy is not None:
        st.sidebar.metric("Test accuracy", f"{accuracy:.2%}")
    else:
        st.sidebar.write("Test accuracy: `N/A`")
    st.sidebar.write(f"File: `{model_path.name}`")
    st.sidebar.caption(str(model_path.parent))


def render_mask_debug(masks: dict[str, np.ndarray]) -> None:
    st.subheader("Mask phân đoạn")
    cols = st.columns(len(MASK_NAMES))
    for idx, mask_name in enumerate(MASK_NAMES):
        mask = masks[mask_name].astype(np.uint8) * 255
        cols[idx].image(mask, caption=mask_name, use_container_width=True)


def render_prediction_card(prediction: str, top_df: pd.DataFrame | None) -> None:
    confidence_text = ""
    confidence = prediction_confidence(top_df)
    if confidence is not None:
        confidence_text = f"<p>Độ tin cậy của lần dự đoán: <strong>{confidence:.2%}</strong></p>"

    st.markdown(
        f"""
        <div style="
            background:#fff5f5;
            border:1px solid #ffd6d6;
            border-left:8px solid #c92a2a;
            border-radius:8px;
            padding:24px;
            text-align:center;">
            <p style="margin:0;color:#862e2e;font-weight:700;letter-spacing:0;">
                KẾT QUẢ DỰ ĐOÁN NHÓM MÁU
            </p>
            <h1 style="margin:8px 0 4px;color:#7f1d1d;font-size:72px;line-height:1;">
                {prediction}
            </h1>
            {confidence_text}
            <p style="margin:0;color:#555;">
                Kết quả chỉ phục vụ mục đích học tập và demo, không thay thế xét nghiệm y tế.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title(APP_TITLE)
    st.caption("Demo nhận diện nhóm máu từ ảnh khay thử bằng xử lý ảnh và machine learning.")

    model_files = find_model_files()
    if not model_files:
        st.error("Không tìm thấy file .joblib nào trong thư mục models hoặc các thư mục con.")
        st.stop()

    selected_model_path, show_masks, show_features = render_sidebar(model_files)

    try:
        model_package, model_path = load_model_package(str(selected_model_path))
    except Exception as exc:
        st.error(f"Không tải được model: {exc}")
        st.stop()

    model = model_package["model"]
    feature_columns = model_package["feature_columns"]
    render_model_info(model_package, model_path)

    st.info("Ứng dụng này là bản demo nghiên cứu. Không sử dụng kết quả để ra quyết định chẩn đoán hay điều trị.")

    uploaded_file = st.file_uploader(
        "Tải lên ảnh khay thử nhóm máu",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.write("Hãy tải lên ảnh rõ nét, ánh sáng ổn định và nhìn thấy đầy đủ các vùng phản ứng.")
        return

    try:
        rgb = decode_uploaded_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.subheader("Ảnh đầu vào")
        st.image(rgb, use_container_width=True)

    with right_col:
        st.subheader("Sẵn sàng phân tích")
        st.write(f"Kích thước ảnh: `{rgb.shape[1]} x {rgb.shape[0]}` px")
        run_prediction = st.button("Phân tích và dự đoán", type="primary", use_container_width=True)

    if not run_prediction:
        return

    with st.spinner("Đang phân đoạn màu, trích xuất feature và dự đoán..."):
        try:
            input_df, masks = extract_features_from_image(rgb, feature_columns)
            prediction = str(model.predict(input_df)[0])
            top_df = top_predictions(model, input_df)
        except Exception as exc:
            st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {exc}")
            return

    result_tab, mask_tab, feature_tab = st.tabs(["Kết quả", "Mask", "Feature"])
    with result_tab:
        render_prediction_card(prediction, top_df)
        confidence = prediction_confidence(top_df)
        if confidence is not None:
            st.metric("Độ tin cậy của lần dự đoán", f"{confidence:.2%}")
            st.caption("Điểm này là xác suất model gán cho kết quả vừa dự đoán, không phải accuracy kiểm thử tổng thể.")
        if top_df is not None:
            st.subheader("Top dự đoán")
            st.dataframe(
                top_df.assign(do_tin_cay=top_df["do_tin_cay"].map(lambda value: f"{value:.2%}")),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("Model hiện tại không hỗ trợ predict_proba nên app không hiển thị độ tin cậy.")

    with mask_tab:
        if show_masks:
            render_mask_debug(masks)
        else:
            st.caption("Bật tùy chọn 'Hiển thị mask phân đoạn' trong sidebar để xem mask.")

    with feature_tab:
        if show_features:
            st.dataframe(input_df, use_container_width=True)
        else:
            st.caption("Bật tùy chọn 'Hiển thị bảng feature' trong sidebar để xem feature đầu vào model.")


if __name__ == "__main__":
    main()
