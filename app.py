import streamlit as st
import joblib
import pandas as pd
import numpy as np
import cv2
import math
from pathlib import Path

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="Hệ Thống Dự Đoán Nhóm Máu 🩸",
    page_icon="🩸",
    layout="centered"
)

# --- TẢI MÔ HÌNH VÀ CẤU HÌNH (Dùng Cache để chạy mượt hơn) ---
@st.cache_resource
def load_blood_group_model():
    # Tìm file .joblib mới nhất trong thư mục models/
    model_dir = Path("models")
    joblib_files = list(model_dir.glob("*.joblib"))
    
    if not joblib_files:
        st.error("❌ Không tìm thấy file mô hình .joblib nào trong thư mục models/!")
        return None
        
    # Lấy file được sửa đổi gần nhất (hoặc bạn có thể điền thẳng tên file cố định vào đây)
    latest_model_path = max(joblib_files, key=lambda p: p.stat().st_mtime)
    return joblib.load(latest_model_path), latest_model_path

# Nạp model package đã lưu từ Notebook 3
package_data = load_blood_group_model()

if package_data is not None:
    model_package, model_file_name = package_data
    loaded_model = model_package["model"]
    feature_columns = model_package["feature_columns"]
    target_name = model_package["target"]
    saved_time = model_package.get("saved_at", "N/A")

    # --- GIAO DIỆN CHÍNH ---
    st.title("🩸 Hệ Thống Nhận Diện & Dự Đoán Nhóm Máu")
    st.markdown("---")
    
    # Hiển thị thông tin mô hình hiện tại ở sidebar
    st.sidebar.header("⚙️ Thông tin mô hình")
    st.sidebar.info(f"**Thuật toán:** {model_package['model_name'].upper()}\n\n"
                    f"**Số lượng đặc trưng:** {len(feature_columns)} features\n\n"
                    f"**File đang dùng:** `{model_file_name.name}`")

    # --- HÀNG HELPER FUNCTIONS (Từ Notebook 2) ---
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
            gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:],
            gray[1:-1, 2:], gray[2:, 2:], gray[2:, 1:-1],
            gray[2:, :-2], gray[1:-1, :-2],
        ]
        lbp = np.zeros_like(center, dtype=np.uint8)
        for i, n in enumerate(neighbors):
            lbp |= ((n >= center).astype(np.uint8) << i)
        return lbp

    def lbp_features(gray: np.ndarray, mask: np.ndarray, prefix: str, bins: int = 16) -> dict:
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return {f"{prefix}_lbp_hist_{i}": 0.0 for i in range(bins)} | {f"{prefix}_lbp_entropy": 0.0}
        lbp = lbp_image(gray)
        inner_mask = mask[1:-1, 1:-1]
        values = lbp[inner_mask]
        feats = normalized_histogram(values, bins=bins, value_range=(0, 256), prefix=f"{prefix}_lbp")
        feats[f"{prefix}_lbp_entropy"] = entropy_from_values(values, bins=bins, value_range=(0, 256))
        return feats

    def glcm_features(gray: np.ndarray, mask: np.ndarray, prefix: str, levels: int = 16) -> dict:
        if gray.shape[0] < 2 or gray.shape[1] < 2 or mask.sum() < 2:
            return {
                f"{prefix}_glcm_contrast": 0.0,
                f"{prefix}_glcm_homogeneity": 0.0,
                f"{prefix}_glcm_energy": 0.0,
                f"{prefix}_glcm_entropy": 0.0,
            }
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
            return {
                f"{prefix}_glcm_contrast": 0.0,
                f"{prefix}_glcm_homogeneity": 0.0,
                f"{prefix}_glcm_energy": 0.0,
                f"{prefix}_glcm_entropy": 0.0,
            }
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
            return {"component_count": 0, "largest_component_area": 0, "largest_component_ratio": 0.0}
        largest = max(areas)
        return {
            "component_count": len(areas),
            "largest_component_area": largest,
            "largest_component_ratio": largest / mask.size,
        }

    def contour_shape_features(mask: np.ndarray, prefix: str) -> dict:
        mask_u8 = (mask.astype(np.uint8) * 255)
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
        circularity = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
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

    def segment_image(rgb: np.ndarray) -> dict:
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
        masks["reaction_candidate"] = masks["foreground"] & (masks["blood_red"] | masks["anti_a_blue"] | masks["anti_b_yellow"])
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
            for name in ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std", "h_mean", "s_mean", "v_mean", "h_std", "s_std", "v_std", "gray_mean", "gray_std", "edge_density", "gradient_mean", "gradient_std", "entropy"]:
                result[f"{prefix}_{name}"] = 0.0
            result.update(normalized_histogram(np.array([]), bins=12, value_range=(0, 180), prefix=f"{prefix}_hue"))
            result.update(normalized_histogram(np.array([]), bins=8, value_range=(0, 256), prefix=f"{prefix}_sat"))
            result.update(normalized_histogram(np.array([]), bins=8, value_range=(0, 256), prefix=f"{prefix}_val"))
            result.update(lbp_features(gray, mask, prefix))
            result.update(glcm_features(gray, mask, prefix))
            return result

        result.update({
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
        })
        result.update(normalized_histogram(values_hsv[:, 0], bins=12, value_range=(0, 180), prefix=f"{prefix}_hue"))
        result.update(normalized_histogram(values_hsv[:, 1], bins=8, value_range=(0, 256), prefix=f"{prefix}_sat"))
        result.update(normalized_histogram(values_hsv[:, 2], bins=8, value_range=(0, 256), prefix=f"{prefix}_val"))
        result.update(lbp_features(gray, mask, prefix))
        result.update(glcm_features(gray, mask, prefix))
        return result

    # --- HÀM CHÍNH CHO STREAMLIT ---
    def extract_features_from_uploaded_image(uploaded_file, expected_features):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]
        max_side = 1200
        scale = max(h, w) / max_side
        if scale > 1:
            rgb = cv2.resize(rgb, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)

        features = {
            "resized_width": rgb.shape[1],
            "resized_height": rgb.shape[0],
        }

        masks = segment_image(rgb)

        st.write("### 🔍 Màn hình Debug: Các vùng màu máy tính nhận diện được")
        cols = st.columns(len(masks))
        for idx, mask_name in enumerate(masks.keys()):
            debug_mask = (masks[mask_name].astype(np.uint8) * 255)
            cols[idx].image(debug_mask, caption=f"Mask: {mask_name}", use_container_width=True)
        st.markdown("---")

        for mask_name, mask in masks.items():
            features.update(features_for_mask(rgb, mask, mask_name))

        df_features = pd.DataFrame([features])

        for col in expected_features:
            if col not in df_features.columns:
                df_features[col] = 0.0

        df_features = df_features[expected_features]
        return df_features

    # --- KHU VỰC UPLOAD VÀ XỬ LÝ ---
    uploaded_file = st.file_uploader(
        "Tải lên hình ảnh khay thử nghiệm nhóm máu (JPG, JPEG, PNG)...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Hiển thị ảnh người dùng vừa upload
        st.image(uploaded_file, caption="📷 Ảnh khay mẫu máu đã tải lên", use_container_width=True)
        
        # Thêm nút bấm kích hoạt dự đoán
        if st.button("🔮 Tiến hành phân tích và dự đoán", type="primary"):
            with st.spinner("Đang chạy phân tách màu và trích xuất đặc trưng kết cấu..."):
                try:
                    # 1. Trích xuất đặc trưng từ ảnh
                    input_df = extract_features_from_uploaded_image(uploaded_file, feature_columns)
                    
                    st.write(input_df)
                    # 2. Đưa vào mô hình SVM (.joblib) đã load để dự đoán
                    prediction = loaded_model.predict(input_df)[0]
                    
                    # 3. Hiển thị kết quả trực quan
                    st.success("🎉 Quá trình phân tích hoàn tất!")
                    
                    # Bo góc hiển thị kết quả lớn sinh động
                    st.markdown(
                        f"""
                        <div style="background-color:#ffe6e6; padding:20px; border-radius:10px; border-left: 8px solid #cc0000; text-align:center">
                            <h3 style="color:#cc0000; margin:0;">KẾT QUẢ DỰ ĐOÁN NHÓM MÁU</h3>
                            <h1 style="color:#800000; font-size:70px; margin:10px 0;">{prediction}</h1>
                            <p style="color:#555; font-size:14px; margin:0;">Mô hình phân tích dựa trên 382 đặc trưng hình thái và phân rã màu sắc.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Đã xảy ra lỗi trong quá trình xử lý: {e}")
                    
    else:
        st.info("💡 Vui lòng cung cấp hình ảnh khay mẫu máu sạch, rõ nét để thuật toán đạt độ chính xác tối ưu (Baseline SVM: ~93.9%).")