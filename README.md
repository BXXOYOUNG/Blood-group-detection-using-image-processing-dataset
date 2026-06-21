# Blood Group Detection Using Image Processing Dataset

Project này xây dựng một pipeline phân loại nhóm máu từ ảnh khay thử bằng xử lý ảnh và machine learning truyền thống.  
Luồng chính là:

`dataset COCO -> EDA -> segmentation -> feature extraction -> train model -> Streamlit inference`

Mục tiêu của repo là:

- Tách vùng phản ứng từ ảnh.
- Trích xuất feature thủ công từ màu sắc, texture, tần số và hình thái.
- Huấn luyện nhiều model cổ điển.
- Cho phép dự đoán trực tiếp trên Streamlit và chọn model linh hoạt.

## 1. Cách dự án hoạt động

1. Dataset gốc được giữ theo cấu trúc COCO trong `train/`, `valid/`, `test/`.
2. Các notebook EDA và feature extraction tạo ra CSV feature trong `processed/`.
3. Notebook classification train nhiều model và lưu vào `models/`.
4. `app.py` đọc các model đã lưu, cho user upload ảnh, trích feature cùng logic với notebook, rồi dự đoán nhóm máu.

Đây là pipeline classical ML, không phải end-to-end deep learning.

## 2. Cấu trúc thư mục và file

### 2.1. File chính ở thư mục gốc

| File | Vai trò |
|---|---|
| `app.py` | Ứng dụng Streamlit. Tự tìm model trong `models/`, cho upload ảnh, segment, trích feature, dự đoán và hiển thị kết quả. |
| `01_eda_blood_group.ipynb` | EDA dataset, kiểm tra phân bố ảnh, annotation và xuất bảng thống kê ra `eda_outputs/`. |
| `02_color_segmentation_feature_extraction.ipynb` | Prototype phân đoạn màu và trích feature theo mask. Sinh file `processed/color_segmentation/color_segmentation_features.csv`. |
| `03_classification_model_training.ipynb` | Notebook train các model phân loại dựa trên feature đã trích xuất. |
| `03_classification_model_training_updated.ipynb` | Phiên bản cập nhật của notebook train, dùng để chạy lại hoặc tinh chỉnh pipeline huấn luyện. |
| `04_frequency_domain_feature_extraction.ipynb` | Trích feature miền tần số bằng FFT và DoG. Sinh file trong `processed/frequency_domain/`. |
| `05_morphological_spatial_feature_extraction.ipynb` | Trích feature hình thái học và phân bố không gian. Sinh file trong `processed/morphological_spatial/`. |
| `EDA_Blood_Group_Document.md` | Tài liệu giải thích phần EDA của dataset. |
| `Color_Segmentation_Feature_Extraction_Document.md` | Tài liệu giải thích segmentation và feature extraction theo màu. |
| `Classification_Model_Training_Document.md` | Tài liệu giải thích phần train model và lưu model package. |
| `Frequency_Domain_Feature_Extraction_Document.md` | Tài liệu giải thích feature miền tần số. |
| `Morphological_Spatial_Feature_Extraction_Document.md` | Tài liệu giải thích feature hình thái học và phân bố không gian. |
| `README.md` | Tài liệu tổng quan của dự án này. |

### 2.2. Dataset gốc

| Folder | Nội dung |
|---|---|
| `train/` | Split train của dataset, gồm ảnh và file `_annotations.coco.json`. |
| `valid/` | Split validation của dataset, gồm ảnh và file `_annotations.coco.json`. |
| `test/` | Split test của dataset, gồm ảnh và file `_annotations.coco.json`. |

Dataset đang ở định dạng COCO. Các annotation trong JSON mô tả vùng phản ứng A/B/D, còn nhóm máu đầy đủ được suy ra từ tên file ảnh theo logic của project.

### 2.3. Ảnh demo

| Folder | Nội dung |
|---|---|
| `demo/` | Ảnh mẫu để test nhanh các nhóm máu như `A+`, `A-`, `B+`, `B-`, `AB+`, `AB-`, `O+`, `O-`. |

### 2.4. Kết quả trung gian

| Folder | Nội dung |
|---|---|
| `processed/` | Output do các notebook sinh ra, gồm CSV feature và ảnh preview. |

Chi tiết các subfolder trong `processed/`:

| Subfolder | File sinh ra | Ý nghĩa |
|---|---|---|
| `processed/color_segmentation/` | `color_segmentation_features.csv`, `segmentation_preview_samples.png` | Feature từ phân đoạn màu và ảnh preview mask. |
| `processed/frequency_domain/` | `frequency_domain_features.csv`, `frequency_domain_preview.png`, `frequency_feature_distributions.png` | Feature miền tần số và các ảnh kiểm tra phân bố feature. |
| `processed/morphological_spatial/` | `morphological_spatial_features.csv`, `morphological_spatial_preview.png`, `morphological_feature_distributions.png` | Feature hình thái học và phân bố không gian, kèm ảnh preview. |

### 2.5. Kết quả EDA

| Folder | Nội dung |
|---|---|
| `eda_outputs/` | Các bảng thống kê và kiểm tra dữ liệu sinh từ notebook EDA. |

Các file thường có trong `eda_outputs/`:

- `images_table.csv`
- `annotations_table.csv`
- `overview.csv`
- `blood_group_counts.csv`
- `annotation_counts.csv`
- `quality_summary.csv`
- `observed_patterns.csv`
- `crop_level_features.csv`

### 2.6. Thư mục model

| Folder | Vai trò |
|---|---|
| `models/` | Chứa toàn bộ model đã train, metric và report. `app.py` sẽ quét đệ quy các file `.joblib` trong đây để cho user chọn model. |

Cấu trúc logic bên trong `models/`:

| Subfolder | Loại feature |
|---|---|
| `all_blood_group_classification/` | Tập feature tổng hợp. |
| `color_blood_group_classification/` | Feature màu. |
| `color_segmentation_blood_group_classification/` | Feature từ color segmentation. |
| `color_frequency_blood_group_classification/` | Kết hợp color + frequency. |
| `color_morphological_blood_group_classification/` | Kết hợp color + morphological. |
| `frequency_blood_group_classification/` | Feature miền tần số. |
| `morphological_blood_group_classification/` | Feature hình thái học. |

Mỗi subfolder thường có:

- `*.joblib`: model package dùng cho inference.
- `*.pkl`: bản serialize bổ sung/legacy.
- `*_metrics_*.csv`: bảng metric.
- `*_classification_report_*.txt`: báo cáo đánh giá.

### 2.7. Thư mục tham chiếu

| Folder | Nội dung |
|---|---|
| `ref/` | Thư mục tham chiếu, hiện đang để trống. |

### 2.8. Thư mục phụ do Python sinh ra

| Folder | Nội dung |
|---|---|
| `__pycache__/` | Cache bytecode của Python, không phải dữ liệu dự án. |

## 3. Nội dung từng notebook

### `01_eda_blood_group.ipynb`

Notebook này dùng để hiểu dataset trước khi train:

- Đọc COCO JSON trong `train/`, `valid/`, `test/`.
- Thống kê số ảnh, số annotation và phân bố nhóm máu.
- Kiểm tra mức lệch lớp giữa các split.
- Xuất bảng EDA ra `eda_outputs/`.

### `02_color_segmentation_feature_extraction.ipynb`

Notebook này là nền tảng cho phần trích feature từ ảnh:

- Tạo mask theo màu và foreground.
- Các mask chính là `anti_a_blue`, `anti_b_yellow`, `blood_red`, `foreground`, `reaction_candidate`.
- Trích feature trong từng mask từ màu, histogram, texture, edge, contour và blob.
- Sinh `processed/color_segmentation/color_segmentation_features.csv`.

### `03_classification_model_training.ipynb`

Notebook này đọc feature CSV và train các model classical ML:

- Chọn cột numeric làm feature.
- Train nhiều model khác nhau.
- So sánh bằng metric phù hợp với dữ liệu lệch lớp.
- Lưu model, metric và classification report vào `models/`.

### `03_classification_model_training_updated.ipynb`

Đây là bản cập nhật của notebook train. Mục đích là chạy lại pipeline huấn luyện hoặc tinh chỉnh so với bản gốc.

### `04_frequency_domain_feature_extraction.ipynb`

Notebook này mở rộng feature sang miền tần số:

- Dùng FFT 2D để lấy năng lượng theo vòng tần số.
- Dùng DoG để bắt blob ở nhiều thang đo.
- Sinh file trong `processed/frequency_domain/`.

### `05_morphological_spatial_feature_extraction.ipynb`

Notebook này bổ sung feature về hình thái học và phân bố không gian:

- Erosion, dilation, shell và thickness.
- Hu Moments.
- Centroid, radial distribution, angular distribution, bbox features.
- Sinh file trong `processed/morphological_spatial/`.

## 4. `app.py` hoạt động như thế nào

`app.py` là Streamlit app dùng để inference với model đã train sẵn.

### 4.1. Tìm model

- App quét `models/**/*.joblib`.
- Các model được đưa vào dropdown ở sidebar.
- Người dùng có thể chọn tự do model muốn dùng để dự đoán.

### 4.2. Load model package

Mỗi file `.joblib` được load bằng `joblib.load()`.

App kiểm tra model package phải có tối thiểu:

- `model`
- `feature_columns`
- `target`

Nếu model package có thêm `metrics`, `model_name`, `test_accuracy` thì app sẽ hiển thị thêm thông tin đó.

### 4.3. Xử lý ảnh upload

Khi user upload ảnh:

- Ảnh được decode từ `jpg/jpeg/png`.
- Ảnh được resize nếu cạnh lớn nhất quá lớn.
- Ảnh được chuyển sang RGB để segment và trích feature.

### 4.4. Segment và trích feature

App tái sử dụng đúng logic feature extraction của project:

- Tạo mask màu.
- Làm sạch mask bằng morphological operations.
- Trích feature từ từng mask.
- Ghép vào một `DataFrame` 1 dòng.
- Reindex theo đúng `feature_columns` của model đang chọn.

### 4.5. Dự đoán

Sau khi có feature input:

- App gọi `model.predict()` để lấy nhãn dự đoán.
- Nếu model hỗ trợ `predict_proba`, app hiển thị xác suất top class.
- Nếu model không có `predict_proba`, app dùng `decision_function` rồi chuẩn hóa để vẫn hiển thị độ tin cậy tương đối.

### 4.6. Hiển thị trên UI

App có 3 tab:

- `Kết quả`
- `Mask`
- `Feature`

Kết quả hiển thị gồm:

- Nhóm máu dự đoán.
- Độ tin cậy của lần dự đoán.
- Top predictions.
- Mask phân đoạn.
- Bảng feature đầu vào nếu bật chế độ xem feature.

## 5. `accuracy` và `confidence` khác nhau thế nào

Đây là điểm quan trọng:

- `Test accuracy` trong sidebar là accuracy của model trên dữ liệu kiểm tra, lấy từ metrics/report đã lưu khi train.
- `Độ tin cậy của lần dự đoán` là xác suất mà model gán cho ảnh đang upload, tức confidence cho một mẫu duy nhất.

Hai giá trị này không giống nhau.

- `accuracy` dùng để đánh giá model trên tập test.
- `confidence` dùng để xem model tự tin đến mức nào với ảnh hiện tại.

## 6. Cách chạy project

Từ thư mục `Blood-group-detection-using-image-processing-dataset/`, chạy:

```bash
streamlit run app.py
```

Điều kiện để chạy đúng:

- Giữ `app.py` và `models/` trong cùng thư mục gốc của project.
- Các model phải nằm trong `models/` dưới dạng `.joblib`.
- Ảnh upload nên là `jpg`, `jpeg` hoặc `png`.

## 7. Thư viện chính

Các thư viện chính được dùng trong app và notebook gồm:

- `streamlit`
- `opencv-python`
- `numpy`
- `pandas`
- `joblib`
- `scikit-learn`
- `scipy`

Nếu chạy notebook, thường cần thêm:

- `jupyter`
- `matplotlib`
- `seaborn`

## 8. Khi thêm model mới

Nếu muốn thêm model mới vào app:

1. Train model với cùng format feature.
2. Lưu package vào `models/` dưới dạng `.joblib`.
3. Đảm bảo package có `model`, `feature_columns`, `target`.
4. Nếu có thêm `metrics` và `test_accuracy`, app sẽ hiển thị đầy đủ hơn.

## 9. Ghi chú quan trọng

- Đây là project demo/nghiên cứu, không thay thế xét nghiệm y tế.
- `processed/` và `eda_outputs/` là dữ liệu sinh ra từ notebook, không nên sửa tay.
- `ref/` chứa tài liệu tham khảo.
- `__pycache__/` chỉ là cache của Python, không cần quan tâm khi dùng project.
