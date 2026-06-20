# Document EDA - Blood Group Detection Dataset

## 1. Mục tiêu EDA

EDA (Exploratory Data Analysis) được thực hiện để hiểu dữ liệu trước khi đi vào hai bước chính của đồ án:

- Feature extraction: trích xuất đặc trưng từ ảnh hoặc từ vùng phản ứng.
- Classification: phân loại nhóm máu dựa trên các đặc trưng đã trích xuất.

Dataset hiện tại được export từ Roboflow theo định dạng COCO. Vì vậy, dữ liệu không được chia sẵn thành folder `A+`, `A-`, `B+`, ... mà gồm ảnh và file annotation JSON trong từng split.

## 2. Cấu trúc dataset

Dataset gồm ba folder chính:

```text
train/
  images...
  _annotations.coco.json

valid/
  images...
  _annotations.coco.json

test/
  images...
  _annotations.coco.json
```

File `_annotations.coco.json` chứa:

- `images`: thông tin ảnh, gồm `file_name`, `width`, `height`, `id`.
- `annotations`: thông tin bbox của vùng phản ứng.
- `categories`: danh sách category trong COCO.

Các category trong JSON là:

```text
A
B
D
```

Ý nghĩa:

- `A`: vùng phản ứng với anti-A.
- `B`: vùng phản ứng với anti-B.
- `D`: vùng phản ứng Rh/anti-D.

Nhóm máu đầy đủ như `A+`, `B-`, `AB+`, `O-` không nằm trực tiếp trong category COCO, mà được suy ra từ tên file ảnh.

## 3. Thống kê số lượng ảnh và annotation

Kết quả khảo sát nhanh:

| Split | Số ảnh | Số annotation |
|---|---:|---:|
| train | 1386 | 2157 |
| valid | 446 | 998 |
| test | 240 | 383 |

Nhận xét:

- `train` chiếm phần lớn dữ liệu, hợp lý cho huấn luyện.
- `valid` và `test` nhỏ hơn, dùng để kiểm tra và đánh giá.
- Số annotation không bằng số ảnh vì mỗi ảnh có thể có 0, 1, 2 hoặc 3 vùng phản ứng được đánh dấu.

## 4. Phân bố nhóm máu suy ra từ tên file

Nhóm máu được parse từ phần đầu tên ảnh, ví dụ:

```text
A+ (12)_jpg.rf....
B- (8)_jpg.rf....
AB+ (65)_jpg.rf....
```

Phân bố hiện tại:

| Split | Phân bố nhóm máu |
|---|---|
| train | A+: 299, O+: 299, B-: 298, B+: 150, A-: 128, AB+: 110, AB-: 100, O-: 2 |
| valid | AB-: 185, B+: 149, AB+: 111, O-: 1 |
| test | A-: 168, AB+: 71, A+: 1 |

Nhận xét quan trọng:

- Dữ liệu bị lệch mạnh theo split.
- `train` có nhiều nhóm máu hơn, nhưng `O-` gần như không có.
- `valid` chỉ có `AB-`, `B+`, `AB+`, gần như không có các nhóm còn lại.
- `test` gần như chỉ có `A-` và `AB+`.

Điều này rất quan trọng cho classification: nếu train/test không đại diện đủ cho 8 nhóm máu, kết quả đánh giá cuối cùng có thể không phản ánh đúng năng lực mô hình.

## 5. Phân bố annotation A/B/D

Số bbox theo category:

| Split | A | B | D |
|---|---:|---:|---:|
| train | 640 | 660 | 857 |
| valid | 297 | 442 | 259 |
| test | 239 | 71 | 73 |

Nhận xét:

- Category `A`, `B`, `D` không cân bằng giữa các split.
- `test` có rất nhiều annotation `A`, nhưng ít `B` và `D`.
- Điều này phù hợp với việc `test` chủ yếu là `A-` và `AB+`.

## 6. Kiểm tra logic annotation theo nhóm máu

Ta có thể đối chiếu nhóm máu với vùng phản ứng kỳ vọng:

| Nhóm máu | Vùng phản ứng kỳ vọng |
|---|---|
| A+ | A, D |
| A- | A |
| B+ | B, D |
| B- | B |
| AB+ | A, B, D |
| AB- | A, B |
| O+ | D |
| O- | Không có A/B/D dương tính |

Notebook có cell so sánh `expected_set` và `observed_set` cho từng ảnh. Mục tiêu là phát hiện:

- Ảnh thiếu bbox.
- Ảnh có bbox dư.
- Ảnh có trùng category.
- Ảnh có annotation không khớp với tên nhóm máu.

Phần này rất hữu ích trước khi tạo feature table, vì classifier sẽ học sai nếu annotation hoặc label bị lệch.

## 7. Kích thước ảnh và bbox

Kết quả nhanh:

| Split | Width range | Height range | Median bbox area ratio | Invalid bbox |
|---|---|---|---:|---:|
| train | 265 - 3790 | 82 - 2082 | 0.1834 | 0 |
| valid | 347 - 3773 | 107 - 1325 | 0.1787 | 0 |
| test | 1096 - 3159 | 460 - 1388 | 0.1665 | 0 |

Nhận xét:

- Không có bbox invalid trong khảo sát nhanh.
- Bbox chiếm khoảng 16-18% diện tích ảnh theo median, đủ lớn để crop và trích xuất đặc trưng.
- Kích thước ảnh thay đổi khá rộng, nên khi trích xuất feature nên crop theo bbox và resize crop về kích thước chuẩn nếu dùng feature dạng pixel/histogram.

## 8. Liên hệ với feature extraction

Notebook có cell tạo bảng feature sơ bộ ở cấp crop. Mỗi bbox crop có các đặc trưng ban đầu:

- `r_mean`, `g_mean`, `b_mean`
- `r_std`, `g_std`, `b_std`
- `gray_mean`, `gray_std`
- `edge_density`
- `bbox_area_ratio`

Các feature này chỉ là baseline. Sau này có thể mở rộng thêm:

- Color histogram.
- HSV features.
- Texture features như LBP, GLCM.
- Entropy.
- Edge/contour density.
- Blob count để mô tả hiện tượng ngưng kết.

## 9. Output từ notebook

Khi chạy notebook, các file CSV sẽ được lưu vào folder:

```text
eda_outputs/
```

Các file chính:

```text
images_table.csv
annotations_table.csv
overview.csv
blood_group_counts.csv
annotation_counts.csv
quality_summary.csv
observed_patterns.csv
crop_level_features.csv
```

Trong đó:

- `images_table.csv`: bảng thông tin từng ảnh.
- `annotations_table.csv`: bảng thông tin từng bbox.
- `crop_level_features.csv`: bảng đặc trưng cơ bản từ các crop A/B/D.

## 10. Kết luận EDA

Dataset hiện tại nên được giữ nguyên theo cấu trúc COCO, không nên move ảnh gốc sang các folder `A+`, `A-`, ... vì JSON đang liên kết trực tiếp với tên ảnh trong từng split.

Hướng xử lý phù hợp cho đồ án:

```text
Giữ dataset gốc
-> đọc COCO JSON
-> crop vùng A/B/D
-> trích xuất đặc trưng
-> gom feature theo từng ảnh
-> train classifier để dự đoán nhóm máu
```

Điểm cần lưu ý lớn nhất là phân bố split hiện tại bị lệch. Nếu mục tiêu là đánh giá classifier cho đủ 8 nhóm máu, nên cân nhắc tạo lại split cân bằng hơn từ toàn bộ dataset hoặc ít nhất ghi rõ hạn chế này trong báo cáo.
