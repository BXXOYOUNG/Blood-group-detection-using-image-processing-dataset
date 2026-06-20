# Morphological & Spatial Distribution Feature Extraction Document

## 1. Lý do chọn phương pháp hình thái học và phân bố không gian

Notebook 02 đã trích texture (LBP, GLCM), màu sắc (RGB/HSV histogram), và một số đặc trưng hình dạng cơ bản (contour circularity, component count). Tuy nhiên vẫn còn hai khía cạnh quan trọng chưa được khai thác:

**1. Độ dày và cấu trúc lõi của vùng phản ứng:**
Ngưng kết mạnh tạo blob đặc (eroded area còn nhiều), ngưng kết yếu tạo blob mỏng (eroded area gần bằng 0). Contour circularity chỉ nắm hình dạng bên ngoài; morphological profile nắm được cấu trúc bên trong.

**2. Vị trí và hình dạng tổng thể bất biến hình học:**
Hai ảnh cùng nhóm máu có thể chụp ở góc khác nhau, khoảng cách khác nhau. Hu Moments bất biến với tịnh tiến, xoay và tỷ lệ → mô tả hình dạng mà không bị ảnh hưởng bởi cách chụp. Phân bố bán kính và góc cho biết mask đặc ở trung tâm hay phân tán ra ngoài.

Notebook thực hiện phương pháp này:

```text
05_morphological_spatial_feature_extraction.ipynb
```

Output chính:

```text
processed/morphological_spatial/morphological_spatial_features.csv
```

## 2. Hai phương pháp chính

### 2.1. Morphological Profile (Erosion/Dilation Đa Thang Đo)

**Ý tưởng cốt lõi**: Áp dụng erosion và dilation với kernel tròn ở nhiều bán kính (R = 2, 4, 8, 16 px). Theo dõi cách mask thay đổi khi "mài mòn" dần từ ngoài vào trong (erosion) hoặc "phình" ra (dilation) cho biết cấu trúc lớp của blob.

**Kernel sử dụng:**

```python
cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*R+1, 2*R+1))
```

Kernel hình ellipse phù hợp hơn kernel vuông vì blob ngưng kết thường có dạng tròn/không đều.

**Feature trích xuất cho mỗi bán kính R:**

| Feature | Ý nghĩa |
|---|---|
| `*_erode_r{R}_ratio` | Tỷ lệ mask còn lại sau erosion R px → đo "lõi đặc" |
| `*_dilate_r{R}_ratio` | Tỷ lệ mask sau dilation R px → đo "vùng ảnh hưởng" |
| `*_shell_r{R}_ratio` | Tỷ lệ vùng giữa dilated và eroded → đo "lớp vỏ" |
| `*_thickness_score_r{R}` | eroded / original area → blob dày (≈1) hay mỏng (≈0) |
| `*_fill_ratio_r{R}` | eroded / dilated → blob đặc (≈1) hay rỗng (≈0) |

**Feature bổ sung:**

- `*_morph_gradient_ratio`: Diện tích morphological gradient (dilated − eroded tại R=2) / image_area → đo độ "viền".
- `*_open_close_diff`: Hiệu diện tích giữa opened và closed mask tại kernel 3×3 → đo độ thô/nhám của biên mask.

**Cách đọc feature:**

```text
thickness_score_r2 ≈ 1.0 → mask rất dày (blob đặc)
thickness_score_r2 ≈ 0.0 → mask mỏng (chỉ là viền hay blob nhỏ)

fill_ratio_r4 ≈ 1.0 → eroded chiếm hầu hết dilated → blob đặc tròn
fill_ratio_r4 ≈ 0.0 → chỉ còn viền, không có lõi

shell_r8_ratio cao → phần lớn mask là "vỏ" ở độ dày 8px → không có lõi đặc
```

**Ví dụ phân tích:**

```text
Anti-A dương tính mạnh (A+):
  anti_a_blue_thickness_score_r4 → gần 0.6 (blob đặc vừa)
  anti_a_blue_fill_ratio_r8 → gần 0.4

Anti-A âm tính (O+):
  anti_a_blue_mask_present = 0 → tất cả morph features = 0
```

### 2.2. Spatial Distribution Features

Nhóm này gồm ba loại đặc trưng mô tả vị trí và hình dạng tổng thể của mask trong không gian ảnh.

#### 2.2.1. Hu Moments (7 giá trị)

**Ý tưởng**: Central moments của ảnh nhị phân được chuẩn hóa để bất biến với phép tịnh tiến, xoay, và thay đổi tỷ lệ.

```text
Hu_0: Đặc trưng chính của hình dạng (related to area)
Hu_1: Độ kéo dài hình dạng
Hu_2..6: Bất đối xứng, biến dạng, độ góc cạnh
```

**Lưu ý quan trọng**: Giá trị Hu Moments thường rất nhỏ (10⁻⁵ đến 10⁻⁰). Notebook dùng log-scale để cải thiện phân tách:

```python
sign(v) × log(|v| + 1e-9)
```

Feature được lưu: `*_hu_0` đến `*_hu_6`.

**Ưu điểm trong bài toán nhóm máu:**
Hai ảnh A+ chụp ở góc khác nhau vẫn có mask anti_a_blue có Hu Moments gần giống nhau, dù contour shape features có thể khác nhau vì tư thế chụp.

#### 2.2.2. Trọng tâm tương đối (Centroid)

```text
*_centroid_x_norm = m10 / (m00 * W)  ∈ [0, 1]
*_centroid_y_norm = m01 / (m00 * H)  ∈ [0, 1]
```

Vị trí trọng tâm mask so với kích thước ảnh. Nếu giọt phản ứng thường được đặt ở một vị trí cố định trong ảnh, centroid sẽ là feature phân biệt tốt.

#### 2.2.3. Phân bố bán kính (Radial Distribution)

Chia khoảng cách từ trọng tâm mask ra 10 bin. Histogram pixel theo khoảng cách này cho biết:

- Blob đặc: histogram đỉnh ở các bin gần trung tâm.
- Blob viền/rỗng: histogram đỉnh ở bin xa.
- Blob phân tán: histogram phẳng.

Features:

- `*_radial_bin{0..9}`: Tỷ lệ pixel trong mỗi bin khoảng cách.
- `*_radial_max_bin`: Bin có nhiều pixel nhất → đỉnh phân bố.
- `*_radial_spread`: Độ lệch chuẩn khoảng cách → blob tập trung hay phân tán.

#### 2.2.4. Phân bố góc (Angular Distribution)

Chia góc 0–360° thành 8 sector (mỗi sector 45°). Histogram pixel theo góc từ trọng tâm cho biết:

- Blob đối xứng: histogram phẳng → `angular_entropy` cao.
- Blob lệch về một phía: histogram tập trung → `angular_entropy` thấp.

Features:

- `*_angular_bin{0..7}`: Tỷ lệ pixel trong mỗi sector góc.
- `*_angular_entropy`: Entropy Shannon của phân bố góc.

#### 2.2.5. Bounding Box Features

```text
*_bbox_aspect_ratio  = height / width → blob dài hay vuông
*_bbox_fill_ratio    = mask_area / bbox_area → blob đặc hay lỗ chỗ
*_bbox_area_norm     = bbox_area / image_area → kích thước tương đối
```

## 3. So sánh với các đặc trưng trong notebook 02

| Đặc trưng | Notebook 02 | Notebook 05 |
|---|---|---|
| Hình dạng contour | max_contour_circularity, perimeter | Hu Moments (bất biến hình học) |
| Diện tích | area_ratio | bbox_fill_ratio, bbox_area_norm |
| Cấu trúc bên trong | Không có | thickness_score, fill_ratio (morphological) |
| Vị trí trong ảnh | Không có | centroid_x_norm, centroid_y_norm |
| Phân bố không gian | Không có | radial_bin, angular_bin, angular_entropy |
| Độ viền | edge_density (Canny) | morph_gradient_ratio, open_close_diff |
| Số component | component_count | Không có (đã có ở notebook 02) |

## 4. Input/Output của notebook 05

**Input:**

```text
Ảnh gốc từ dataset (train/, valid/, test/)
_annotations.coco.json (chỉ để lấy danh sách file và label)
```

**Output:**

```text
processed/morphological_spatial/morphological_spatial_features.csv
processed/morphological_spatial/morphological_spatial_preview.png
processed/morphological_spatial/morphological_feature_distributions.png
```

Cột trong CSV:

- Metadata: `split`, `file_name`, `blood_group`, `abo`, `rh`, `resized_width`, `resized_height`
- Morphological: `{mask}_erode_r{R}_ratio`, `{mask}_dilate_r{R}_ratio`, `{mask}_shell_r{R}_ratio`, `{mask}_thickness_score_r{R}`, `{mask}_fill_ratio_r{R}`, `{mask}_morph_gradient_ratio`, `{mask}_open_close_diff`
- Spatial: `{mask}_hu_{0..6}`, `{mask}_centroid_x_norm`, `{mask}_centroid_y_norm`, `{mask}_radial_bin{0..9}`, `{mask}_radial_max_bin`, `{mask}_radial_spread`, `{mask}_angular_bin{0..7}`, `{mask}_angular_entropy`, `{mask}_bbox_aspect_ratio`, `{mask}_bbox_fill_ratio`, `{mask}_bbox_area_norm`

## 5. Số lượng feature chi tiết

**Morphological profile** mỗi mask:

```text
5 features × 4 radii = 20
2 bổ sung (gradient, open_close_diff) = 2
Tổng = 22
```

**Spatial distribution** mỗi mask:

```text
7 Hu Moments
2 centroid
10 radial bins + 2 (max_bin, spread) = 12
8 angular bins + 1 entropy = 9
3 bbox features
Tổng = 33
```

**Tổng mỗi mask:** 22 + 33 + 1 (mask_present) = **56 features**

**5 masks × 56 = 280 features** + metadata.

## 6. Phụ thuộc thư viện

```python
cv2.morphologyEx     # Erosion, dilation, gradient, open, close
cv2.moments          # Moments ảnh nhị phân
cv2.HuMoments        # Hu Moments từ moments
numpy                # Tính toán radial và angular histogram
```

Không cần cài thêm thư viện. Tất cả đều có sẵn trong `opencv-python` và `numpy`.

## 7. Xử lý mask rỗng

Khi mask rỗng (ví dụ `anti_a_blue` ở nhóm O+):

- Tất cả morphological features = 0.0 (erosion/dilation của rỗng vẫn là rỗng).
- Hu Moments: `cv2.moments` trả về 0 cho m00 → centroid được đặt về `(W/2, H/2)`, hu = log(1e-9) ≈ -20.7 (có thể dùng `mask_present` để phân biệt với mask thật).
- Radial và angular histograms: tất cả 0.0.
- Feature `*_mask_present = 0` đóng vai trò flag quan trọng.

## 8. Kết hợp với classifier ở notebook 03

`FEATURE_MODE = "morphological"`: Dùng độc lập

`FEATURE_MODE = "color+morphological"`: Kết hợp với feature màu/texture từ notebook 02

`FEATURE_MODE = "all"`: Kết hợp cả ba nguồn (02 + 04 + 05)

Khi merge, `file_name` được dùng làm khóa nối giữa các CSV. Cột feature không bị trùng tên vì mỗi notebook có prefix đặt tên riêng.

## 9. Điểm mạnh trong bài toán nhóm máu

**Morphological Profile** đặc biệt hữu ích vì:

- Ngưng kết dương tính mạnh → blob đặc → `thickness_score` và `fill_ratio` cao.
- Ngưng kết yếu hoặc âm tính → chỉ còn viền mỏng → `shell_ratio` cao, `fill_ratio` thấp.
- Phân biệt được phản ứng mạnh/yếu mà chỉ nhìn diện tích tổng thể không phân biệt được.

**Hu Moments** đặc biệt hữu ích khi:

- Dataset chụp từ các góc độ và tỷ lệ khác nhau.
- Bất biến hình học đảm bảo cùng một loại blob → Hu tương tự nhau bất kể cách chụp.

**Angular Entropy** đặc biệt hữu ích vì:

- Blob đối xứng (phản ứng đều): `angular_entropy` cao (~3 bit).
- Blob lệch (bọt khí, nhiễu): `angular_entropy` thấp (< 2 bit).

## 10. Hạn chế

- Hu Moments có thể nhiễu khi mask quá nhỏ (< 20 pixel).
- Radial distribution phụ thuộc vào trọng tâm; nếu mask có nhiều vùng rời rạc, trọng tâm không đại diện cho cả mask.
- Morphological profile ở R=16 có thể gây dilation quá lớn với ảnh nhỏ.
- Angular distribution không phân biệt được blob ở phần trên vs dưới nếu đối xứng qua trục ngang.

## 11. Hướng phát triển

- Thêm Zernike Moments: bất biến hình học tốt hơn Hu ở bậc cao.
- Radial Distance Transform: dùng `cv2.distanceTransform` để tính phân bố khoảng cách đến biên.
- Multi-scale blob detection: `SimpleBlobDetector` trong OpenCV để đếm và đo kích thước ngưng kết.
- Convex Hull features: tỷ lệ mask / convex hull → đo độ lõm/lồi của vùng phản ứng.
- Skeleton analysis: mỏng hóa mask và đo chiều dài, số nhánh để phân tích cấu trúc nhánh của ngưng kết.
