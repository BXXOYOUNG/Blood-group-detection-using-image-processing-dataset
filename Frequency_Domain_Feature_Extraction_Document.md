# Frequency Domain Feature Extraction Document

## 1. Lý do chọn phương pháp miền tần số

Notebook 02 đã khai thác rất tốt các đặc trưng không gian: màu sắc (RGB/HSV), texture cục bộ (LBP, GLCM), cạnh (Canny, Sobel), và hình dạng (contour). Tuy nhiên, các phương pháp đó đều phân tích pixel trong miền không gian.

Một số thông tin quan trọng về phản ứng ngưng kết lại thể hiện rõ hơn trong **miền tần số**:

- Vùng phản ứng có hạt ngưng kết nhỏ → nhiều thành phần tần cao → năng lượng FFT phân bố ở vòng bán kính lớn.
- Vùng màu đồng đều (anti-A hay anti-B rõ ràng) → chủ yếu tần thấp → năng lượng tập trung ở vòng bán kính nhỏ.
- Phân tích đa thang đo (DoG) phát hiện cấu trúc blob ở kích thước khác nhau mà histogram màu không nắm được.

Notebook thực hiện phương pháp này:

```text
04_frequency_domain_feature_extraction.ipynb
```

Output chính:

```text
processed/frequency_domain/frequency_domain_features.csv
```

## 2. Hai phương pháp chính

### 2.1. FFT 2D Ring Spectrum

**Ý tưởng cốt lõi**: Biến đổi Fourier 2D chuyển ảnh từ không gian pixel sang không gian tần số không gian. Tần số thấp (gần tâm) mô tả cấu trúc lớn; tần số cao (xa tâm) mô tả chi tiết nhỏ và cạnh.

**Quy trình xử lý:**

1. Cắt bounding-box của mask từ ảnh grayscale.
2. Resize về patch 64×64 để chuẩn hóa kích thước trước khi FFT.
3. Tính FFT 2D và dịch tâm phổ về giữa (fftshift).
4. Tính biên độ phổ.
5. Chia bán kính thành 8 vòng đồng tâm từ tâm ra ngoài.
6. Tính trung bình biên độ trong mỗi vòng.

**Feature trích xuất:**

Cho mỗi mask (ví dụ `anti_a_blue`):

- `*_fft_band{0..7}_energy`: Năng lượng trung bình trong mỗi vòng tần số.
- `*_fft_band{0..7}_ratio`: Tỷ lệ năng lượng mỗi vòng so với tổng.
- `*_fft_dc_ratio`: Tỷ lệ năng lượng hai vòng đầu (tần thấp) → đo độ đồng đều màu.
- `*_fft_high_ratio`: Tỷ lệ 4 vòng ngoài cùng (tần cao) → đo mức độ hạt/ngưng kết.
- `*_fft_spectral_centroid`: Bán kính trung bình có trọng số biên độ → tóm tắt vị trí năng lượng chủ đạo.
- `*_fft_spectral_spread`: Độ phân tán xung quanh spectral centroid.

**Ý nghĩa phân tích nhóm máu:**

```text
A+  → anti_a_blue_fft_dc_ratio cao  (màu xanh đồng đều)
B+  → anti_b_yellow_fft_dc_ratio cao (màu vàng đồng đều)
AB+ → cả hai dc_ratio đều cao
O+  → cả hai dc_ratio thấp, blood_red_fft_high_ratio cao hơn
```

### 2.2. DoG Wavelet (Difference of Gaussians)

**Ý tưởng cốt lõi**: DoG xấp xỉ Laplacian of Gaussian (LoG), là bộ lọc phát hiện blob tại một thang đo cụ thể. Dùng 4 thang đo khác nhau cho phép phát hiện cả ngưng kết nhỏ lẫn cấu trúc lớn.

**Quy trình xử lý:**

1. Áp dụng Gaussian blur hai lần với `σ_fine` và `σ_coarse` (σ_coarse ≈ σ_fine × 1.6).
2. Lấy hiệu: `DoG = G(σ_fine) − G(σ_coarse)`.
3. Lấy pixel trong vùng mask để phân tích.

**Bốn thang đo:**

| Thang đo | σ_fine | σ_coarse | Phát hiện cấu trúc |
|---|---|---|---|
| s010 | 1.0 | 1.6 | Hạt rất nhỏ, pixel lẻ |
| s020 | 2.0 | 3.2 | Cụm nhỏ (1–3 px) |
| s040 | 4.0 | 6.4 | Blob vừa (5–10 px) |
| s080 | 8.0 | 12.8 | Blob lớn, vùng phản ứng rộng |

**Feature trích xuất** (mỗi thang đo, mỗi mask):

- `*_dog_{tag}_mean`: Giá trị trung bình phản hồi → vùng sáng hay tối hơn nền cục bộ.
- `*_dog_{tag}_std`: Độ biến thiên → mức không đồng đều.
- `*_dog_{tag}_energy`: Tổng bình phương → mức độ có cấu trúc tại thang đó.
- `*_dog_{tag}_peak_ratio`: Tỷ lệ pixel có |resp| > 1σ → mật độ điểm nổi bật.
- `*_dog_{tag}_neg_ratio`: Tỷ lệ phản hồi âm → vùng tối hơn nền (hữu ích với mask máu).
- `*_dog_fine_coarse_ratio`: Tỷ lệ energy thang nhỏ nhất / thang lớn nhất → phân biệt texture hạt nhỏ vs vùng lớn.

## 3. Tại sao dùng patch 64×64 cho FFT

FFT nhạy cảm với kích thước ảnh đầu vào. Bounding-box của mask thay đổi kích thước giữa các ảnh. Nếu so sánh trực tiếp phổ FFT của hai patch khác kích thước, các vòng tần số không tương ứng nhau.

Resize về 64×64 trước FFT đảm bảo mỗi band tần số có cùng ý nghĩa vật lý giữa tất cả ảnh.

Hạn chế: resize làm mất thông tin tỷ lệ tuyệt đối của vật thể. Đây là sự đánh đổi hợp lý vì mục tiêu là đặc trưng tần số tương đối, không phải kích thước tuyệt đối.

## 4. Mask-level extraction

Giống notebook 02, feature tần số được tính **trong từng mask riêng lẻ**, không phải toàn bộ ảnh. Lý do:

- Background trắng sẽ tạo peak tần thấp rất lớn, che lấp tín hiệu từ vùng phản ứng.
- Mỗi mask có ý nghĩa sinh học khác nhau, nên cần phân tích độc lập.

Các mask được dùng:

```text
anti_a_blue
anti_b_yellow
blood_red
foreground
reaction_candidate
```

Khi mask rỗng, tất cả feature của mask đó được đặt về `0.0` (có nghĩa là "không có tín hiệu tại mask này").

## 5. Input/Output của notebook 04

**Input:**

```text
Ảnh gốc từ dataset (train/, valid/, test/)
_annotations.coco.json (chỉ dùng để lấy danh sách file và label)
```

**Output:**

```text
processed/frequency_domain/frequency_domain_features.csv
processed/frequency_domain/frequency_domain_preview.png
processed/frequency_domain/frequency_feature_distributions.png
```

Cột trong CSV:

- Metadata: `split`, `file_name`, `blood_group`, `abo`, `rh`, `resized_width`, `resized_height`
- Feature: `{mask_name}_mask_present`, `{mask_name}_fft_*`, `{mask_name}_dog_*`

Số feature trên mỗi ảnh: khoảng **185 cột** (không tính metadata).

## 6. Số lượng feature chi tiết

**FFT features** mỗi mask:

```text
8 bands × 2 (energy, ratio)  = 16
4 summary features            =  4
Tổng mỗi mask                 = 20
```

**DoG features** mỗi mask:

```text
4 thang đo × 5 features       = 20
1 cross-scale ratio            =  1
Tổng mỗi mask                 = 21
```

**Tổng mỗi mask:** 20 + 21 + 1 (mask_present) = **42 features**

**5 masks × 42 = 210 features** + metadata.

## 7. Phụ thuộc thư viện

```python
scipy.fft.fft2      # FFT 2D
scipy.fft.fftshift  # Dịch tâm phổ
scipy.ndimage.gaussian_filter  # Gaussian blur cho DoG
```

Không cần cài thêm thư viện bên ngoài. `scipy` đã có sẵn trong môi trường chuẩn.

## 8. Kết hợp với classifier ở notebook 03

File `frequency_domain_features.csv` có cùng cấu trúc metadata với `color_segmentation_features.csv`. Có thể:

1. Dùng độc lập: đặt `FEATURE_MODE = "frequency"` trong notebook 03 cập nhật.
2. Kết hợp với feature màu: `FEATURE_MODE = "color+frequency"`.
3. Kết hợp cả ba nguồn: `FEATURE_MODE = "all"`.

## 9. Giải thích trực quan

**FFT Ring Spectrum:**

```text
Vòng 0-1 (tần thấp): Cấu trúc lớn, màu đồng đều
  → Anti-A rõ ràng: band0, band1 energy cao
Vòng 4-7 (tần cao): Cạnh, hạt nhỏ, ngưng kết
  → Phản ứng dương tính: high_ratio cao
Spectral centroid gần tâm: vùng màu mịn
Spectral centroid xa tâm: nhiều chi tiết nhỏ
```

**DoG Wavelet:**

```text
s010_energy cao: nhiều hạt rất nhỏ (ngưng kết đầu tiên)
s080_energy cao: blob lớn (vùng máu rộng)
fine_coarse_ratio cao: ngưng kết hạt nhỏ chiếm ưu thế
fine_coarse_ratio thấp: cấu trúc lớn chiếm ưu thế
```

## 10. Hạn chế

- Patch 64×64 mất thông tin tỷ lệ tuyệt đối.
- FFT nhạy với ánh sáng không đồng đều trong ảnh.
- DoG đơn kênh (grayscale) không nắm thông tin màu sắc.
- Khi mask quá nhỏ (< 5×5 px sau cắt), patch resize có thể bị artifacts.
- Không thay thế được feature màu, chỉ bổ sung thông tin cấu trúc tần số.

## 11. Hướng phát triển

- Thử 2D Gabor filter để kết hợp tần số và hướng (orientation).
- Dùng Short-Time Fourier Transform (STFT) theo hàng/cột để phân tích định hướng.
- Log-polar FFT để bất biến hơn với xoay.
- Kết hợp feature tần số với feature màu trong SVM với kernel kết hợp (multiple kernel learning).
