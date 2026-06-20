# Color Segmentation and Feature Extraction Prototype

## 1. Lý do chuyển hướng

Sau EDA, ta thấy file COCO JSON hiện tại chủ yếu đánh dấu các vùng phản ứng dương tính. Nếu dùng trực tiếp JSON để suy luận nhóm máu, bài toán sẽ bị phụ thuộc vào annotation có sẵn và không khách quan khi đưa ảnh mới vào.

Vì vậy, prototype này thử khai thác thông tin trực tiếp từ ảnh:

```text
Ảnh gốc
-> tách vùng màu/foreground
-> rút đặc trưng trong mask
-> dùng feature cho classification
```

JSON vẫn được giữ lại, nhưng chỉ dùng để lấy danh sách ảnh và label từ tên file, không dùng bbox như đáp án cuối.

## 2. Tri thức miền đang dùng

Quan sát từ dữ liệu:

- Anti-A có màu xanh.
- Anti-B có màu vàng.
- Anti-D không màu hoặc ít đặc trưng màu, nên khó tách chỉ bằng màu thuốc thử.
- Background có thể ảnh hưởng mạnh nếu trích feature toàn ảnh.

Do đó, hướng xử lý là dùng màu để tạo mask, rồi chỉ rút đặc trưng trong các vùng mask thay vì toàn bộ ảnh.

## 3. Notebook triển khai

Notebook:

```text
02_color_segmentation_feature_extraction.ipynb
```

Output chính:

```text
processed/color_segmentation/color_segmentation_features.csv
```

Notebook gồm các phần:

1. Load danh sách ảnh từ JSON.
2. Parse nhóm máu từ tên file.
3. Segment ảnh bằng HSV.
4. Visualize mask trên các ảnh mẫu.
5. Rút feature trong từng mask.
6. Lưu feature ra CSV.
7. Kiểm tra nhanh phân bố feature theo nhóm máu.

## 4. Các mask hiện tại

Prototype tạo các mask:

```text
anti_a_blue
anti_b_yellow
blood_red
foreground
reaction_candidate
```

Ý nghĩa:

- `anti_a_blue`: vùng xanh/cyan, ứng viên anti-A.
- `anti_b_yellow`: vùng vàng, ứng viên anti-B.
- `blood_red`: vùng đỏ/nâu của máu.
- `foreground`: vùng có màu hoặc không quá trắng, dùng để giảm background.
- `reaction_candidate`: vùng ứng viên phản ứng, kết hợp foreground và các mask màu.

## 5. Các phương pháp rút trích đặc trưng

Prototype hiện tại dùng nhiều nhóm đặc trưng để mô tả ảnh từ nhiều góc nhìn khác nhau. Việc kết hợp nhiều phương pháp là hợp lý vì phản ứng nhóm máu không chỉ thể hiện bằng màu, mà còn thể hiện bằng texture, cạnh, vùng kết tụ và phân bố pixel.

### 5.1. Color statistics

Color statistics đo giá trị màu trung bình và độ phân tán trong từng mask:

```text
r_mean, g_mean, b_mean
r_std, g_std, b_std
h_mean, s_mean, v_mean
h_std, s_std, v_std
gray_mean, gray_std
```

Ý nghĩa:

- `RGB` cho biết cường độ màu gốc.
- `HSV` giúp mô tả sắc màu, độ bão hòa và độ sáng.
- `gray_mean/std` cho biết vùng phản ứng sáng/tối và biến thiên mức xám.

Nhóm feature này có ích vì anti-A có xu hướng xanh, anti-B có xu hướng vàng, còn vùng máu có sắc đỏ/nâu.

### 5.2. HSV histogram

Histogram HSV mô tả phân bố màu trong mask thay vì chỉ dùng giá trị trung bình:

```text
hue_hist_0 ... hue_hist_11
sat_hist_0 ... sat_hist_7
val_hist_0 ... val_hist_7
```

Ý nghĩa:

- `Hue histogram`: phân bố sắc màu.
- `Saturation histogram`: mức độ đậm/nhạt màu.
- `Value histogram`: mức độ sáng/tối.

Lý do dùng histogram: hai vùng có cùng màu trung bình nhưng phân bố màu khác nhau vẫn có thể biểu hiện phản ứng khác nhau.

### 5.3. LBP texture

LBP (Local Binary Pattern) là đặc trưng texture cục bộ. Nó so sánh mỗi pixel với các pixel lân cận để tạo pattern nhị phân.

Trong prototype, LBP được lượng hóa thành histogram:

```text
lbp_hist_0 ... lbp_hist_15
lbp_entropy
```

Ý nghĩa:

- Vùng phản ứng mịn thường có texture đều hơn.
- Vùng có ngưng kết/lốm đốm thường tạo nhiều pattern cục bộ hơn.
- `lbp_entropy` cao hơn có thể biểu hiện texture phức tạp hơn.

LBP phù hợp với bài toán này vì phản ứng ngưng kết có thể nhìn như các hạt/cụm nhỏ trên nền máu.

### 5.4. GLCM texture

GLCM (Gray-Level Co-occurrence Matrix) mô tả quan hệ giữa mức xám của các pixel lân cận.

Prototype hiện tại rút:

```text
glcm_contrast
glcm_homogeneity
glcm_energy
glcm_entropy
```

Ý nghĩa:

- `contrast`: mức chênh lệch cường độ giữa pixel lân cận.
- `homogeneity`: độ đồng nhất của texture.
- `energy`: mức tập trung của ma trận đồng xuất hiện.
- `entropy`: độ phức tạp/ngẫu nhiên của texture.

GLCM hỗ trợ phân biệt vùng phản ứng đều và vùng phản ứng bị vón/kết cụm.

### 5.5. Edge and gradient features

Nhóm này đo cạnh và biến thiên cường độ:

```text
edge_density
gradient_mean
gradient_std
entropy
```

Ý nghĩa:

- `edge_density`: tỷ lệ pixel cạnh theo Canny.
- `gradient_mean/std`: mức thay đổi cường độ trong vùng.
- `entropy`: độ phức tạp của phân bố mức xám.

Nếu phản ứng có nhiều cụm/ngưng kết, vùng ảnh thường có nhiều biên nhỏ hơn vùng mịn.

### 5.6. Blob, component, and contour features

Nhóm này mô tả hình dạng vùng mask:

```text
area_ratio
component_count
largest_component_area
largest_component_ratio
contour_count
max_contour_area_ratio
max_contour_perimeter_ratio
max_contour_circularity
```

Ý nghĩa:

- `area_ratio`: mask chiếm bao nhiêu phần ảnh.
- `component_count`: số vùng rời rạc sau segmentation.
- `largest_component_ratio`: vùng lớn nhất chiếm bao nhiêu diện tích.
- `contour_circularity`: độ tròn của vùng lớn nhất.

Nhóm này giúp phát hiện mask bị vỡ thành nhiều vùng nhỏ hoặc vùng phản ứng có dạng cụm.

## 6. Feature được lưu trong CSV

```text
area_ratio
component_count
largest_component_area
largest_component_ratio
r_mean, g_mean, b_mean
h_mean, s_mean, v_mean
gray_mean, gray_std
edge_density
entropy
```

Ngoài các feature ban đầu trên, phiên bản mới còn có:

```text
RGB/HSV std
HSV histogram
LBP histogram and entropy
GLCM contrast/homogeneity/energy/entropy
Gradient features
Contour shape features
```

Các feature này phục vụ ba mục tiêu:

- Màu sắc: giúp phân biệt vùng xanh/vàng/đỏ.
- Texture: giúp mô tả phản ứng ngưng kết.
- Shape/blob: giúp mô tả vùng phản ứng và giảm ảnh hưởng background.

## 7. Cách giảm ảnh hưởng background

Prototype không trích feature trực tiếp từ toàn ảnh. Thay vào đó:

```text
Ảnh RGB
-> chuyển HSV
-> tạo mask màu/foreground
-> làm sạch mask bằng morphological open/close
-> bỏ component quá nhỏ
-> tính feature chỉ trên pixel thuộc mask
```

Cách này giúp giảm nhiễu từ nền trắng, giấy, bàn, ánh sáng xung quanh.

## 8. Hạn chế hiện tại

Đây mới là prototype, chưa phải pipeline cuối cùng.

Các hạn chế chính:

- Ngưỡng màu HSV có thể cần chỉnh theo ánh sáng thực tế.
- Anti-D không màu nên không thể nhận diện tốt chỉ bằng màu thuốc thử.
- Nếu ảnh có ánh sáng vàng/xanh từ môi trường, mask có thể bắt nhầm background.
- Nếu vùng phản ứng quá nhạt, mask có thể bỏ sót.
- Dataset hiện tại vẫn lệch lớp, đặc biệt O- quá ít.

## 9. Tiêu chí đánh giá prototype

Sau khi chạy notebook, cần xem phần visualization:

- Mask xanh có bám đúng vùng anti-A không?
- Mask vàng có bám đúng vùng anti-B không?
- Mask máu/foreground có bỏ được nền không?
- Background có bị bắt nhầm quá nhiều không?
- Các feature như `anti_a_blue_area_ratio`, `anti_b_yellow_area_ratio`, `blood_red_edge_density` có khác nhau giữa các nhóm máu không?

Nếu visualization ổn, ta có thể bước sang notebook classification.

Nếu visualization không ổn, cần:

- Chỉnh ngưỡng HSV.
- Thêm tiền xử lý cân bằng sáng.
- Dùng Lab color space.
- Hoặc annotate subset ảnh để có vùng anti-A/B/D đầy đủ.

## 10. Kết quả smoke test ban đầu

Đã chạy thử trên 8 ảnh đại diện, mỗi nhóm máu một ảnh nếu có trong dataset. Output kiểm tra nhanh:

```text
processed/color_segmentation/color_segmentation_features_smoke_test.csv
processed/color_segmentation/multi_feature_smoke_test.csv
processed/color_segmentation/segmentation_preview_samples.png
```

Nhận xét ban đầu:

- Mask xanh có bắt được vùng anti-A trong nhiều ảnh có thuốc thử xanh.
- Mask vàng có bắt được vùng anti-B trong nhiều ảnh có thuốc thử vàng.
- Mask đỏ/máu và foreground giúp tập trung vào giọt phản ứng thay vì toàn bộ nền.
- Một số ảnh vẫn bị bắt nhầm màu vàng hoặc vùng foreground rộng, nên không nên dùng rule cứng chỉ dựa vào mask area.
- `multi_feature_smoke_test.csv` có 382 cột feature cho 8 ảnh mẫu, cho thấy pipeline đã kết hợp nhiều nhóm đặc trưng.
- Anti-D vẫn là phần khó nhất vì không có màu thuốc thử rõ ràng; cần texture feature hoặc annotation bổ sung.

Kết luận sau smoke test: hướng color segmentation có tiềm năng để giảm ảnh hưởng background, nhưng cần tiếp tục kiểm tra bằng classifier baseline và chỉnh threshold.

## 11. Hướng tiếp theo

Hướng tiếp theo đề xuất:

```text
1. Chạy notebook segmentation.
2. Kiểm tra mask trực quan.
3. Chỉnh threshold nếu cần.
4. Tạo feature CSV.
5. Train baseline classifier truyền thống.
6. So sánh:
   - feature toàn ảnh
   - feature theo mask màu
   - nếu có thể, feature từ annotation subset thủ công
```

Classifier có thể thử:

```text
SVM
Random Forest
KNN
Logistic Regression
```

Với báo cáo, hướng này khách quan hơn việc dùng JSON bbox dương tính, vì model bắt đầu học từ đặc trưng ảnh thay vì học trực tiếp từ annotation.
