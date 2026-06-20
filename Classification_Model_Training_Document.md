# Classification Model Training Document

## 1. Muc tieu cua buoc classification

Sau buoc color segmentation va feature extraction, do an da co file feature:

```text
processed/color_segmentation/color_segmentation_features.csv
```

File nay la input chinh cho buoc classification. Moi dong tuong ung mot anh, moi cot feature mo ta thong tin mau sac, texture, canh, hinh dang va su xuat hien cua cac mask phan ung.

Muc tieu cua buoc nay:

- Doc bang feature da trich xuat tu notebook 02.
- Chon cac cot numeric lam input cho model.
- Train nhieu mo hinh machine learning truyen thong.
- So sanh cac mo hinh bang metric phu hop voi du lieu lech lop.
- Chon mo hinh tot nhat.
- Luu mo hinh da hoc vao folder `models/` de dung lai cho inference hoac app demo.

Notebook duoc tao cho buoc nay:

```text
03_classification_model_training.ipynb
```

Notebook nay chua duoc run san. Nguoi dung se tu chay sau khi da co feature CSV moi nhat.

## 2. Input cua notebook classification

Input chinh:

```text
processed/color_segmentation/color_segmentation_features.csv
```

Feature CSV hien tai co cac nhom cot:

- Metadata:
  - `split`
  - `file_name`
  - `blood_group`
  - `abo`
  - `rh`
- Kich thuoc anh sau resize:
  - `resized_width`
  - `resized_height`
- Feature theo tung mask:
  - `anti_a_blue_*`
  - `anti_b_yellow_*`
  - `blood_red_*`
  - `foreground_*`
  - `reaction_candidate_*`

Moi mask co cac nhom feature:

- Mask presence:
  - `*_mask_present`
- Dien tich va connected component:
  - `*_area_ratio`
  - `*_component_count`
  - `*_largest_component_area`
  - `*_largest_component_ratio`
- Contour shape:
  - `*_contour_count`
  - `*_max_contour_area_ratio`
  - `*_max_contour_perimeter_ratio`
  - `*_max_contour_circularity`
- Mau RGB/HSV:
  - `*_r_mean`, `*_g_mean`, `*_b_mean`
  - `*_r_std`, `*_g_std`, `*_b_std`
  - `*_h_mean`, `*_s_mean`, `*_v_mean`
  - `*_h_std`, `*_s_std`, `*_v_std`
- Gray, edge, gradient, entropy:
  - `*_gray_mean`
  - `*_gray_std`
  - `*_edge_density`
  - `*_gradient_mean`
  - `*_gradient_std`
  - `*_entropy`
- Histogram:
  - `*_hue_hist_*`
  - `*_sat_hist_*`
  - `*_val_hist_*`
- Texture:
  - `*_lbp_hist_*`
  - `*_lbp_entropy`
  - `*_glcm_contrast`
  - `*_glcm_homogeneity`
  - `*_glcm_energy`
  - `*_glcm_entropy`

## 3. Xu ly NaN va y nghia cua mask rong

Trong buoc feature extraction ban dau, mot so feature co the bi `NaN` khi mask rong. Vi du, neu anh nhom mau `B` hoac `O` khong co phan ung voi anti-A, mau xanh anti-A co the bi mau lan ra lam mat dau hieu xanh ro rang. Khi do mask `anti_a_blue` rong.

Ve mat mien bai toan, mask rong khong phai loi. No la tin hieu co y nghia:

```text
anti_a_blue rong -> khong phat hien vung xanh anti-A ro rang
```

De training on dinh hon, notebook 02 da duoc dieu chinh theo huong:

- Them cot `*_mask_present`.
- Neu mask rong, cac feature cua mask do duoc gan `0`.
- Feature CSV moi khong con `NaN`.

Notebook 03 van giu `SimpleImputer(strategy="constant", fill_value=0.0)` trong pipeline nhu mot lop bao ve. Neu sau nay CSV co NaN phat sinh, model van co the train duoc.

## 4. Target classification

Notebook 03 dat target mac dinh:

```python
TARGET = "blood_group"
```

Target nay tuong ung bai toan phan loai 8 nhom mau:

```text
A+, A-, B+, B-, AB+, AB-, O+, O-
```

Ngoai ra notebook co the doi target de train cac bai toan nho hon:

```python
TARGET = "abo"
```

hoac:

```python
TARGET = "rh"
```

Y nghia:

- `blood_group`: phan loai day du 8 lop.
- `abo`: phan loai he ABO gom `A`, `B`, `AB`, `O`.
- `rh`: phan loai Rh `+` hoac `-`.

Voi bao cao, co the train ca 3 kieu target de so sanh do kho cua tung bai toan.

## 5. Chon feature dau vao

Notebook chi dung cac cot numeric lam input:

```python
numeric_cols = df.select_dtypes(include="number").columns.tolist()
```

Sau do loai cac cot metadata/label:

```python
metadata_cols = ["split", "file_name", "blood_group", "abo", "rh"]
feature_cols = [c for c in numeric_cols if c not in metadata_cols]
```

Ly do:

- `file_name` khong nen dua vao model vi co the gay leak label.
- `blood_group`, `abo`, `rh` la label, khong phai feature.
- `split` chi la thong tin chia tap, khong phai dac trung anh.

Input model:

```python
X = df[feature_cols]
y = df[TARGET].astype(str)
```

## 6. Chia train/test

Dataset goc bi lech split manh:

- `valid` khong co du 8 nhom mau.
- `test` hau nhu chi co `A-`, `AB+` va mot it `A+`.
- `O-` chi co rat it anh.

Neu dung split goc de danh gia bai toan 8 lop, ket qua co the khong dai dien dung nang luc model.

Vi vay notebook 03 dat mac dinh:

```python
USE_ORIGINAL_SPLIT = False
```

Khi do notebook dung stratified split tren toan bo feature CSV:

```python
train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)
```

Loi ich:

- Train/test deu co phan bo lop gan giong nhau.
- Phu hop hon de so sanh baseline model.
- Giam rui ro test set thieu lop.

Neu muon so sanh voi split goc, co the doi:

```python
USE_ORIGINAL_SPLIT = True
```

Khi do:

- `train` duoc dung de train.
- `valid` va `test` duoc gop lam test.

Tuy nhien ket qua can duoc ghi chu ro trong bao cao vi split goc khong can bang.

## 7. Cac mo hinh duoc train

Notebook 03 tao nhieu candidate models de so sanh.

### 7.1. Logistic Regression

```python
LogisticRegression(max_iter=3000, class_weight="balanced")
```

Day la baseline tuyen tinh. Uu diem:

- De giai thich.
- Chay nhanh.
- Phu hop lam moc so sanh ban dau.

Dung `class_weight="balanced"` de giam anh huong lech lop.

### 7.2. Linear SVM

```python
LinearSVC(class_weight="balanced", max_iter=10000)
```

Linear SVM phu hop khi so feature nhieu va can boundary tuyen tinh manh. Uu diem:

- Thuong tot voi feature vector co chieu cao.
- Chay nhanh hon SVM kernel tren tap lon.

### 7.3. RBF SVM

```python
SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")
```

RBF SVM la model phi tuyen. Uu diem:

- Co the hoc quan he phuc tap giua feature mau, texture va label.
- Phu hop khi boundary giua cac nhom mau khong tuyen tinh.

Han che:

- Co the cham hon Linear SVM.
- Can tuning `C`, `gamma` neu muon toi uu.

### 7.4. KNN

```python
KNeighborsClassifier(n_neighbors=5)
```

KNN dung khoang cach giua feature vectors. Uu diem:

- Don gian.
- Cho biet feature space co tach lop tu nhien hay khong.

Han che:

- Nhay voi scale cua feature.
- De bi anh huong boi lop lech va nhieu feature nhiem.

### 7.5. Random Forest

```python
RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    n_jobs=-1
)
```

Random Forest la ensemble cua decision trees. Uu diem:

- Xu ly feature phi tuyen tot.
- It can scale feature hon SVM/KNN.
- Co the hoc tu feature area, histogram, texture va contour.

### 7.6. Extra Trees

```python
ExtraTreesClassifier(
    n_estimators=300,
    class_weight="balanced",
    n_jobs=-1
)
```

Extra Trees gan giong Random Forest nhung them tinh ngau nhien khi split. Uu diem:

- Chay nhanh.
- Thuong la baseline manh cho tabular features.
- Co the giam overfitting trong mot so truong hop.

## 8. Pipeline tien xu ly

Notebook dung hai kieu pipeline.

Voi model can scale feature:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
    ("scaler", StandardScaler()),
    ("model", model),
])
```

Ap dung cho:

- Logistic Regression
- Linear SVM
- RBF SVM
- KNN

Ly do can scaler:

- Cac feature co thang do khac nhau, vi du `area_ratio` tu 0 den 1, nhung `largest_component_area` co the rat lon.
- SVM, Logistic Regression va KNN nhay voi scale.

Voi tree-based model:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
    ("model", model),
])
```

Ap dung cho:

- Random Forest
- Extra Trees

Tree model khong bat buoc scale feature, nen khong dung `StandardScaler`.

## 9. Cross-validation va metric danh gia

Notebook dung Stratified K-Fold:

```python
StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
```

Metric chinh de so sanh model:

```text
f1_macro
```

Ly do dung `f1_macro`:

- Dataset lech lop.
- Accuracy co the cao neu model doan tot lop nhieu mau nhung bo qua lop it mau.
- `f1_macro` tinh trung binh F1 tren tung lop, moi lop co trong so ngang nhau.

Notebook cung tinh:

- `test_accuracy`
- `test_f1_macro`
- `test_f1_weighted`

Y nghia:

- `accuracy`: ty le doan dung tong quat.
- `f1_macro`: danh gia cong bang hon giua cac lop.
- `f1_weighted`: F1 co trong so theo so mau moi lop.

## 10. Chon model tot nhat

Sau khi train tat ca candidate models, notebook tao bang:

```python
results_df
```

Model tot nhat duoc chon theo:

```python
best_name = results_df.loc[0, "model"]
```

Vi `results_df` da duoc sort theo:

```python
test_f1_macro
```

Model tot nhat sau do duoc dung de:

- In `classification_report`.
- Ve confusion matrix.
- Luu vao folder `models/`.

## 11. Output sau khi train

Notebook se tao folder:

```text
models/
```

Va luu cac file:

```text
models/color_segmentation_blood_group_<model_name>_<timestamp>.joblib
models/color_segmentation_blood_group_metrics_<timestamp>.csv
models/color_segmentation_blood_group_classification_report_<timestamp>.txt
```

Neu doi target thanh `abo` hoac `rh`, ten file se tuong ung:

```text
models/color_segmentation_abo_<model_name>_<timestamp>.joblib
models/color_segmentation_rh_<model_name>_<timestamp>.joblib
```

## 12. Noi dung cua file model joblib

File `.joblib` khong chi luu model. No luu mot package gom:

```python
model_package = {
    "model": best_model,
    "model_name": best_name,
    "target": TARGET,
    "feature_columns": feature_cols,
    "classes": labels,
    "feature_path": str(FEATURE_PATH),
    "use_original_split": USE_ORIGINAL_SPLIT,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "saved_at": timestamp,
    "metrics": results_df.to_dict(orient="records"),
}
```

Thanh phan quan trong nhat:

- `model`: pipeline da fit, gom imputer/scaler/model.
- `feature_columns`: danh sach cot feature dung khi train.
- `classes`: danh sach lop.
- `target`: label ma model duoc train de du doan.

Luu `feature_columns` la rat quan trong. Khi predict anh moi, app phai tao DataFrame co dung cac cot nay, dung thu tu nay.

## 13. Cach load model da luu

Vi du load model:

```python
import joblib
import pandas as pd

package = joblib.load("models/color_segmentation_blood_group_extra_trees_20260604_153000.joblib")
model = package["model"]
feature_columns = package["feature_columns"]

X_new = pd.DataFrame([feature_dict])
X_new = X_new.reindex(columns=feature_columns, fill_value=0.0)

pred = model.predict(X_new)[0]
```

Neu model co `predict_proba`, co the lay xac suat:

```python
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X_new)[0]
```

Luu y: `LinearSVC` khong co `predict_proba` mac dinh.

## 14. Co dung file models de test app Streamlit duoc khong?

Co. File `.joblib` trong folder `models/` co the dung cho Streamlit app.

Tuy nhien app Streamlit khong chi load model la du. App phai lam du cac buoc inference:

```text
Anh upload tu nguoi dung
-> doc anh RGB
-> segment bang dung ham trong notebook 02
-> extract feature bang dung logic trong notebook 02
-> tao DataFrame 1 dong
-> reindex dung feature_columns trong model package
-> model.predict()
-> hien thi ket qua
```

Noi cach khac:

```text
models/*.joblib dung duoc cho app.py
nhung app.py phai co cung pipeline feature extraction voi notebook 02
```

Neu app chi upload anh roi dua truc tiep anh vao `.joblib`, se khong chay dung, vi model da train tren tabular features, khong train truc tiep tren pixel anh.

## 15. Streamlit app can co nhung thanh phan gi?

Mot `app.py` co the gom:

1. Load model package:

```python
package = joblib.load(model_path)
model = package["model"]
feature_columns = package["feature_columns"]
```

2. Upload image:

```python
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
```

3. Chuyen uploaded image thanh RGB array.

4. Goi lai cac ham:

```python
segment_image(rgb)
features_for_mask(rgb, mask, prefix)
extract_image_features_from_rgb(rgb)
```

5. Tao DataFrame:

```python
X_new = pd.DataFrame([features])
X_new = X_new.reindex(columns=feature_columns, fill_value=0.0)
```

6. Predict:

```python
pred = model.predict(X_new)[0]
```

7. Hien thi ket qua:

```python
st.metric("Predicted blood group", pred)
```

Co the hien thi them:

- Anh goc.
- Mask anti-A blue.
- Mask anti-B yellow.
- Mask blood red.
- Confidence neu model co `predict_proba`.

## 16. Khuyen nghi khi lam Streamlit

Nen tach code feature extraction tu notebook 02 thanh file Python rieng, vi du:

```text
src/feature_extraction.py
```

File nay chua:

- `read_rgb`
- `clean_mask`
- `segment_image`
- `features_for_mask`
- `extract_image_features_from_rgb`

Sau do:

- Notebook 02 import lai file nay de tao CSV.
- `app.py` import lai file nay de predict anh moi.

Cach nay giup dam bao:

```text
feature extraction luc train == feature extraction luc app inference
```

Neu copy code thu cong giua notebook va app, rat de lech threshold hoac thieu cot feature.

## 17. Han che can ghi trong bao cao

Mot so diem can neu ro:

- Dataset bi lech lop, dac biet `O-` rat it mau.
- Color segmentation phu thuoc nguong HSV, co the bi anh huong boi anh sang.
- Anti-D kho hon anti-A/anti-B vi khong co mau thuoc thu ro rang.
- Model hien tai la classifier tren handcrafted features, khong phai deep learning end-to-end.
- Ket qua baseline nen duoc danh gia bang macro F1 va confusion matrix, khong chi accuracy.

## 18. Huong phat trien tiep theo

Sau baseline classification, co the cai tien theo cac huong:

- Tune hyperparameter bang GridSearchCV hoac RandomizedSearchCV.
- Train rieng model `abo` va model `rh`, sau do ghep thanh nhom mau day du.
- Can bang lai dataset hoac tao stratified split tot hon.
- Tach code feature extraction thanh module Python de dung chung cho notebook va Streamlit.
- Them Lab color space hoac color constancy de giam anh huong anh sang.
- Thu CNN/transfer learning neu co them du lieu va tai nguyen tinh toan.

