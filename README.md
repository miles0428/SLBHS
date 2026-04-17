# TWSLT — Taiwanese Sign Language Translation

## 目標
對台灣手語影片的 MediaPipe hand landmark 資料進行分析、對齊、與動作聚類。

## 資料格式
每支影片輸出為 `.h5` 檔，包含：
- `aligned_63d` — (N, 63) 對齊後 landmark，21 joints × 3 components
- `x_vec`, `y_vec`, `z_vec` — (N, 3) 手部座標軸
- `wrist_px` — (N, 3) 腕關節像素座標
- `is_mirror` — (N,) 是否為鏡像（數學推導）資料

## Pipeline
1. **Preprocessing** — 從 MP4 提取 landmark + 旋轉對齊
2. **K-Means** — 對 `aligned_63d` 做聚類（第一層）
3. **Temporal Segmentation** — 時間序列切分
4. **Model** — 手語動作辨識

## 目前進度
- [x] 對齊驗證：掌心平面已固定到 z=0 平面，掌心法向量已統一朝向 +z
- [ ] K-Means 聚類
- [ ] Temporal segmentation
- [ ] 動作模型
