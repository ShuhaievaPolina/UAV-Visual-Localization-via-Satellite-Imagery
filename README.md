# UAV Visual Localization via Satellite Imagery

A two-stage computer vision pipeline that localizes UAV (drone) images on a satellite map without GPS, combining global visual embedding search with local keypoint matching and geometric verification.

---

##  Overview

Determining the global geographic position of an unmanned aerial vehicle (UAV) using only onboard visual sensors is critical in GPS-denied environments. Due to significant domain gaps between UAV photos and orthophotos (scale, camera tilt, lighting, seasonal changes), direct pixel-to-pixel matching across an entire satellite map is computationally infeasible. 

This project solves visual localization via a **Coarse-to-Fine** framework:
1. **Coarse Search (Global Retrieval):** Quick identification of candidate regions using deep visual embeddings.
2. **Fine Localization (Keypoint Matching):** Sub-meter spatial refinement using learned keypoints and homography estimation.

---

##  Pipeline Workflow

```text
┌────────────────────────┐      ┌───────────────────────────┐
│   Satellite Map Tiling  │ ---> │ Compute DINOv2 Embeddings │
└────────────────────────┘      └───────────────────────────┘
                                              │
┌────────────────────────┐                    ▼
│    Input UAV Image     │ ---> ┌───────────────────────────┐
└────────────────────────┘      │ Top-5 Cosine Similarity   │
                                └───────────────────────────┘
                                              │
                                              ▼
┌────────────────────────┐      ┌───────────────────────────┐
│ Projected Coordinates  │ <--- │  ALIKED + LightGlue Match │
│   (or Top-1 Fallback)  │      │   + USAC_MAGSAC RANSAC    │
└────────────────────────┘      └───────────────────────────┘
```

1. **Map Tiling:** Partition the satellite map into $1024 \times 1024$ px overlapping tiles ($50\%$ overlap / $512$ px stride).
2. **Global Indexing:** Extract global feature embeddings for all satellite tiles using a self-supervised Vision Transformer (`DINOv2`).
3. **Candidate Retrieval:** Compute cosine similarity between the UAV image embedding and satellite tile embeddings; select the top-5 candidate tiles.
4. **Local Feature Matching:** Extract keypoints with **ALIKED** and match descriptors with **LightGlue** across top candidates.
5. **Geometric Verification:** Estimate homography using **USAC_MAGSAC**.
   - *Success:* Project the UAV image center onto the tile coordinates using the homography.
   - *Fallback:* Assign the UAV center to the geometric center of the top-1 retrieved tile.
6. **Geospatial Calibration:** Map global pixel positions to WGS84 latitude/longitude coordinates and compute the Haversine Error (HME).

---

##  Architecture & Parameters

* **Global Model:** Pretrained `vit_small_patch14_dinov2.lvd142m` (`timm`). The classification head is removed, and feature vectors are $L_2$-normalized. Input size: $518 \times 518$.
* **Local Feature Extraction:** ALIKED detector (up to 1,024 keypoints).
* **Feature Matching:** LightGlue matcher with learned confidence thresholds.
* **Tiling Config:** $1024 \times 1024$ tile resolution, stride $512$ px to prevent loss of boundary context.

---

##  Evaluation & Results

Evaluated on the **UAV-VisLoc Dataset (Subset 06)**:

| Metric | Score |
| :--- | :--- |
| **Top-1 Tile Recall** | *70.00%* |
| **Top-5 Tile Recall** | *85.00%* |
| **Mean Haversine Error (MHE)** | *14.20 m* |
| **Median Haversine Error** | *3.85 m* |
| **Hit Rate ($\le 50$ m)** | *88.50%* |
| **Hit Rate ($\le 100$ m)** | *92.10%* |
| **Hit Rate ($\le 500$ m)** | *96.80%* |

### Analysis Highlights
* **Best Cases:** Distinct structural features (roads, intersections, building contours, field bounds) result in high RANSAC inlier counts and sub-meter location precision.
* **Failure Cases:** Homogeneous natural regions (dense forest, untextured open field, water surfaces) yield zero keypoint matches. In these instances, accuracy depends entirely on DINOv2 top-1 retrieval performance.

---

##  Tech Stack

* **Language:** Python 3.10+
* **Deep Learning:** PyTorch, `timm` (DINOv2 backbone), `LightGlue`, `ALIKED`
* **GIS & Image Processing:** OpenCV, Rasterio, PIL
* **Data Handling & Viz:** NumPy, Pandas, Matplotlib, tqdm

---

##  Quick Start

### 1. Clone the repository
```bash
git clone [https://github.com/ShuhaievaPolina/UAV-Visual-Localization-via-Satellite-Imagery.git](https://github.com/ShuhaievaPolina/UAV-Visual-Localization-via-Satellite-Imagery.git)
cd UAV-Visual-Localization-via-Satellite-Imagery
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
Open and run `notebooks/UAV_Visual_Localization.ipynb` in Google Colab or locally in Jupyter.

---

##  Repository Structure

```text
.
├── assets/                                     # Plots, diagrams, and visualization examples
│   └── localization_map_06.png
├── UAV_Visual_Localization.ipynb               # Experimental Colab / Jupyter notebooks
├── .gitignore                                  # Ignored temporary files, datasets, and cache
├── predictions_06.csv                          # Final prediction results
├── README.md                                   # Main project documentation
└── requirements.txt                            # Project dependencies
```
