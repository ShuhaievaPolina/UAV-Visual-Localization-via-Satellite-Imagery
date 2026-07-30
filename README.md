# UAV-Visual-Localization-via-Satellite-Imagery
A two-stage computer vision pipeline that localizes UAV (drone) images on a satellite map without GPS, by combining global visual embedding search with local keypoint matching.

Overview

Given a drone image and a large satellite map, the goal is to determine where on the map the drone image was taken — purely from visual content. The approach first narrows down the search to the most visually similar region of the map using global image embeddings, then refines the exact position using local keypoint matching and geometric verification. This two-stage design was chosen because of the significant visual gap between UAV photos and satellite imagery (different viewpoint, scale, and appearance), which makes direct pixel-level matching across the entire map impractical.

Pipeline<br>
Tile the satellite map into overlapping patches<br>
Compute a global embedding for every tile<br>
Build a searchable database of satellite tile candidates<br>
Compute a global embedding for the UAV image<br>
Rank tiles by cosine similarity to the UAV embedding<br>
Select the top-5 candidate tiles<br>
Run ALIKED keypoint detection on the UAV image and candidate tiles<br>
Match keypoints between the UAV image and each candidate using LightGlue<br>
Estimate a homography with USAC_MAGSAC (RANSAC-based)<br>
If a homography with enough inliers is found → project the UAV image center through it; otherwise → fall back to the center of the top-1 tile<br>
Convert the projected point to global pixel coordinates on the satellite map<br>
Convert pixel coordinates to real-world geographic coordinates (lat/lon)<br>
Compute the Haversine error between predicted and ground-truth location<br>
Visualize results and report metrics<br>

Architecture

Global retrieval: a pretrained vit_small_patch14_dinov2.lvd142m (DINOv2 Vision Transformer, self-supervised) is used purely as a feature extractor — its classification head is discarded, and each image is reduced to a global embedding describing its overall visual content. Images are resized to 518×518, normalized with ImageNet statistics, and embeddings are L2-normalized before comparison via cosine similarity.
Local matching: ALIKED for keypoint detection and LightGlue for learned keypoint matching between the UAV image and the selected satellite tile, followed by RANSAC-based homography estimation (USAC_MAGSAC) for geometric verification.

Tiling Parameters

The satellite map is split into 1024×1024 pixel tiles with 50% overlap (stride of 512 pixels), which reduces the risk of the target region falling on a tile boundary without enough spatial context. The 1024px tile size balances context for global retrieval against localization precision within a tile.

Tech Stack

Language: Python (Google Colab)<br>
Deep learning: PyTorch, timm (DINOv2 backbone), LightGlue, ALIKED<br>
Geospatial / imaging: rasterio, OpenCV<br>
Data & visualization: NumPy, Pandas, Matplotlib<br>

Evaluation

The pipeline is evaluated on a subset of the UAV-VisLoc dataset using:
Mean / median Haversine Error (HME) and 90th percentile error<br>
Hit rate at 50 m / 100 m / 500 m thresholds<br>
Top-1 and Top-5 retrieval recall for the global search stage<br>

Results & Analysis

The best localization results occur on scenes with distinctive landmarks — roads, buildings, and clear field or forest boundaries — where a high number of RANSAC inliers corresponds to a stable homography and low error. Cosine similarity from the global search doesn't directly correlate with final accuracy; what matters is that the correct tile appears in the top-k candidates, after which local keypoint matching drives the precision down to a few meters. The worst results occurred on homogeneous natural terrain, where local matching found zero reliable inliers, forcing the pipeline to fall back to the center of an incorrectly selected tile.

Conclusions & Future Work

The combined global-search + local-matching approach performs well on scenes with clear structural features, but struggles on textureless natural terrain where local matching fails outright. Potential improvements include multi-scale tiling and evaluating a larger pool of keypoint candidates, as well as fine-tuning the embedding model on data more representative of this specific task.
