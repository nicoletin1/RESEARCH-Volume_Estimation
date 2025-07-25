# Project Plan: Volume Prediction Pipeline for Segmented Image Objects

## Objective
Build a Python-based pipeline to predict the volume of segmented objects in synthetic SEM images using paired 2D RGB images and 3D ground truth data. The system will:
- Segment multiple objects per image using Segment Anything.
- Predict the volume of each segmented object using deep learning (CNN/3D reconstruction).
- Output results in CSV format.
- Support extensibility for new shapes and data generation.

## Data Overview
- **Input:**
  - 2D RGB PNG images (multiple objects per image)
  - JSON files containing 3D coordinates and particle sizes (one JSON per image)
- **Output:**
  - CSV file with predicted volumes for each segmented object

## Pipeline Steps
1. **Data Preprocessing**
   - Parse JSON files to extract 3D ground truth volumes for each object.
   - Pair each PNG image with its corresponding JSON file.
   - Prepare training/validation splits.

2. **Segmentation**
   - Integrate Segment Anything to generate masks for each object in the image.
   - Store masks for downstream processing.

3. **Volume Prediction Model**
   - Design and implement a deep learning model (CNN or 3D reconstruction) to predict object volumes from segmented image regions.
   - Train the model using paired image/mask/ground truth volume data.
   - Validate and test model performance.

4. **Inference Pipeline**
   - For new images, segment objects and predict their volumes.
   - Save results in CSV format: `image_id, object_id, predicted_volume`

5. **Extensibility for New Shapes**
   - Develop scripts to generate new json/png pairs for additional shapes.
   - Update training pipeline to include new data.

## Implementation Details
- **Language:** Python
- **Frameworks:** PyTorch or TensorFlow (user preference)
- **Segmentation:** Segment Anything (SAM)
- **Output:** CSV (preferred), JSON (optional)
- **Structure:** Script/module-based (not notebook-based)

## Directory Structure
- `src/volume_prediction/` – Core pipeline modules
- `data/` – Raw and processed data
- `scripts/` – Utility scripts for data generation and processing
- `models/` – Model definitions and checkpoints
- `outputs/` – Predicted volumes (CSV)

## Milestones
1. Data parsing and preprocessing scripts
2. Segment Anything integration
3. Model architecture and training
4. Inference and CSV output
5. Extensibility for new shapes

## Next Steps
- Confirm deep learning framework (PyTorch/TensorFlow)
- Begin with data parsing and mask generation
- Prototype model and training loop
- Validate pipeline and iterate

---
For questions or changes, update this plan and commit to the project root.
