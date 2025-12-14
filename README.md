# Benchmarking Computer Vision Models for Non-Invasive Screening of ALS and Stroke

Neurological disorders, such as Amyotrophic Lateral Sclerosis (ALS) and stroke, affect millions globally and are characterized by subtle symptoms that inhibit early detection. Early and accurate diagnosis is critical for improving patient outcomes, enabling timely therapeutic intervention, and facilitating long-term care planning. Given the limitations of current subjective clinical assessments, there is an urgent need for accessible, objective, and automated screening tools. As AI Engineers, it is our moral duty to help with this dire situation.

This project aims to advance the field of Neurology by establishing an algorithmic and explainable method for the prediction and classification of neurological conditions using facial video data. The Toronto NeuroFace Dataset was created by the University of Toronto Speech Language Pathology Lab for exactly this purpose. We benchmark the performance of both traditional Convolutional Neural Networks (CNNs) and advanced Vision Transformers (ViTs) on annotated still frames, aiming to assess their ability to capture facial-motor abnormalities. Furthermore, we use GRAD-CAM to visualize the reasoning of the model’s decisions. This ensures the model’s predictions are explainable and trustworthy for clinicians and patients.

---

## Project Structure

This repository contains a modular, task-aware deep learning pipeline for classifying neurological conditions using facial motion analysis. 

`neuroface`:
- `config.py`: Global configuration
- `metadata.py`: Frame => subject => task indexing
- `splits.py`: Subject-wise train/val/test split
- `io_bbox_landmarks.py`: Landmark & bbox I/O
- `image_preproc.py`: Face cropping and resizing
- `landmark_features.py`: Landmark normalization & feature extraction
- `aggregation.py`: Subject-level prediction aggregation
- `classical_models.py`: Task aware classical ML pipeline
- `transforms.py`: Safe Image Augmentations

`scripts`:
- `build_metadata.py`: Builds the metadata file for preprocessing
- `build_landmark_features.py`: Builds the landmark features
- `create_splits.py`: Creates the train/val/test split
- `preprocess_frames.py`: Preprocesses the frames
- `train_classical.py`: Trains LR, SVM and RF models
- `dataset_utils.py`: Utility files for the dataset
- `eval_utils.py`: Utility files for evaluating the data
- `RandomForest_model.py`: Trains the RnadomForest model
- `MLP_models.py`: Architecture for MLP model
- `MLP_train.py`: Trains the MLP models
- `grad_cam.py`: Explainability using GradCAM
- `grad_cam_als.py`: GradCAM on ALS patients

`metadata`:
- `metdata_frames.csv`: Consists of metadata for all subjects
- `metadata_frames_with_splits.csv`: Consists of metadata split into train/val/test
- `landmark_frame_features.csv`: Normalized landmark features for all the frames
- `landmark_subject_task_features.csv`: Normalized landmark features aggregated according to task

The `MainProject_workflow.ipynb` file consists of the workflow for training all the Computer Vision models.

## Installation

Upload the NeuroFace data in the root directory structure of the project and then run:

```
pip install -r requirements.txt
```