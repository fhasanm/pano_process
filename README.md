# Panoramic Process Pipeline

This project provides a pipeline for processing 360° panoramic video footage. It includes stages for sampling frames from videos, anonymizing images by blurring sensitive information (faces, license plates), identifying building façades near panorama locations, rotating panoramas to face these façades, extracting 2D cube faces from the 360° images, and sorting the outputs into per-building folders with associated metadata.

---

## 🌟 Features

* **Video Sampling:** Extracts frames from MP4 videos based on distance traveled.
* **Image Anonymization:** Detects and blurs faces and license plates using YOLOv8 and SAM.
* **Interactive Camera Offset Calibration:** Helps determine the yaw offset of the camera relative to the vehicle.
* **Façade Matching:** Identifies relevant building façades from a GeoJSON dataset based on panorama locations.
* **Panorama Rotation:** Orients panoramas to directly face identified building façades.
* **Cube Face Extraction:** Converts rotated 360° panoramas into selected 2D cubemap faces (e.g., front, left, right).
* **Sorted Output Generation:** Organizes processed cube faces and metadata into per-building directories.
* **Standalone Deployment:** Option to process data for a single, specific building.

---

## 🛠️ Prerequisites

* Python 3.8+
* Conda (for environment management)
* Git (for cloning the repository)
* `mapillary_tools` command-line utility.
* Access to a CUDA-enabled GPU is recommended for faster model inference, but CPU is also supported.

---

## ⚙️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/fhasanm/pano_process.git
    cd pano_process
    ```

2.  **Create and Activate Conda Environment:**
    The `environment.yml` file lists all necessary dependencies.
    ```bash
    conda env create -f environment.yml
    conda activate lafire_env 
    ```
    *(The environment name in `environment.yml` is `lafire_env`. If you change it there, activate the new name.)*

3.  **Install `mapillary_tools`:**
    If not already included or fully set up by the Conda environment, ensure `mapillary_tools` is installed and accessible from your command line within the activated `lafire_env`.
    ```bash
    pip install mapillary-tools
    ```
    Verify the installation:
    ```bash
    mapillary_tools --version
    ```

4.  **Download Models:**
    The pipeline uses YOLOv8 and SAM models. Download them from the provided links and place them into the `models/` directory within the project. Ensure the filenames in `models/` match those specified in `config.py`.

    * **Model Files (YOLOv8, SAM):**
        * Download link: [Model Files (YOLO & SAM)](https://drive.google.com/drive/folders/1xFkKj4-fnDxxQuLy1cQjXGFRJBunOZUz?usp=sharing)
        * **Important:** After downloading, ensure the following files (or similar, matching `config.py`) are present in your local `models/` directory:
            * `yolov8n-face.pt`
            * `license_plate_detector.pt`
            * `sam2.1_hiera_tiny.pt`
            * `sam2.1_hiera_t.yaml` 
    

5.  **Download Input Data (GeoJSON, CSV):**
    The necessary building footprint and damage assessment data can be downloaded from the link below. Place the contents into the `data/` directory.
    * **Data Files:**
        * Download link: `[Data Files](https://drive.google.com/drive/folders/1soyVTu883yjU1dJxuahBrSK3lNtnUnAn?usp=sharing)
        * **Important:** After downloading, ensure the following files are present in your local `data/` directory:
            * `2023_Buildings_with_DINS_data.geojson`
            * `2023_Buildings_with_DINS_data.csv`
        * *Instructions for Google Drive link:* Create a shareable link for your Google Drive folder containing these data files. Set permissions to "Anyone with the link can view."
    * **Input Video:** You will need to provide your own input video(s) (e.g., `.mp4`). The path to the video is specified in the configuration of `main.py` or `deploy.py`.

6.  **Review and Update Configuration (`config.py`):**
    Open `config.py` in the root of the project and verify/update the following:
    * `MODELS_DIR`: Should point to your `models/` directory (default is `./models`).
    * `FACE_WEIGHTS_PATH`, `PLATE_WEIGHTS_PATH`, `SAM_CHECKPOINT_PATH`: Ensure these filenames match the model files you downloaded into `MODELS_DIR`.
    * `MAPILLARY_TOOLS_PATH`: If `mapillary_tools` is not in your system PATH after installation, provide the full executable path here (e.g., `/path/to/your/conda/envs/lafire_env/bin/mapillary_tools`).
    * `DEVICE`: Set to `"cuda"` to use GPU (recommended) or `"cpu"`. The script attempts to use CUDA if available.
    * Review other default parameters as needed.

---

## 🚀 Usage

There are two main ways to run the pipeline:

### 1. Full Pipeline Processing (`main.py`)

This script processes an entire video through all selected stages.

* **Configure `main.py`:**
    Open `main.py` and adjust the settings in the "USER CONFIGURATION" section:
    * `VIDEO_FILE_PATH`: Path to your input MP4 video.
    * `BUILDING_FOOTPRINTS_GEOJSON_PATH`: Path to your building footprints (e.g., `./data/2023_Buildings_with_DINS_data.geojson`).
    * `BUILDING_DAMAGE_CSV_PATH`: Path to the CSV containing damage information (e.g., `./data/2023_Buildings_with_DINS_data.csv`).
    * `BASE_OUTPUT_DIR`: Directory where all processed subfolders and files will be saved (e.g., `./output`).
    * **Stage Execution Flags (`RUN_STAGE_X`):** Set these to `True` or `False` to control which parts of the pipeline run.
    * **Stage-Specific Parameters:** Review and adjust parameters for sampling, blurring, façade processing, etc. For `PANO_ZERO_OFFSET_FOR_ROTATION`, if `RUN_STAGE_3_OFFSET_MEASUREMENT` is `True`, an interactive step will allow you to measure this; otherwise, the configured value is used.

* **Run `main.py`:**
    Ensure your `lafire_env` Conda environment is activated. From the project root directory (`pano_process`):
    ```bash
    python main.py
    ```
    *(If you encounter import errors related to `config` or `utils` not being found, ensure they are in the same directory as `main.py`, or run as a module from the parent directory: `cd ..` then `python -m pano_process.main` after ensuring `pano_process/__init__.py` exists and imports in `main.py` are `from . import config`, etc.)*

### 2. Standalone Deployment for a Single Building (`deploy.py`)

This script processes a video specifically to generate outputs for a single target building. It performs its own sampling and processing workflow internally.

* **Configure `deploy.py`:**
    Open `deploy.py` and adjust settings in its "USER CONFIGURATION" section:
    * `VIDEO_FILE_PATH`, `BUILDING_FOOTPRINTS_GEOJSON_PATH`, `BUILDING_DAMAGE_CSV_PATH`.
    * `DEPLOYMENT_BASE_OUTPUT_DIR`: Main directory for this script's outputs.
    * **Target Building Identification:**
        * Set `TARGET_BLD_ID` (if known).
        * OR set `TARGET_LATITUDE` and `TARGET_LONGITUDE` for the script to find the nearest building.
    * `PREDETERMINED_PANO_ZERO_OFFSET`: This can be set in case the offset value is clearly known, either from visual assumption or if its set fixed during data collection.
    * Other stage parameters as needed.

* **Run `deploy.py`:**
    From the project root directory (`pano_process`):
    ```bash
    python deploy.py
    ```
    *(Similarly, if import errors, try `cd ..` then `python -m pano_process.deploy` with appropriate package setup.)*

---

## 📁 Expected Output Structure

Outputs from `main.py` are typically organized within the `BASE_OUTPUT_DIR` (e.g., `./output/`) as follows:

* `01_sampled_images/`: Raw sampled JPGs and `mapillary_image_description.json`.
* `02_blurred_images/`: Blurred JPGs (if blurring stage is run) and `blur_log.json`.
* `03_intermediate_data/`: CSV file (`pano_building_facade_matches.csv`) from façade processing.
* `04_rotated_panoramas/`: Rotated JPGs and `rotated_panoramas_metadata.json`.
* `05_cube_faces/`: Extracted PNG cube faces and `cube_faces_metadata.json`.
* `06_sorted_outputs_by_building/`:
    * `<BLD_ID>/`: Subfolder for each building.
        * `building_info.json`: Consolidated metadata for the building.
        * `*.png`: Cube faces related to this building.

Outputs from `deploy.py` will be in the `DEPLOYMENT_BASE_OUTPUT_DIR`, in a subfolder named after the target building ID, containing the sorted cube faces and `building_info.json`.

---

## 📜 License

This project is licensed under the terms of the MIT License. 
---

## 🧰 Core Scripts Overview

(Your project root `pano_process/` contains these scripts)

* `config.py`: Central configuration for model paths, default parameters.
* `utils.py`: Shared utility functions used across different scripts.
* `sample.py`: Handles video frame sampling using `mapillary_tools`.
* `privacy_blur.py`: Anonymizes images by blurring faces and license plates.
* `offset.py`: Interactively measures camera yaw offset.
* `process_facade.py`: Identifies building façades near panoramas.
* `rotate.py`: Rotates panoramas to face identified façades.
* `extract_cube.py`: Extracts 2D cube faces from 360° panoramas.
* `sort.py`: Sorts extracted cube faces into per-building folders.
* `main.py`: Orchestrates the full processing pipeline.
* `deploy.py`: Standalone script for processing data related to a single building.
