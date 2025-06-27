# Panoramic Façade Processing Pipeline

This project provides a comprehensive pipeline for processing 360° panoramic video footage to extract building façade imagery. It includes stages for sampling frames from videos, anonymizing images by blurring sensitive information (faces, license plates), identifying building façades from geospatial data, rotating panoramas to face these façades, extracting 2D cube faces, and sorting the final images into per-building folders with rich metadata.

---

## 🌟 Features

* **Multi-Video Processing:** Automatically processes all videos within a specified directory.
* **Flexible Input:** Can start from raw video files or directories of pre-sampled panoramic images.
* **Image Anonymization:** Detects and blurs faces and license plates using YOLOv8 and SAM models.
* **Intelligent Offset Calibration:** Offers an interactive tool to measure camera offset, which can be applied to a single video or shared across an entire batch.
* **Geospatial Façade Matching:** Identifies relevant building façades from a GeoJSON dataset based on panorama locations.
* **Targeted Panorama Rotation:** Orients panoramas to directly face identified building façades.
* **Cube Face Extraction:** Converts rotated 360° panoramas into selected 2D cubemap faces (e.g., front, left, right).
* **Organized Outputs:** Sorts processed images and metadata into a clean, per-building directory structure.
* **On-Demand Extraction:** Includes a dedicated script (`get_building.py`) to quickly extract images for a single, specific building by ID or coordinates.

---

## 🛠️ Prerequisites

* Python 3.8+
* Conda (for environment management)
* Git (for cloning the repository)
* `mapillary_tools` command-line utility.
* Access to a CUDA-enabled GPU is highly recommended for faster model inference, but CPU is also supported.

---

## ⚙️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/fhasanm/pano_process.git](https://github.com/fhasanm/pano_process.git)
    cd pano_process
    ```

2.  **Create and Activate Conda Environment:**
    This process uses a Conda specification file for core dependencies and a Pip requirements file for additional packages.
    ```bash
    # Create the environment from the spec file
    conda create --name lafire_env --file spec-file.txt

    # Activate the new environment
    conda activate lafire_env

    # Install the remaining packages with pip
    pip install -r requirements-pip.txt
    ```

3.  **Install `mapillary_tools`:**
    If not already installed by the previous step, ensure `mapillary_tools` is available in your activated environment.
    ```bash
    pip install mapillary-tools
    ```
    Verify the installation:
    ```bash
    mapillary_tools --version
    ```

4.  **Download Models:**
    The pipeline requires pre-trained models for object detection and segmentation. Download them and place them into the `models/` directory.

    * **Download Link:** [Model Files (YOLO & SAM)](https://drive.google.com/drive/folders/1xFkKj4-fnDxxQuLy1cQjXGFRJBunOZUz?usp=sharing)
    * **Important:** After downloading, ensure the following files are present in your local `models/` directory:
        * `yolov8n-face.pt`
        * `license_plate_detector.pt`
        * `sam2.1_hiera_tiny.pt`

5.  **Download Input Data:**
    Download the necessary building footprint and attribute data and place it into the `data/` directory.

    * **Download Link:** [Data Files](https://drive.google.com/drive/folders/1soyVTu883yjU1dJxuahBrSK3lNtnUnAn?usp=sharing)
    * **Important:** After downloading, ensure the following files are present in your local `data/` directory:
        * `2023_Buildings_with_DINS_data.geojson`
        * `2023_Buildings_with_DINS_data.csv`
    * **Input Videos:** Place your own `.mp4` video files into the `data/input_video/` directory (or any directory you specify in the scripts).

6.  **Review Configuration (`config.py`):**
    Open `config.py` and verify that the paths and default parameters are correct for your system. Pay special attention to:
    * `MAPILLARY_TOOLS_PATH`: If `mapillary_tools` is not in your system PATH, provide the full executable path here.
    * `DEVICE`: Set to `"cuda"` to use GPU or `"cpu"`.

---

## 🚀 Usage

There are two main ways to run the pipeline:

### 1. Full Pipeline for All Videos (`main.py`)

This script processes a directory of videos through all selected stages, organizing outputs into per-video subfolders.

* **Configure `main.py`:**
    Open `main.py` and adjust the settings in the "USER CONFIGURATION" section:
    * `VIDEO_DIRECTORY_PATH`: Path to the folder containing your input videos.
    * `SAME_OFFSET_FOR_ALL_VIDEOS`: Set to `True` to measure camera offset once (on the first video) and apply it to all. Set to `False` to run the interactive measurement for each video.
    * `BASE_OUTPUT_DIR`: Main directory where all per-video output folders will be created.
    * **Stage Execution Flags (`RUN_STAGE_X`):** Set these to `True` or `False` to control the workflow.

* **Run `main.py`:**
    With your `lafire_env` environment activated, run the script from the project root directory:
    ```bash
    python main.py
    ```

### 2. Targeted Extraction for a Single Building (`get_building.py`)

This script is for on-demand processing. It takes a single source (video or image directory) and generates outputs for only one specific building.

* **Configure `get_building.py`:**
    Open `get_building.py` and adjust its "USER CONFIGURATION" section:
    * `INPUT_SOURCE_PATH`: Path to your source data (can be a single video file or a directory of pre-sampled images).
    * `INPUT_IS_VIDEO`: Set to `True` if the source is a video, `False` if it's a directory.
    * **Target Building Identification:**
        * Set `TARGET_BLD_ID` if known.
        * OR set `TARGET_LATITUDE` and `TARGET_LONGITUDE` for the script to find the nearest building.
    * `KNOWN_PANO_ZERO_OFFSET_DEG`: Set this to a number (e.g., `179.5`) if the offset is known. Set it to `None` to trigger the interactive measurement window.
    * **Desired Outputs (`GET_...` flags):** Set `GET_ROTATED_PANOS` or `GET_CUBE_FACES` to `True` to select your desired outputs.

* **Run `get_building.py`:**
    From the project root directory:
    ```bash
    python get_building.py
    ```

---

## 📁 Expected Output Structure

### For `main.py`:

Outputs are organized inside the `BASE_OUTPUT_DIR`, with a subfolder for each processed video:


<BASE_OUTPUT_DIR>/
└── <VIDEO_1_NAME>/
├── 01_sampled_images/
├── 02_blurred_images/
├── 03_intermediate_data/
├── 04_rotated_panoramas/
├── 05_cube_faces/
└── 07_sorted_by_building/
└── <BLD_ID>/
├── building_info.json
└── *.png
└── <VIDEO_2_NAME>/
└── ... (similar structure)


### For `get_building.py`:

Outputs are organized inside the `FINAL_OUTPUT_DIR`, in a subfolder named after the target building:


<FINAL_OUTPUT_DIR>/
└── <TARGET_BLD_ID>/
├── rotated_panoramas/  (if requested)
│   ├── rotated_panoramas_metadata.json
│   └── *.jpg
└── sorted_cube_faces/  (if requested)
└── <BLD_ID>/
├── building_info.json
└── *.png


---

## 📜 License

This project is licensed under the terms of the MIT License.

---

## 🧰 Core Scripts Overview

* `config.py`: Central configuration for model paths and default pipeline parameters.
* `utils.py`: Shared utility functions used across different scripts.
* `sample.py`: Handles video frame sampling using `mapillary_tools`.
* `privacy_blur.py`: Anonymizes images by blurring faces and license plates.
* `offset.py`: Interactively measures camera yaw offset.
* `process_facade.py`: Identifies building façades near panoramas from geospatial data.
* `rotate.py`: Rotates panoramas to face identified façades.
* `extract_cube.py`: Extracts 2D cube faces from 360° panoramas.
* `sort.py`: Sorts extracted cube faces into per-building folders with rich metadata.
* `main.py`: Orchestrates the full processing pipeline for multiple videos.
* `get_building.py`: Standalone script for on-demand extraction for a single building.
