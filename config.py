# config.py
import os
import torch

# --- MODEL CONFIGURATIONS ---
# Ensure this path points to the directory where your models are stored.
# You might need to adjust this path based on where you run the scripts from,
# or make it an absolute path.
MODELS_DIR = "./models" # Assuming a 'models' folder in the same directory as pipeline_library or project root.

FACE_WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolov8n-face.pt")
PLATE_WEIGHTS_PATH = os.path.join(MODELS_DIR, "license_plate_detector.pt")
SAM_CHECKPOINT_PATH = os.path.join(MODELS_DIR, "sam2.1_hiera_tiny.pt")
SAM_HF_MODEL_NAME = "facebook/sam2-hiera-tiny"

# --- MAPILLARY TOOLS ---
# If 'mapillary_tools' is not in your system PATH, specify the full executable path here.
MAPILLARY_TOOLS_PATH = "mapillary_tools"

# --- DEFAULT PARAMETERS for pipeline stages ---
DEFAULT_SAMPLING_DISTANCE_M = 2.0
DEFAULT_MAX_FACADE_DIST_M = 35.0
DEFAULT_FRONTAL_TOL_DEG = 45.0
DEFAULT_ROTATION_DIST_CUTOFF_M = 35.0 # Used in panorama_rotator, often same as MAX_FACADE_DIST_M
DEFAULT_PANO_ZERO_OFFSET = 0.0 # This will be updated by offset_analyzer if run

# Cube extraction: names in order by equi2cube with cube_format='list'
# ["front", "right", "back", "left", "top", "bottom"]
DEFAULT_FACES_TO_SAVE = ["front"]
ALL_POSSIBLE_CUBE_FACES = ["front", "right", "back", "left", "top", "bottom"]

# --- CUDA DEVICE ---
DEVICE = "cuda" # if torch.cuda.is_available() is True else 'cpu' # "cuda" or "cpu"