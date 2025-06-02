# main.py
#!/usr/bin/env python3
"""
Main Pipeline Orchestrator
--------------------------
This script drives the entire image processing pipeline, from video sampling
to sorted cube face outputs. It allows users to configure various parameters
and select which stages of the pipeline to execute.

Core Stages:
1. Video Sampling: Extracts frames from a video based on distance.
2. Image Blurring (Optional): Anonymizes panoramas by blurring faces and license plates.
3. Offset Measurement (Interactive, Optional): Measures camera yaw offset relative to vehicle.
4. Façade Processing: Identifies building façades near panoramas.
5. Panorama Rotation: Rotates panoramas to face identified façades.
6. Cube Face Extraction: Extracts selected 2D faces from rotated 360 panoramas.
7. Output Sorting: Sorts extracted cube faces into per-building folders with metadata.

Configuration:
All input paths, parameters, and stage execution flags are set in the
"USER CONFIGURATION" section below.

Output Structure:
All outputs are saved into subdirectories within the `BASE_OUTPUT_DIR`.
- 01_sampled_images/
  - mapillary_image_description.json (metadata from sampling)
  - *.jpg (sampled panorama images)
- 02_blurred_images/ (if blurring is enabled)
  - blur_log.json
  - *_blurred.jpg (blurred panorama images)
- 03_intermediate_data/
  - pano_building_facade_matches.csv (details of façades matched to panoramas)
- 04_rotated_panoramas/
  - rotated_panoramas_metadata.json
  - *_ROT.jpg (panoramas rotated to face façades)
- 05_cube_faces/
  - cube_faces_metadata.json
  - *_front.png, *_left.png, etc. (extracted 2D cube faces)
- 06_sorted_outputs_by_building/
  - <BLD_ID>/
    - building_info.json (consolidated metadata for the building)
    - *.png (cube faces related to this building)
  - ... (other building folders)

How to Use:
1. Ensure all dependencies are installed (cv2, numpy, torch, ultralytics, sam2,
   equilib, geopandas, pandas, matplotlib, pyproj, tqdm).
2. Place model files (YOLO, SAM) in the directory specified by `config.MODELS_DIR`.
3. Configure the "USER CONFIGURATION" section in this script:
    - Set `VIDEO_FILE_PATH` to your input MP4 video.
    - Set `BUILDING_FOOTPRINTS_GEOJSON_PATH` to your building data.
    - Set `BUILDING_DAMAGE_CSV_PATH` for damage assessment data (optional for sorting).
    - Adjust `BASE_OUTPUT_DIR` for where to save results.
    - Modify parameters for each stage as needed (sampling distance, thresholds, etc.).
    - Set `RUN_STAGE_X` flags to True/False to control which parts of the pipeline run.
4. Run the script: `python path/to/pipeline_library/main.py`

Notes:
- Metadata (JSON, CSV files) is crucial as it carries information between stages.
- If a stage is skipped, subsequent stages that depend on its output might fail or
  produce incorrect results unless the required input files already exist from a
  previous run.
- The interactive offset measurement stage will display a plot; user interaction is required.
  The measured offset is then used in the panorama rotation stage.
"""
import os
import sys
import json
import config # Loads default model paths and parameters

from utils import ensure_dir_exists
import sample, privacy_blur, offset, process_facade, rotate, extract_cube, sort


def main():
    # ========================== USER CONFIGURATION ==========================
    # --- Essential Input Paths ---
    VIDEO_FILE_PATH = "/home/fuad/Downloads/VID_20250327_120940_20250507131706.mp4" # REQUIRED
    BUILDING_FOOTPRINTS_GEOJSON_PATH = "./data/2023_Buildings_with_DINS_data.geojson" # REQUIRED for facade processing onwards
    BUILDING_DAMAGE_CSV_PATH = "./data/2023_Buildings_with_DINS_data.csv" # REQUIRED for sorting stage if damage info is desired

    # --- Output Directory ---
    BASE_OUTPUT_DIR = "./output" # All processed data will go into subfolders here

    # --- Stage Execution Flags (Set to True to run, False to skip) ---
    RUN_STAGE_1_VIDEO_SAMPLING = True
    RUN_STAGE_2_IMAGE_BLURRING = False # If True, subsequent stages use blurred images
    RUN_STAGE_3_OFFSET_MEASUREMENT = True # Interactive. Result used in STAGE_5.
    RUN_STAGE_4_FACADE_PROCESSING = True #Keep it True
    RUN_STAGE_5_PANORAMA_ROTATION = True
    RUN_STAGE_6_CUBE_EXTRACTION = True
    RUN_STAGE_7_OUTPUT_SORTING = True

    # --- Stage-Specific Parameters (Can override defaults from config.py) ---
    # Stage 1: Video Sampling
    SAMPLING_DISTANCE_M = config.DEFAULT_SAMPLING_DISTANCE_M # Meters

    # Stage 2: Image Blurring (Models are set in config.py)
    YOLO_CONF_THRESHOLD = 0.25
    MIN_BOX_SIZE_PX = 15
    BLUR_KERNEL_SIZE = 31 # Must be odd

    # Stage 3: Offset Measurement (Interactive)
    # No specific params here other than inputs derived from previous stages.
    # The measured offset will override PANO_ZERO_OFFSET_FOR_ROTATION if this stage runs.

    # Stage 4: Façade Processing
    MAX_DISTANCE_TO_BUILDING_M = config.DEFAULT_MAX_FACADE_DIST_M # meters
    FRONTAL_VIEW_TOLERANCE_DEG = config.DEFAULT_FRONTAL_TOL_DEG   # degrees

    # Stage 5: Panorama Rotation
    # This will be updated by Stage 3 if it runs and succeeds.
    # Set a fallback/manual value if Stage 3 is skipped but rotation is run.
    PANO_ZERO_OFFSET_FOR_ROTATION = config.DEFAULT_PANO_ZERO_OFFSET # degrees
    ROTATION_DISTANCE_CUTOFF_M = config.DEFAULT_ROTATION_DIST_CUTOFF_M # meters

    # Stage 6: Cube Extraction
    # Options: "front", "right", "back", "left", "top", "bottom" or empty list for all.
    CUBE_FACES_TO_EXTRACT = config.DEFAULT_FACES_TO_SAVE # Example: extract front, left, right
    # CUBE_FACES_TO_EXTRACT = ["front"] # For only front faces
    # CUBE_FACES_TO_EXTRACT = [] # For all 6 faces

    # Stage 7: Output Sorting
    MOVE_FILES_IN_SORTING = False # False to copy, True to move cube faces during sorting

    # --- Advanced Configuration ---
    MAPILLARY_TOOLS_EXEC_PATH = config.MAPILLARY_TOOLS_PATH # Path to mapillary_tools
    COMPUTATION_DEVICE = config.DEVICE # "cuda" or "cpu" (mainly for blurring)
    # Model paths for blurring are taken from config.py (FACE_WEIGHTS_PATH, etc.)
    # ======================= END OF USER CONFIGURATION =======================

    print("Starting pipeline processing...")
    ensure_dir_exists(BASE_OUTPUT_DIR)

    # --- Variables to store intermediate paths and results ---
    sampled_images_output_dir = None
    mapillary_meta_json_path = None # From video sampling
    
    # This will point to either sampled_images_output_dir or blurred_images_output_dir
    current_pano_image_source_dir = None 
    
    blurred_images_output_dir = None
    # blur_log_path = None # If needed

    facade_matches_csv = None # From facade processing
    
    rotated_panos_output_dir = None
    rotated_panos_meta_json = None # From panorama rotation
    
    cube_faces_dir = None
    cube_faces_meta = None # From cube extraction

    # --- STAGE 1: Video Sampling ---
    if RUN_STAGE_1_VIDEO_SAMPLING:
        print("\n--- Running Stage 1: Video Sampling ---")
        if not os.path.isfile(VIDEO_FILE_PATH):
            print(f"Error: Video file not found at {VIDEO_FILE_PATH}. Terminating.")
            return

        sampled_images_output_dir, mapillary_meta_json_path = sample.sample_video_by_distance(
            video_path=VIDEO_FILE_PATH,
            base_output_dir=BASE_OUTPUT_DIR,
            distance_m=SAMPLING_DISTANCE_M,
            mapillary_tools_path=MAPILLARY_TOOLS_EXEC_PATH
        )
        if not sampled_images_output_dir or not mapillary_meta_json_path:
            print("Video sampling failed. Terminating.")
            return
        current_pano_image_source_dir = sampled_images_output_dir
    else:
        print("\n--- Skipping Stage 1: Video Sampling ---")
        # Attempt to find pre-existing outputs if skipping
        sampled_images_output_dir = os.path.join(BASE_OUTPUT_DIR, "01_sampled_images")
        mapillary_meta_json_path = os.path.join(sampled_images_output_dir, "mapillary_image_description.json")
        current_pano_image_source_dir = sampled_images_output_dir
        if not os.path.isdir(sampled_images_output_dir) or not os.path.isfile(mapillary_meta_json_path):
            print("Skipped sampling, but required outputs (sampled images dir or mapillary_image_description.json) not found. Subsequent stages may fail.")
        else:
            print(f"Using existing sampled images from: {sampled_images_output_dir}")
            print(f"Using existing Mapillary metadata: {mapillary_meta_json_path}")


    # --- STAGE 2: Image Blurring (Optional) ---
    if RUN_STAGE_2_IMAGE_BLURRING:
        print("\n--- Running Stage 2: Image Blurring ---")
        if not current_pano_image_source_dir or not os.path.isdir(current_pano_image_source_dir):
            print("Error: Source directory for blurring (sampled images) not available. Skipping blurring.")
        else:
            blurred_images_output_dir, _ = privacy_blur.blur_equirectangular_images(
                source_dir=current_pano_image_source_dir, # Input is the output of sampling
                base_output_dir=BASE_OUTPUT_DIR,
                face_weights_path=config.FACE_WEIGHTS_PATH,
                plate_weights_path=config.PLATE_WEIGHTS_PATH,
                sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
                sam_hf_model_name=config.SAM_HF_MODEL_NAME,
                device=COMPUTATION_DEVICE,
                yolo_confidence_threshold=YOLO_CONF_THRESHOLD,
                min_box_size_px=MIN_BOX_SIZE_PX,
                blur_kernel_size=BLUR_KERNEL_SIZE
            )
            if blurred_images_output_dir:
                current_pano_image_source_dir = blurred_images_output_dir # Subsequent stages use blurred images
                print(f"Blurred images ready in: {current_pano_image_source_dir}")
            else:
                print("Image blurring failed. Subsequent stages will use unblurred images if available.")
                # current_pano_image_source_dir remains as sampled_images_output_dir
    else:
        print("\n--- Skipping Stage 2: Image Blurring ---")
        # If blurring is skipped, check if blurred images exist from a previous run
        # and if the user intends to use them. For simplicity now, if blurring is skipped,
        # we assume current_pano_image_source_dir (from sampling) is the one to use.
        # A more complex setup could allow choosing pre-blurred images.
        blurred_output_candidate_dir = os.path.join(BASE_OUTPUT_DIR, "02_blurred_images")
        if os.path.isdir(blurred_output_candidate_dir) and any(f.endswith("_blurred.jpg") for f in os.listdir(blurred_output_candidate_dir)):
            print(f"Found existing blurred images at {blurred_output_candidate_dir}. Consider enabling blurring or adjusting source paths if these are intended for use.")


    # --- STAGE 3: Offset Measurement (Interactive, Optional) ---
    if RUN_STAGE_3_OFFSET_MEASUREMENT:
        print("\n--- Running Stage 3: Offset Measurement (Interactive) ---")
        if not current_pano_image_source_dir or not mapillary_meta_json_path:
            print("Error: Cannot run offset measurement. Panorama source directory or metadata JSON is missing.")
        else:
            measured_offset = offset.measure_yaw_offset_interactively(
                panoramas_image_dir=current_pano_image_source_dir,
                mapillary_image_description_json_path=mapillary_meta_json_path
            )
            if measured_offset is not None: # Can be 0.0, which is valid
                PANO_ZERO_OFFSET_FOR_ROTATION = measured_offset # Override with measured value
                print(f"Using measured offset for rotation: {PANO_ZERO_OFFSET_FOR_ROTATION:+.2f}°")
            else:
                print(f"Offset measurement failed or skipped by user. Using default/configured offset: {PANO_ZERO_OFFSET_FOR_ROTATION:+.2f}°")
    else:
        print("\n--- Skipping Stage 3: Offset Measurement ---")
        print(f"Using pre-configured PANO_ZERO_OFFSET_FOR_ROTATION: {PANO_ZERO_OFFSET_FOR_ROTATION:+.2f}°")

    # --- STAGE 4: Façade Processing ---
    if RUN_STAGE_4_FACADE_PROCESSING:
        print("\n--- Running Stage 4: Façade Processing ---")
        if not mapillary_meta_json_path or not os.path.isfile(mapillary_meta_json_path):
            print("Error: Mapillary metadata JSON path not available. Cannot run façade processing.")
        elif not os.path.isfile(BUILDING_FOOTPRINTS_GEOJSON_PATH):
            print(f"Error: Building footprints GeoJSON not found at {BUILDING_FOOTPRINTS_GEOJSON_PATH}. Cannot run façade processing.")
        else:
            facade_matches_csv = process_facade.process_building_footprints(
                mapillary_image_description_json_path=mapillary_meta_json_path,
                footprint_geojson_path=BUILDING_FOOTPRINTS_GEOJSON_PATH,
                base_output_dir=BASE_OUTPUT_DIR,
                max_distance_to_building_m=MAX_DISTANCE_TO_BUILDING_M,
                frontal_view_tolerance_deg=FRONTAL_VIEW_TOLERANCE_DEG
            )
            if not facade_matches_csv:
                print("Façade processing failed or produced no output. Subsequent stages might fail.")
    else:
        print("\n--- Skipping Stage 4: Façade Processing ---")
        # Attempt to find pre-existing output
        facade_matches_csv = os.path.join(BASE_OUTPUT_DIR, "03_intermediate_data", "pano_building_facade_matches.csv")
        if not os.path.isfile(facade_matches_csv):
            print(f"Skipped façade processing, but its output CSV ({facade_matches_csv}) not found. Subsequent stages may fail.")
        else:
             print(f"Using existing facade matches CSV: {facade_matches_csv}")


    # --- STAGE 5: Panorama Rotation ---
    if RUN_STAGE_5_PANORAMA_ROTATION:
        print("\n--- Running Stage 5: Panorama Rotation ---")
        if not facade_matches_csv or not os.path.isfile(facade_matches_csv):
            print("Error: Façade matches CSV not available. Cannot run panorama rotation.")
        elif not current_pano_image_source_dir or not os.path.isdir(current_pano_image_source_dir):
            print("Error: Source directory for panoramas (sampled or blurred) not available. Cannot run rotation.")
        else:
            rotated_panos_output_dir, rotated_panos_meta_json = rotate.rotate_panoramas_to_facades(
                facade_matches_csv_path=facade_matches_csv,
                source_panoramas_dir=current_pano_image_source_dir, # Use latest images (blurred if available)
                base_output_dir=BASE_OUTPUT_DIR,
                measured_camera_offset_deg=PANO_ZERO_OFFSET_FOR_ROTATION,
                distance_cutoff_m=ROTATION_DISTANCE_CUTOFF_M
            )
            if not rotated_panos_output_dir or not rotated_panos_meta_json:
                print("Panorama rotation failed. Subsequent stages might fail.")
    else:
        print("\n--- Skipping Stage 5: Panorama Rotation ---")
        # Attempt to find pre-existing outputs
        rotated_panos_output_dir = os.path.join(BASE_OUTPUT_DIR, "04_rotated_panoramas")
        rotated_panos_meta_json = os.path.join(rotated_panos_output_dir, "rotated_panoramas_metadata.json")
        if not os.path.isdir(rotated_panos_output_dir) or not os.path.isfile(rotated_panos_meta_json):
            print(f"Skipped panorama rotation, but its outputs ({rotated_panos_output_dir} or {rotated_panos_meta_json}) not found. Subsequent stages may fail.")
        else:
            print(f"Using existing rotated panoramas from: {rotated_panos_output_dir}")
            print(f"Using existing rotated panoramas metadata: {rotated_panos_meta_json}")


    # --- STAGE 6: Cube Face Extraction ---
    if RUN_STAGE_6_CUBE_EXTRACTION:
        print("\n--- Running Stage 6: Cube Face Extraction ---")
        if not rotated_panos_meta_json or not os.path.isfile(rotated_panos_meta_json):
            print("Error: Rotated panoramas metadata JSON not available. Cannot run cube face extraction.")
        else:
            if not CUBE_FACES_TO_EXTRACT: # If user provided empty list meaning "all"
                CUBE_FACES_TO_EXTRACT = config.ALL_POSSIBLE_CUBE_FACES
                print(f"Extracting ALL cube faces: {CUBE_FACES_TO_EXTRACT}")
            
            cube_faces_dir, cube_faces_meta = extract_cube.extract_cubemap_faces(
                rotated_panoramas_meta_json_path=rotated_panos_meta_json,
                base_output_dir=BASE_OUTPUT_DIR,
                faces_to_extract=CUBE_FACES_TO_EXTRACT
            )
            if not cube_faces_dir or not cube_faces_meta:
                print("Cube face extraction failed. Subsequent stages might fail.")
    else:
        print("\n--- Skipping Stage 6: Cube Face Extraction ---")
        # Attempt to find pre-existing outputs
        cube_faces_dir = os.path.join(BASE_OUTPUT_DIR, "05_cube_faces")
        cube_faces_meta = os.path.join(cube_faces_dir, "cube_faces_metadata.json")
        if not os.path.isdir(cube_faces_dir) or not os.path.isfile(cube_faces_meta):
            print(f"Skipped cube face extraction, but its outputs ({cube_faces_dir} or {cube_faces_meta}) not found. Subsequent stages may fail.")
        else:
            print(f"Using existing cube faces from: {cube_faces_dir}")
            print(f"Using existing cube faces metadata: {cube_faces_meta}")


    # --- STAGE 7: Output Sorting ---
    if RUN_STAGE_7_OUTPUT_SORTING:
        print("\n--- Running Stage 7: Output Sorting ---")
        if not cube_faces_meta or not os.path.isfile(cube_faces_meta):
            print("Error: Cube faces metadata JSON not available. Cannot run output sorting.")
        elif not os.path.isfile(BUILDING_DAMAGE_CSV_PATH):
            print(f"Warning: Building damage CSV for sorting not found at {BUILDING_DAMAGE_CSV_PATH}. Damage info will be 'N/A'.")
            # Proceed without damage info if CSV is missing
            sort.sort_cube_faces_by_building(
                cube_faces_metadata_json_path=cube_faces_meta,
                base_output_dir=BASE_OUTPUT_DIR,
                building_damage_csv_path="", # Pass empty path if not found
                move_files=MOVE_FILES_IN_SORTING
            )
        else:
            sort.sort_cube_faces_by_building(
                cube_faces_metadata_json_path=cube_faces_meta,
                base_output_dir=BASE_OUTPUT_DIR,
                building_damage_csv_path=BUILDING_DAMAGE_CSV_PATH,
                move_files=MOVE_FILES_IN_SORTING
            )
    else:
        print("\n--- Skipping Stage 7: Output Sorting ---")

    print("\nPipeline processing finished.")
    print(f"All selected outputs and intermediate files are in subdirectories of: {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()