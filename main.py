#!/usr/bin/env python3
"""
Main Pipeline Orchestrator
------------------------------------------
This script drives the entire image processing pipeline for multiple videos.
It allows users to configure run-specific parameters and select which stages
of the pipeline to execute, while pulling default settings from config.py.
"""
import os
import sys
from pathlib import Path

# Import default configurations and library modules
import config
from utils import ensure_dir_exists
import sample, privacy_blur, offset, process_facade, rotate, extract_cube, sort


def main():
    # ========================== USER CONFIGURATION ==========================
    # --- Essential Input & Output Paths ---
    
    VIDEO_DIRECTORY_PATH = Path("C:/Users/cviss/Downloads/lafire_video")  # REQUIRED
    BUILDING_FOOTPRINTS_GEOJSON_PATH = Path("./data/2023_Buildings_with_DINS_data.geojson")
    BUILDING_DAMAGE_CSV_PATH = Path("./data/2023_Buildings_with_DINS_data.csv")
    BASE_OUTPUT_DIR = Path("./output")

    # --- Run-Specific Behavior ---
    SAME_OFFSET_FOR_ALL_VIDEOS = True
    COPY_FILES_IN_SORTING = True

    # --- Stage Execution Flags (Set to True to run, False to skip) ---
    RUN_STAGE_1_VIDEO_SAMPLING = True
    RUN_STAGE_2_IMAGE_BLURRING = False
    RUN_STAGE_3_OFFSET_MEASUREMENT = True
    RUN_STAGE_4_FACADE_PROCESSING = True
    RUN_STAGE_5_PANORAMA_ROTATION = True
    RUN_STAGE_6_CUBE_EXTRACTION = True
    RUN_STAGE_7_OUTPUT_SORTING = True

    # --- Run-Specific Parameter Overrides ---
    pano_zero_offset_for_rotation = config.DEFAULT_PANO_ZERO_OFFSET
    cube_faces_to_extract = config.DEFAULT_FACES_TO_SAVE
    # ======================= END OF USER CONFIGURATION =======================

    print("Starting Multi-Video Pipeline Processing...")
    ensure_dir_exists(BASE_OUTPUT_DIR)

    video_dir = Path(VIDEO_DIRECTORY_PATH)
    if not video_dir.is_dir():
        print(f"Error: Video directory not found at '{video_dir}'. Terminating.")
        return
    video_files = sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.avi')])
    if not video_files:
        print(f"No video files found in '{video_dir}'. Terminating.")
        return
    print(f"Found {len(video_files)} videos to process.")

    shared_camera_offset = None

    for video_path in video_files:
        video_name = video_path.stem
        video_output_dir = BASE_OUTPUT_DIR / video_name
        print(f"\n{'='*20} PROCESSING VIDEO: {video_name} {'='*20}")
        ensure_dir_exists(video_output_dir)

        # --- Per-Video State ---
        current_pano_source_dir = None
        mapillary_meta_json_path = None
        facade_matches_csv = None
        rotated_panos_meta_json = None
        cube_faces_meta_json = None

        # --- STAGE 1: Video Sampling ---
        sampled_images_output_dir = video_output_dir / "01_sampled_images"
        if RUN_STAGE_1_VIDEO_SAMPLING:
            print("\n--- Running Stage 1: Video Sampling ---")
            sampled_dir, meta_path = sample.sample_video_by_distance(
                video_path=str(video_path),
                base_output_dir=str(video_output_dir),
                distance_m=config.DEFAULT_SAMPLING_DISTANCE_M,
                mapillary_tools_path=config.MAPILLARY_TOOLS_PATH
            )
            if not sampled_dir:
                print(f"Video sampling failed for {video_name}. Skipping this video.")
                continue
            current_pano_source_dir = Path(sampled_dir)
            mapillary_meta_json_path = Path(meta_path)
        else:
            print("\n--- Skipping Stage 1: Video Sampling ---")
            mapillary_meta_json_path = sampled_images_output_dir / "mapillary_image_description.json"
            if not sampled_images_output_dir.is_dir() or not mapillary_meta_json_path.is_file():
                print(f"Required outputs from skipped Stage 1 not found for {video_name}. Skipping.")
                continue
            current_pano_source_dir = sampled_images_output_dir

        # --- STAGE 2: Image Blurring ---
        if RUN_STAGE_2_IMAGE_BLURRING:
            print("\n--- Running Stage 2: Image Blurring ---")
            blurred_dir, _ = privacy_blur.blur_equirectangular_images(
                source_dir=str(current_pano_source_dir),
                base_output_dir=str(video_output_dir),
                face_weights_path=config.FACE_WEIGHTS_PATH,
                plate_weights_path=config.PLATE_WEIGHTS_PATH,
                sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
                sam_hf_model_name=config.SAM_HF_MODEL_NAME,
                device=config.DEVICE
            )
            if blurred_dir:
                current_pano_source_dir = Path(blurred_dir)
        else:
            print("\n--- Skipping Stage 2: Image Blurring ---")

        # --- STAGE 3: Offset Measurement ---
        if RUN_STAGE_3_OFFSET_MEASUREMENT:
            if SAME_OFFSET_FOR_ALL_VIDEOS and shared_camera_offset is not None:
                pano_zero_offset_for_rotation = shared_camera_offset
            else:
                measured_offset = offset.measure_yaw_offset_interactively(
                    panoramas_image_dir=str(current_pano_source_dir),
                    mapillary_image_description_json_path=str(mapillary_meta_json_path)
                )
                if measured_offset is not None:
                    pano_zero_offset_for_rotation = measured_offset
                    if SAME_OFFSET_FOR_ALL_VIDEOS:
                        shared_camera_offset = measured_offset
        print(f"Using offset for rotation: {pano_zero_offset_for_rotation:.2f}°")

        # --- STAGE 4: Façade Processing ---
        if RUN_STAGE_4_FACADE_PROCESSING:
            facade_matches_csv = process_facade.process_building_footprints(
                mapillary_image_description_json_path=str(mapillary_meta_json_path),
                footprint_geojson_path=BUILDING_FOOTPRINTS_GEOJSON_PATH,
                base_output_dir=str(video_output_dir),
                max_distance_to_building_m=config.DEFAULT_MAX_FACADE_DIST_M,
                frontal_view_tolerance_deg=config.DEFAULT_FRONTAL_TOL_DEG
            )
        else:
            facade_matches_csv = video_output_dir / "03_intermediate_data" / "pano_building_facade_matches.csv"
        if not Path(facade_matches_csv).is_file():
            print(f"Façade matches CSV not found for {video_name}. Cannot proceed."); continue

        # --- STAGE 5: Panorama Rotation ---
        if RUN_STAGE_5_PANORAMA_ROTATION:
            _, rotated_panos_meta_json = rotate.rotate_panoramas_to_facades(
                facade_matches_csv_path=facade_matches_csv,
                source_panoramas_dir=str(current_pano_source_dir),
                base_output_dir=str(video_output_dir),
                measured_camera_offset_deg=pano_zero_offset_for_rotation,
                distance_cutoff_m=config.DEFAULT_ROTATION_DIST_CUTOFF_M
            )
        else:
            rotated_panos_meta_json = video_output_dir / "04_rotated_panoramas" / "rotated_panoramas_metadata.json"
        if not Path(rotated_panos_meta_json).is_file():
            print(f"Rotated panorama metadata not found for {video_name}. Cannot proceed."); continue

        # --- STAGE 6: Cube Face Extraction ---
        if RUN_STAGE_6_CUBE_EXTRACTION:
            faces_to_get = cube_faces_to_extract if cube_faces_to_extract else config.ALL_POSSIBLE_CUBE_FACES
            _, cube_faces_meta_json = extract_cube.extract_cubemap_faces(
                rotated_panoramas_meta_json_path=rotated_panos_meta_json,
                base_output_dir=str(video_output_dir),
                faces_to_extract=faces_to_get
            )
        else:
            cube_faces_meta_json = video_output_dir / "05_cube_faces" / "cube_faces_metadata.json"
        if not Path(cube_faces_meta_json).is_file():
            print(f"Cube face metadata not found for {video_name}. Cannot proceed."); continue
            
        # --- STAGE 7: OUTPUT SORTING (now inside the loop) ---
        if RUN_STAGE_7_OUTPUT_SORTING:
            print("\n--- Running Stage 7: Output Sorting ---")
            # **THE FIX for output location**:
            # The final sorted directory is now specific to this video.
            final_sorted_output_dir = video_output_dir / "07_sorted_by_building"
            
            sort.sort_cube_faces_by_building(
                cube_faces_metadata_json_path=str(cube_faces_meta_json),
                building_damage_csv_path=str(BUILDING_DAMAGE_CSV_PATH),
                # Pass the specific destination directory
                final_sorted_dir=str(final_sorted_output_dir),
                move_files=(not COPY_FILES_IN_SORTING)
            )
        else:
            print("\n--- Skipping Stage 7: Output Sorting ---")

    print("\n\n================ Pipeline Finished ================")

if __name__ == "__main__":
        main()
