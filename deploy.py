#!/usr/bin/env python3
"""
Standalone Deployment Script for Specific Building Outputs
----------------------------------------------------------
This script processes a video to generate sorted cube face outputs for a
*single specific building*. It is standalone and does not require outputs
from a previous `main.py` run, though it uses the same core processing functions.

Workflow:
1. Identifies the target building using a BLD_ID or by finding the closest
   building to a given latitude/longitude.
2. Samples the *entire* input video to extract panoramic frames and metadata.
3. Filters these panoramas to select only those near the target building.
4. (Optional) Interactively measures camera yaw offset using one of the filtered panoramas.
5. (Optional) Blurs the selected panoramas.
6. Processes façades for the selected panoramas against the target building.
7. Rotates the selected panoramas to face the identified façades of the target building.
8. Extracts specified cube faces from these rotated panoramas.
9. Sorts the extracted cube faces into a dedicated output directory for the
   target building, including a `building_info.json` with metadata.

Prerequisites:
- Same dependencies as `main.py` (OpenCV, PyTorch, ultralytics, sam2, equilib,
  geopandas, pandas, etc.).
- `mapillary_tools` installed and accessible.
- Model files (YOLO, SAM) correctly placed and configured in `config.py`
  (or `scripts/config.py` depending on your final structure).

Configuration:
All necessary paths, parameters, and target identification details must be
set in the "USER CONFIGURATION" section of this script.
"""
import os
import sys
import json
import shutil
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tempfile

# Assuming 'config.py' and 'utils.py' are in a location findable by Python,
# e.g., in the same directory as this script or in a 'scripts' subdirectory
# if the PYTHONPATH is set up or imports are adjusted accordingly.
# User's renamed scripts:
import config
from utils import (
    ensure_dir_exists,
    find_closest_building_by_latlon,
    calculate_distance_meters
)
import sample
import privacy_blur
import offset 
import process_facade
import rotate
import extract_cube
import sort


def deploy_single_building_standalone():
    # ========================== USER CONFIGURATION ==========================
    # --- Essential Input Paths ---
    VIDEO_FILE_PATH = "/home/fuad/Downloads/VID_20250327_120940_20250507131706.mp4"
    BUILDING_FOOTPRINTS_GEOJSON_PATH = "./data/2023_Buildings_with_DINS_data.geojson"
    BUILDING_DAMAGE_CSV_PATH = "./data/2023_Buildings_with_DINS_data.csv"

    DEPLOYMENT_BASE_OUTPUT_DIR = "./deployment_output"

    TARGET_BLD_ID = ""
    TARGET_LATITUDE = 34.194344300490954
    TARGET_LONGITUDE = -118.1477737109565


    MAX_DIST_FOR_LATLON_MATCH_M = 70.0
    MAX_PANO_DISTANCE_TO_TARGET_BLD_M = 100.0

    # --- Camera Offset Configuration ---
    # Set to True to run interactive offset measurement during this deployment run.
    # If False, PREDETERMINED_PANO_ZERO_OFFSET will be used.
    RUN_INTERACTIVE_OFFSET_MEASUREMENT_DEPLOY = True
    # Fallback offset if interactive measurement is disabled or fails.
    PREDETERMINED_PANO_ZERO_OFFSET = 0 # degrees

    # --- Stage Execution Flags & Parameters ---
    ENABLE_IMAGE_BLURRING = False # Set to False if you want to measure offset on unblurred images first
    YOLO_CONF_THRESHOLD_BLUR = 0.25
    MIN_BOX_SIZE_PX_BLUR = 15
    BLUR_KERNEL_SIZE = 31

    SAMPLING_DISTANCE_M = config.DEFAULT_SAMPLING_DISTANCE_M
    # GEOTAG_SOURCE_TYPE_DEPLOY = None
    # GEOTAG_SOURCE_PATH_DEPLOY = None

    MAX_FACADE_DIST_M_DEPLOY = config.DEFAULT_MAX_FACADE_DIST_M
    FRONTAL_TOL_DEG_DEPLOY = config.DEFAULT_FRONTAL_TOL_DEG

    ROTATION_DIST_CUTOFF_M_DEPLOY = config.DEFAULT_ROTATION_DIST_CUTOFF_M
    
    CUBE_FACES_TO_EXTRACT_DEPLOY = ["front"]

    MOVE_FILES_IN_FINAL_SORT = False

    MAPILLARY_TOOLS_EXEC_PATH = config.MAPILLARY_TOOLS_PATH
    COMPUTATION_DEVICE = config.DEVICE
    
    CLEANUP_TEMP_FULL_SAMPLING_DIR = True
    # ======================= END OF USER CONFIGURATION =======================

    print("Starting STANDALONE deployment for a single building...")
    ensure_dir_exists(DEPLOYMENT_BASE_OUTPUT_DIR)

    actual_target_bld_id = None
    target_building_geometry = None
    target_building_centroid_wgs84 = None
    pano_zero_offset_for_rotation = PREDETERMINED_PANO_ZERO_OFFSET # Initialize with predetermined

    if not os.path.isfile(BUILDING_FOOTPRINTS_GEOJSON_PATH):
        print(f"Error: Building footprints GeoJSON not found at {BUILDING_FOOTPRINTS_GEOJSON_PATH}.")
        return

    try:
        gdf_all_buildings = gpd.read_file(BUILDING_FOOTPRINTS_GEOJSON_PATH)
        if gdf_all_buildings.crs.to_epsg() != 4326:
            gdf_all_buildings = gdf_all_buildings.to_crs(epsg=4326)
    except Exception as e:
        print(f"Error reading building GeoJSON '{BUILDING_FOOTPRINTS_GEOJSON_PATH}': {e}")
        return

    id_column_name = 'BLD_ID' if 'BLD_ID' in gdf_all_buildings.columns else 'id'
    if id_column_name not in gdf_all_buildings.columns:
        print(f"Error: Neither 'BLD_ID' nor 'id' column found in {BUILDING_FOOTPRINTS_GEOJSON_PATH}.")
        return

    if TARGET_BLD_ID:
        actual_target_bld_id = str(TARGET_BLD_ID).strip()
        target_building_series = gdf_all_buildings[gdf_all_buildings[id_column_name].astype(str).str.strip() == actual_target_bld_id]
        if target_building_series.empty:
            print(f"Error: Target BLD_ID '{actual_target_bld_id}' not found in GeoJSON using column '{id_column_name}'.")
            return
        target_building_geometry = target_building_series.iloc[0].geometry
        target_building_centroid_wgs84 = target_building_geometry.centroid
        print(f"Targeting building by BLD_ID: {actual_target_bld_id}")
    elif TARGET_LATITUDE != 0.0 and TARGET_LONGITUDE != 0.0:
        print(f"Attempting to find closest building to LAT: {TARGET_LATITUDE}, LON: {TARGET_LONGITUDE}")
        bld_id_match, bld_lat, bld_lon, dist_m = find_closest_building_by_latlon(
            TARGET_LATITUDE, TARGET_LONGITUDE, gdf_all_buildings, MAX_DIST_FOR_LATLON_MATCH_M
        )
        if bld_id_match:
            actual_target_bld_id = str(bld_id_match)
            target_building_series = gdf_all_buildings[gdf_all_buildings[id_column_name].astype(str).str.strip() == actual_target_bld_id]
            if target_building_series.empty:
                print(f"Error: Could not re-fetch building {actual_target_bld_id} after matching by lat/lon.")
                return
            target_building_geometry = target_building_series.iloc[0].geometry
            target_building_centroid_wgs84 = Point(bld_lon, bld_lat)
            print(f"Found closest building: BLD_ID '{actual_target_bld_id}' at ({bld_lat:.6f}, {bld_lon:.6f}), distance: {dist_m:.2f}m.")
        else:
            print(f"No building found within {MAX_DIST_FOR_LATLON_MATCH_M}m of the provided coordinates.")
            return
    else:
        print("Error: No target specified. Please set TARGET_BLD_ID or TARGET_LATITUDE/LONGITUDE.")
        return

    if not actual_target_bld_id or target_building_geometry is None:
        print("Could not determine target building. Exiting.")
        return

    safe_bld_id_dirname = actual_target_bld_id.replace(os.sep, "_").replace(" ", "_").replace(":", "_")
    final_building_output_dir = os.path.join(DEPLOYMENT_BASE_OUTPUT_DIR, safe_bld_id_dirname)
    ensure_dir_exists(final_building_output_dir)

    with tempfile.TemporaryDirectory(prefix="pano_deploy_temp_") as temp_processing_dir_base:
        print(f"Using temporary processing directory: {temp_processing_dir_base}")

        print("\n--- Running Stage: Full Video Sampling ---")
        if not os.path.isfile(VIDEO_FILE_PATH):
            print(f"Error: Video file not found: {VIDEO_FILE_PATH}")
            return

        full_sampled_images_dir_in_temp, full_mapillary_meta_json_path = sample.sample_video_by_distance(
            video_path=VIDEO_FILE_PATH,
            base_output_dir=temp_processing_dir_base,
            distance_m=SAMPLING_DISTANCE_M,
            mapillary_tools_path=MAPILLARY_TOOLS_EXEC_PATH
            # geotag_source=GEOTAG_SOURCE_TYPE_DEPLOY, # Optional
            # geotag_source_path=GEOTAG_SOURCE_PATH_DEPLOY # Optional
        )
        if not full_sampled_images_dir_in_temp or not full_mapillary_meta_json_path:
            print("Full video sampling failed. Terminating.")
            return
        
        print("\n--- Running Stage: Filtering Panoramas near Target Building ---")
        try:
            with open(full_mapillary_meta_json_path, 'r', encoding='utf-8') as f:
                all_pano_metadata = json.load(f)
        except Exception as e:
            print(f"Error reading full Mapillary metadata '{full_mapillary_meta_json_path}': {e}")
            return

        filtered_pano_records = []
        active_panos_dir = os.path.join(temp_processing_dir_base, "01a_active_panos_for_target")
        ensure_dir_exists(active_panos_dir)

        print(f"Filtering {len(all_pano_metadata)} total panoramas for those within {MAX_PANO_DISTANCE_TO_TARGET_BLD_M}m of target building centroid ({target_building_centroid_wgs84.y:.5f}, {target_building_centroid_wgs84.x:.5f})...")
        for pano_meta in tqdm(all_pano_metadata, desc="Filtering Panos"):
            try:
                pano_lat = float(pano_meta["MAPLatitude"])
                pano_lon = float(pano_meta["MAPLongitude"])
                dist_to_target = calculate_distance_meters(
                    pano_lat, pano_lon,
                    target_building_centroid_wgs84.y, target_building_centroid_wgs84.x
                )

                if dist_to_target <= MAX_PANO_DISTANCE_TO_TARGET_BLD_M:
                    original_pano_path_str = pano_meta["filename"]
                    original_pano_path = Path(original_pano_path_str)
                    
                    if not original_pano_path.is_file():
                        json_dir = Path(full_mapillary_meta_json_path).parent
                        candidate_path = (json_dir / original_pano_path_str).resolve()
                        if candidate_path.is_file():
                            original_pano_path = candidate_path
                        else:
                            print(f"Warning: Panorama file from JSON '{original_pano_path_str}' not found. Skipping.")
                            continue
                        
                    new_pano_basename = original_pano_path.name
                    new_pano_path = os.path.join(active_panos_dir, new_pano_basename)
                    shutil.copy2(str(original_pano_path), new_pano_path)
                    
                    updated_meta_record = dict(pano_meta)
                    updated_meta_record["filename"] = new_pano_path
                    filtered_pano_records.append(updated_meta_record)

            except (KeyError, ValueError, TypeError) as e_filter:
                print(f"Warning: Skipping a panorama record due to missing/invalid data ('{str(e_filter)}'). Record: {str(pano_meta)[:100]}...")
                continue
        
        if not filtered_pano_records:
            print(f"No panoramas found near BLD_ID '{actual_target_bld_id}'. Deployment cannot proceed.")
            if CLEANUP_TEMP_FULL_SAMPLING_DIR and os.path.exists(full_sampled_images_dir_in_temp):
                 shutil.rmtree(full_sampled_images_dir_in_temp)
            return

        print(f"Found {len(filtered_pano_records)} panoramas relevant to the target building.")
        
        filtered_mapillary_meta_json_path = os.path.join(active_panos_dir, "filtered_mapillary_description.json")
        with open(filtered_mapillary_meta_json_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_pano_records, f, indent=2)

        # --- *** NEW STAGE: Interactive Offset Measurement (Optional) *** ---
        if RUN_INTERACTIVE_OFFSET_MEASUREMENT_DEPLOY:
            print("\n--- Running Stage: Interactive Offset Measurement (Deployment) ---")
            if not active_panos_dir or not filtered_mapillary_meta_json_path:
                print("Error: Cannot run offset measurement. Filtered panorama source directory or metadata JSON is missing.")
            else:
                # Ensure offset.py (or your renamed script) is imported correctly
                measured_offset = offset.measure_yaw_offset_interactively(
                    panoramas_image_dir=active_panos_dir, # Use the directory with filtered images
                    mapillary_image_description_json_path=filtered_mapillary_meta_json_path # Use metadata for filtered images
                )
                if measured_offset is not None: # Can be 0.0, which is valid
                    pano_zero_offset_for_rotation = measured_offset # Override with measured value
                    print(f"Using measured offset for rotation: {pano_zero_offset_for_rotation:+.2f}°")
                else:
                    print(f"Offset measurement failed or skipped by user. Using predetermined offset: {pano_zero_offset_for_rotation:+.2f}°")
        else:
            print("\n--- Skipping Stage: Interactive Offset Measurement (Deployment) ---")
            print(f"Using predetermined PANO_ZERO_OFFSET_FOR_ROTATION: {pano_zero_offset_for_rotation:+.2f}°")
        # --- *** END OF NEW STAGE *** ---

        current_pano_source_dir_for_processing = active_panos_dir
        current_mapillary_meta_for_facade_processing = filtered_mapillary_meta_json_path

        if ENABLE_IMAGE_BLURRING:
            print("\n--- Running Stage: Image Blurring (Targeted) ---")
            blurred_active_panos_dir, _ = privacy_blur.blur_equirectangular_images(
                source_dir=active_panos_dir, 
                base_output_dir=temp_processing_dir_base, 
                face_weights_path=config.FACE_WEIGHTS_PATH,
                plate_weights_path=config.PLATE_WEIGHTS_PATH,
                sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
                sam_hf_model_name=config.SAM_HF_MODEL_NAME,
                device=COMPUTATION_DEVICE,
                yolo_confidence_threshold=YOLO_CONF_THRESHOLD_BLUR,
                min_box_size_px=MIN_BOX_SIZE_PX_BLUR,
                blur_kernel_size=BLUR_KERNEL_SIZE
            )
            if blurred_active_panos_dir and os.path.isdir(blurred_active_panos_dir):
                current_pano_source_dir_for_processing = blurred_active_panos_dir
                print(f"Blurred active panoramas generated in: {current_pano_source_dir_for_processing}")
                
                print("Updating metadata to reflect blurred image paths for facade processing...")
                updated_records_for_blurred_files = []
                for record in filtered_pano_records:
                    original_basename_in_active_dir = Path(record["filename"]).name
                    blurred_filename_expected = original_basename_in_active_dir.rsplit(".", 1)[0] + "_blurred.jpg"
                    path_to_blurred_file = os.path.join(blurred_active_panos_dir, blurred_filename_expected)

                    if os.path.exists(path_to_blurred_file):
                        new_record = dict(record) 
                        new_record["filename"] = path_to_blurred_file
                        updated_records_for_blurred_files.append(new_record)
                    else:
                        print(f"Warning: Expected blurred file {path_to_blurred_file} not found. Original record: {record['filename']}")
                
                if not updated_records_for_blurred_files:
                    print("Error: Blurring enabled, but no blurred files could be mapped for metadata update. Cannot proceed.")
                    if CLEANUP_TEMP_FULL_SAMPLING_DIR and os.path.exists(full_sampled_images_dir_in_temp):
                        shutil.rmtree(full_sampled_images_dir_in_temp)
                    return
                
                current_mapillary_meta_for_facade_processing = os.path.join(blurred_active_panos_dir, "blurred_mapillary_description.json")
                with open(current_mapillary_meta_for_facade_processing, 'w', encoding='utf-8') as f:
                    json.dump(updated_records_for_blurred_files, f, indent=2)
                print(f"Metadata for blurred images saved to: {current_mapillary_meta_for_facade_processing}")
            else:
                print("Image blurring failed or produced no output directory. Proceeding with unblurred images.")
        else:
            print("\n--- Skipping Stage: Image Blurring ---")

        print("\n--- Running Stage: Façade Processing (Targeted) ---")
        gdf_target_building_only = gpd.GeoDataFrame(
            [gdf_all_buildings[gdf_all_buildings[id_column_name].astype(str).str.strip() == actual_target_bld_id].iloc[0]],
            crs=gdf_all_buildings.crs
        )
        temp_target_geojson_path = os.path.join(temp_processing_dir_base, "target_building_temp.geojson")
        try:
            gdf_target_building_only.to_file(temp_target_geojson_path, driver="GeoJSON")
        except Exception as e_to_file:
            print(f"Error writing temporary target GeoJSON: {e_to_file}")
            return

        target_facade_matches_csv = process_facade.process_building_footprints(
            mapillary_image_description_json_path=current_mapillary_meta_for_facade_processing,
            footprint_geojson_path=temp_target_geojson_path,
            base_output_dir=temp_processing_dir_base,
            max_distance_to_building_m=MAX_PANO_DISTANCE_TO_TARGET_BLD_M,
            frontal_view_tolerance_deg=FRONTAL_TOL_DEG_DEPLOY
        )
        if os.path.exists(temp_target_geojson_path):
            os.remove(temp_target_geojson_path)

        if not target_facade_matches_csv or not (os.path.exists(target_facade_matches_csv) and pd.read_csv(target_facade_matches_csv).shape[0] > 0):
            print(f"Façade processing for BLD_ID '{actual_target_bld_id}' produced no valid matches. Deployment cannot proceed.")
            if CLEANUP_TEMP_FULL_SAMPLING_DIR and os.path.exists(full_sampled_images_dir_in_temp):
                 shutil.rmtree(full_sampled_images_dir_in_temp)
            return
        print(f"Facade processing complete. Matches CSV: {target_facade_matches_csv}")

        print("\n--- Running Stage: Panorama Rotation (Targeted) ---")
        rotated_panos_dir_target, rotated_panos_meta_target = rotate.rotate_panoramas_to_facades(
            facade_matches_csv_path=target_facade_matches_csv,
            source_panoramas_dir=current_pano_source_dir_for_processing, # This dir contains images referenced by CSV (now possibly blurred)
            base_output_dir=temp_processing_dir_base,
            measured_camera_offset_deg=pano_zero_offset_for_rotation, # Use the determined offset
            distance_cutoff_m=ROTATION_DIST_CUTOFF_M_DEPLOY
        )
        if not rotated_panos_dir_target or not rotated_panos_meta_target:
            print(f"Panorama rotation failed for BLD_ID '{actual_target_bld_id}'.")
            if CLEANUP_TEMP_FULL_SAMPLING_DIR and os.path.exists(full_sampled_images_dir_in_temp):
                 shutil.rmtree(full_sampled_images_dir_in_temp)
            return

        print("\n--- Running Stage: Cube Face Extraction (Targeted) ---")
        if not CUBE_FACES_TO_EXTRACT_DEPLOY:
            faces_to_extract_final = config.ALL_POSSIBLE_CUBE_FACES
        else:
            faces_to_extract_final = CUBE_FACES_TO_EXTRACT_DEPLOY
        
        cube_faces_dir_target, cube_faces_meta_target = extract_cube.extract_cubemap_faces(
            rotated_panoramas_meta_json_path=rotated_panos_meta_target,
            base_output_dir=temp_processing_dir_base,
            faces_to_extract=faces_to_extract_final
        )
        if not cube_faces_dir_target or not cube_faces_meta_target:
            print(f"Cube face extraction failed for BLD_ID '{actual_target_bld_id}'.")
            if CLEANUP_TEMP_FULL_SAMPLING_DIR and os.path.exists(full_sampled_images_dir_in_temp):
                 shutil.rmtree(full_sampled_images_dir_in_temp)
            return

        print("\n--- Running Stage: Output Sorting (Targeted) ---")
        temp_final_sorted_base = os.path.join(temp_processing_dir_base, "final_sort_temp_base")
        ensure_dir_exists(temp_final_sorted_base)

        path_to_intermediate_sorted_dir = sort.sort_cube_faces_by_building(
            cube_faces_metadata_json_path=cube_faces_meta_target,
            base_output_dir=temp_final_sorted_base, 
            building_damage_csv_path=BUILDING_DAMAGE_CSV_PATH,
            move_files=False 
        )

        if path_to_intermediate_sorted_dir:
            source_dir_for_final_copy = os.path.join(path_to_intermediate_sorted_dir, safe_bld_id_dirname)
            
            if os.path.isdir(source_dir_for_final_copy):
                print(f"Copying final sorted outputs from '{source_dir_for_final_copy}' to '{final_building_output_dir}'")
                if os.path.exists(final_building_output_dir) and os.listdir(final_building_output_dir):
                     print(f"Warning: Final output directory '{final_building_output_dir}' is not empty. Contents might be overwritten or merged.")
                
                shutil.copytree(source_dir_for_final_copy, final_building_output_dir, dirs_exist_ok=True)
                print(f"Final sorted outputs for BLD_ID '{actual_target_bld_id}' are in: {final_building_output_dir}")
            else:
                print(f"Error: Expected sorted directory '{source_dir_for_final_copy}' not found after sorting operation.")
        else:
            print(f"Output sorting failed for BLD_ID '{actual_target_bld_id}'.")

        if CLEANUP_TEMP_FULL_SAMPLING_DIR:
            if full_sampled_images_dir_in_temp and os.path.exists(full_sampled_images_dir_in_temp):
                print(f"Cleaning up full sampling directory: {full_sampled_images_dir_in_temp}")
                try:
                    shutil.rmtree(full_sampled_images_dir_in_temp)
                except Exception as e_cleanup:
                    print(f"Warning: Could not cleanup {full_sampled_images_dir_in_temp}: {e_cleanup}")
        
    print(f"\nStandalone deployment script finished for BLD_ID '{actual_target_bld_id}'.")
    print(f"Final outputs should be in: {final_building_output_dir}")

if __name__ == "__main__":
    deploy_single_building_standalone()