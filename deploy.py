# deploy.py
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
4. (Optional) Blurs the selected panoramas.
5. Processes façades for the selected panoramas against the target building.
6. Rotates the selected panoramas to face the identified façades of the target building.
7. Extracts specified cube faces from these rotated panoramas.
8. Sorts the extracted cube faces into a dedicated output directory for the
   target building, including a `building_info.json` with metadata.

Prerequisites:
- Same dependencies as `main.py` (OpenCV, PyTorch, ultralytics, sam2, equilib,
  geopandas, pandas, etc.).
- `mapillary_tools` installed and accessible.
- Model files (YOLO, SAM) correctly placed and configured in `config.py`.

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

# Ensure pipeline_library is in path (adjust if necessary)
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PARENT_DIR = os.path.dirname(SCRIPT_DIR)
# if PARENT_DIR not in sys.path:
#    sys.path.append(PARENT_DIR)

import config
from utils import (
    ensure_dir_exists,
    find_closest_building_by_latlon,
    calculate_distance_meters
)
import sample, privacy_blur, process_facade, rotate, extract_cube, sort


def deploy_single_building_standalone():
    # ========================== USER CONFIGURATION ==========================
    # --- Essential Input Paths ---
    VIDEO_FILE_PATH = "/home/fuad/Downloads/VID_20250327_120940_20250507131706.mp4" # REQUIRED
    BUILDING_FOOTPRINTS_GEOJSON_PATH = "./2023_Buildings_with_DINS_data.geojson" # REQUIRED
    BUILDING_DAMAGE_CSV_PATH = "./2023_Buildings_with_DINS_data.csv" # For final sorting stage

    # --- Output Directory ---
    # All outputs for this specific building deployment will go here.
    # A subdirectory named after the BLD_ID will be created.
    DEPLOYMENT_BASE_OUTPUT_DIR = "./deployment_standalone_output"

    # --- Target Building Identification (Use EITHER BLD_ID OR Lat/Lon) ---
    TARGET_BLD_ID = ""  # e.g., "Building123". If set, Lat/Lon will be ignored.
    # OR
    TARGET_LATITUDE = 34.052200  # Example: Latitude of the desired building
    TARGET_LONGITUDE = -118.243700 # Example: Longitude of the desired building

    MAX_DIST_FOR_LATLON_MATCH_M = 70.0 # Max distance (meters) to link lat/lon to a building centroid.
    # Max distance from a panorama to the target building's centroid for it to be processed.
    MAX_PANO_DISTANCE_TO_TARGET_BLD_M = 100.0 

    # --- Crucial Camera Parameter ---
    # This MUST be pre-determined (e.g., from a previous interactive run or known specs).
    PREDETERMINED_PANO_ZERO_OFFSET = -67.0 # degrees (Example value)

    # --- Stage Execution Flags & Parameters ---
    # Blurring
    ENABLE_IMAGE_BLURRING = True
    YOLO_CONF_THRESHOLD_BLUR = 0.25
    MIN_BOX_SIZE_PX_BLUR = 15
    BLUR_KERNEL_SIZE = 31 # Must be odd

    # Video Sampling
    SAMPLING_DISTANCE_M = config.DEFAULT_SAMPLING_DISTANCE_M

    # Façade Processing
    MAX_FACADE_DIST_M_DEPLOY = config.DEFAULT_MAX_FACADE_DIST_M
    FRONTAL_TOL_DEG_DEPLOY = config.DEFAULT_FRONTAL_TOL_DEG

    # Panorama Rotation
    ROTATION_DIST_CUTOFF_M_DEPLOY = config.DEFAULT_ROTATION_DIST_CUTOFF_M
    
    # Cube Extraction
    CUBE_FACES_TO_EXTRACT_DEPLOY = ["front"] # e.g., ["front"], ["front", "left"], or [] for all.

    # Output Sorting
    MOVE_FILES_IN_FINAL_SORT = False

    # --- Advanced Configuration ---
    MAPILLARY_TOOLS_EXEC_PATH = config.MAPILLARY_TOOLS_PATH
    COMPUTATION_DEVICE = config.DEVICE
    # Model paths for blurring are taken from config.py
    
    CLEANUP_TEMP_FULL_SAMPLING_DIR = True # Delete the directory with all sampled video frames at the end
    # ======================= END OF USER CONFIGURATION =======================

    print("Starting STANDALONE deployment for a single building...")
    ensure_dir_exists(DEPLOYMENT_BASE_OUTPUT_DIR)

    # --- 1. Determine Target Building ID and Geometry ---
    actual_target_bld_id = None
    target_building_geometry = None
    target_building_centroid_wgs84 = None

    if not os.path.isfile(BUILDING_FOOTPRINTS_GEOJSON_PATH):
        print(f"Error: Building footprints GeoJSON not found at {BUILDING_FOOTPRINTS_GEOJSON_PATH}.")
        return

    try:
        gdf_all_buildings = gpd.read_file(BUILDING_FOOTPRINTS_GEOJSON_PATH)
        if gdf_all_buildings.crs.to_epsg() != 4326:
            gdf_all_buildings = gdf_all_buildings.to_crs(epsg=4326)
    except Exception as e:
        print(f"Error reading building GeoJSON: {e}")
        return

    if TARGET_BLD_ID:
        actual_target_bld_id = str(TARGET_BLD_ID).strip()
        # Find the geometry for this BLD_ID
        target_building_series = gdf_all_buildings[gdf_all_buildings['BLD_ID'].astype(str).str.strip() == actual_target_bld_id]
        if target_building_series.empty:
            print(f"Error: Target BLD_ID '{actual_target_bld_id}' not found in GeoJSON.")
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
            target_building_series = gdf_all_buildings[gdf_all_buildings['BLD_ID'].astype(str).str.strip() == actual_target_bld_id]
            target_building_geometry = target_building_series.iloc[0].geometry
            target_building_centroid_wgs84 = Point(bld_lon, bld_lat) # Use centroid from find_closest
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

    # Create a specific final output directory for this building
    safe_bld_id_dirname = actual_target_bld_id.replace(os.sep, "_").replace(" ", "_").replace(":", "_")
    final_building_output_dir = os.path.join(DEPLOYMENT_BASE_OUTPUT_DIR, safe_bld_id_dirname + "_sorted")
    ensure_dir_exists(final_building_output_dir) # This will be the root for this building's sorted output.

    # Use a temporary directory for all intermediate processing steps
    with tempfile.TemporaryDirectory(prefix="pano_deploy_temp_") as temp_processing_dir_base:
        print(f"Using temporary processing directory: {temp_processing_dir_base}")

        # --- 2. Video Sampling (Full video, then filter) ---
        print("\n--- Running Stage: Full Video Sampling ---")
        if not os.path.isfile(VIDEO_FILE_PATH):
            print(f"Error: Video file not found: {VIDEO_FILE_PATH}")
            return

        # sampled_images_output_dir_full will be inside temp_processing_dir_base
        # sample creates "01_sampled_images" inside the base_output_dir it's given
        full_sampled_images_dir, full_mapillary_meta_json_path = sample.sample_video_by_distance(
            video_path=VIDEO_FILE_PATH,
            base_output_dir=temp_processing_dir_base, # Sample into a subdir of temp
            distance_m=SAMPLING_DISTANCE_M,
            mapillary_tools_path=MAPILLARY_TOOLS_EXEC_PATH
        )
        if not full_sampled_images_dir or not full_mapillary_meta_json_path:
            print("Full video sampling failed. Terminating.")
            return
        
        # --- 3. Filter Panoramas near Target Building ---
        print("\n--- Running Stage: Filtering Panoramas near Target Building ---")
        try:
            with open(full_mapillary_meta_json_path, 'r') as f:
                all_pano_metadata = json.load(f)
        except Exception as e:
            print(f"Error reading full Mapillary metadata: {e}")
            return

        filtered_pano_records = []
        active_panos_dir = os.path.join(temp_processing_dir_base, "01a_active_panos_for_target")
        ensure_dir_exists(active_panos_dir)

        print(f"Filtering panoramas within {MAX_PANO_DISTANCE_TO_TARGET_BLD_M}m of target building centroid ({target_building_centroid_wgs84.y:.5f}, {target_building_centroid_wgs84.x:.5f})...")
        for pano_meta in tqdm(all_pano_metadata, desc="Filtering Panos"):
            try:
                pano_lat = float(pano_meta["MAPLatitude"])
                pano_lon = float(pano_meta["MAPLongitude"])
                dist_to_target = calculate_distance_meters(
                    pano_lat, pano_lon,
                    target_building_centroid_wgs84.y, target_building_centroid_wgs84.x
                )

                if dist_to_target <= MAX_PANO_DISTANCE_TO_TARGET_BLD_M:
                    original_pano_path = pano_meta["filename"] # This is often absolute from mapillary_tools
                    if not os.path.isfile(original_pano_path):
                         # If filename is relative in JSON (unlikely for mapillary_tools), try relative to its dir
                         original_pano_path = os.path.join(os.path.dirname(full_mapillary_meta_json_path), pano_meta["filename"])

                    if os.path.isfile(original_pano_path):
                        # Copy relevant pano to active_panos_dir
                        new_pano_basename = os.path.basename(original_pano_path)
                        new_pano_path = os.path.join(active_panos_dir, new_pano_basename)
                        shutil.copy2(original_pano_path, new_pano_path)
                        
                        # Update metadata record with new path
                        updated_meta_record = dict(pano_meta)
                        updated_meta_record["filename"] = new_pano_path # Crucial: update path
                        filtered_pano_records.append(updated_meta_record)
                    else:
                        print(f"Warning: Original panorama file not found: {original_pano_path}. Skipping.")

            except (KeyError, ValueError) as e_filter:
                print(f"Warning: Skipping a panorama record due to missing/invalid lat/lon: {e_filter}")
                continue
        
        if not filtered_pano_records:
            print(f"No panoramas found near BLD_ID '{actual_target_bld_id}'. Deployment cannot proceed.")
            if CLEANUP_TEMP_FULL_SAMPLING_DIR and os.path.exists(full_sampled_images_dir):
                 print(f"Cleaning up full sampling directory: {full_sampled_images_dir}")
                 shutil.rmtree(full_sampled_images_dir)
            return

        print(f"Found {len(filtered_pano_records)} panoramas relevant to the target building.")
        
        # Create a new metadata JSON for these filtered/copied panoramas
        filtered_mapillary_meta_json_path = os.path.join(active_panos_dir, "filtered_mapillary_description.json")
        with open(filtered_mapillary_meta_json_path, 'w') as f:
            json.dump(filtered_pano_records, f, indent=2)

        current_pano_source_dir_for_processing = active_panos_dir
        current_mapillary_meta_for_processing = filtered_mapillary_meta_json_path

        # --- 4. Image Blurring (Optional, on filtered panoramas) ---
        if ENABLE_IMAGE_BLURRING:
            print("\n--- Running Stage: Image Blurring (Targeted) ---")
            # privacy_blur creates "02_blurred_images" inside the base_output_dir it's given
            blurred_active_panos_dir, _ = privacy_blur.blur_equirectangular_images(
                source_dir=active_panos_dir, # Input is the active (filtered) panos
                base_output_dir=temp_processing_dir_base, # Output to a subdir of temp_processing_dir_base
                face_weights_path=config.FACE_WEIGHTS_PATH,
                plate_weights_path=config.PLATE_WEIGHTS_PATH,
                sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
                sam_hf_model_name=config.SAM_HF_MODEL_NAME,
                device=COMPUTATION_DEVICE,
                yolo_confidence_threshold=YOLO_CONF_THRESHOLD_BLUR,
                min_box_size_px=MIN_BOX_SIZE_PX_BLUR,
                blur_kernel_size=BLUR_KERNEL_SIZE
            )
            if blurred_active_panos_dir:
                current_pano_source_dir_for_processing = blurred_active_panos_dir
                print(f"Blurred active panoramas ready in: {current_pano_source_dir_for_processing}")
            else:
                print("Targeted image blurring failed. Subsequent stages will use unblurred active images.")
        else:
            print("\n--- Skipping Stage: Image Blurring ---")

        # --- 5. Façade Processing (Targeted) ---
        print("\n--- Running Stage: Façade Processing (Targeted) ---")
        # Create a temporary GeoDataFrame with only the target building
        gdf_target_building_only = gpd.GeoDataFrame([gdf_all_buildings[gdf_all_buildings['BLD_ID'].astype(str).str.strip() == actual_target_bld_id].iloc[0]], crs=gdf_all_buildings.crs)

        # process_facade creates "03_intermediate_data" in its base_output_dir
        target_facade_matches_csv = process_facade.process_building_footprints(
            mapillary_image_description_json_path=current_mapillary_meta_for_processing, # Use filtered metadata
            footprint_geojson_path=None, # Not used directly, pass gdf instead
            _footprint_gdf_override=gdf_target_building_only, # Pass the GDF directly (requires mod to process_facade or a wrapper)
                                                            # For now, let's assume process_facade can be adapted or we write a temp GeoJSON
            base_output_dir=temp_processing_dir_base,
            max_distance_to_building_m=MAX_FACADE_DIST_M_DEPLOY, # This is distance from pano to centroid
            frontal_view_tolerance_deg=FRONTAL_TOL_DEG_DEPLOY
        )
        # To avoid modifying process_facade, we can write a temp GeoJSON for the single building
        temp_target_geojson_path = os.path.join(temp_processing_dir_base, "target_building_temp.geojson")
        gdf_target_building_only.to_file(temp_target_geojson_path, driver="GeoJSON")

        target_facade_matches_csv = process_facade.process_building_footprints(
            mapillary_image_description_json_path=current_mapillary_meta_for_processing,
            footprint_geojson_path=temp_target_geojson_path, # Use temp GeoJSON of single building
            base_output_dir=temp_processing_dir_base, # Output to intermediate in temp_processing_dir_base
            max_distance_to_building_m=MAX_PANO_DISTANCE_TO_TARGET_BLD_M, # Max distance from pano to building
            frontal_view_tolerance_deg=FRONTAL_TOL_DEG_DEPLOY
        )
        os.remove(temp_target_geojson_path) # Clean up temp GeoJSON

        if not target_facade_matches_csv or not pd.read_csv(target_facade_matches_csv).shape[0] > 0:
            print(f"Façade processing for BLD_ID '{actual_target_bld_id}' produced no matches. Deployment cannot proceed.")
            if CLEANUP_TEMP_FULL_SAMPLING_DIR and os.path.exists(full_sampled_images_dir):
                 shutil.rmtree(full_sampled_images_dir)
            return

        # --- 6. Panorama Rotation (Targeted) ---
        print("\n--- Running Stage: Panorama Rotation (Targeted) ---")
        # rotate creates "04_rotated_panoramas"
        rotated_panos_dir_target, rotated_panos_meta_target = rotate.rotate_panoramas_to_facades(
            facade_matches_csv_path=target_facade_matches_csv,
            source_panoramas_dir=current_pano_source_dir_for_processing, # Use (possibly blurred) active panos
            base_output_dir=temp_processing_dir_base,
            measured_camera_offset_deg=PREDETERMINED_PANO_ZERO_OFFSET,
            distance_cutoff_m=ROTATION_DIST_CUTOFF_M_DEPLOY # Filters matches from CSV further
        )
        if not rotated_panos_dir_target or not rotated_panos_meta_target:
            print(f"Panorama rotation failed for BLD_ID '{actual_target_bld_id}'.")
            if CLEANUP_TEMP_FULL_SAMPLING_DIR and os.path.exists(full_sampled_images_dir):
                 shutil.rmtree(full_sampled_images_dir)
            return

        # --- 7. Cube Face Extraction (Targeted) ---
        print("\n--- Running Stage: Cube Face Extraction (Targeted) ---")
        if not CUBE_FACES_TO_EXTRACT_DEPLOY:
            faces_to_extract_final = config.ALL_POSSIBLE_CUBE_FACES
        else:
            faces_to_extract_final = CUBE_FACES_TO_EXTRACT_DEPLOY
        
        # extract_cube creates "05_cube_faces"
        cube_faces_dir_target, cube_faces_meta_target = extract_cube.extract_cubemap_faces(
            rotated_panoramas_meta_json_path=rotated_panos_meta_target,
            base_output_dir=temp_processing_dir_base,
            faces_to_extract=faces_to_extract_final
        )
        if not cube_faces_dir_target or not cube_faces_meta_target:
            print(f"Cube face extraction failed for BLD_ID '{actual_target_bld_id}'.")
            if CLEANUP_TEMP_FULL_SAMPLING_DIR and os.path.exists(full_sampled_images_dir):
                 shutil.rmtree(full_sampled_images_dir)
            return

        # --- 8. Output Sorting (Targeted, into final user-specified directory) ---
        print("\n--- Running Stage: Output Sorting (Targeted) ---")
        # The sort will create its "06_sorted_outputs_by_building" structure
        # INSIDE the `final_building_output_dir` if we pass `final_building_output_dir` as base.
        # However, sort is designed to make one subdir per BLD_ID.
        # For a single building deployment, we want the contents *directly* in `final_building_output_dir/{safe_bld_id_dirname}`
        # or just `final_building_output_dir` if it's already named appropriately.
        # Let's adjust the call or expect sort to place it correctly.
        # The current sort will create final_building_output_dir/06_sorted_outputs_by_building/BLD_ID_DIR/...
        # To simplify, we will call sort to sort into a temporary "final_sort_location" within temp_processing_dir_base,
        # then copy the *contents* of the specific BLD_ID folder from there to `final_building_output_dir`.

        temp_final_sorted_base = os.path.join(temp_processing_dir_base, "final_sort_temp_base")
        ensure_dir_exists(temp_final_sorted_base)

        sorted_results_path_in_temp = sort.sort_cube_faces_by_building(
            cube_faces_metadata_json_path=cube_faces_meta_target,
            base_output_dir=temp_final_sorted_base, # Sort into a temp base
            building_damage_csv_path=BUILDING_DAMAGE_CSV_PATH,
            move_files=False # Always copy from temp, then main temp dir is deleted
        )

        if sorted_results_path_in_temp:
            # The actual sorted files for this building are in:
            # temp_final_sorted_base / "06_sorted_outputs_by_building" / safe_bld_id_dirname
            source_dir_for_final_copy = os.path.join(temp_final_sorted_base, "06_sorted_outputs_by_building", safe_bld_id_dirname)
            
            if os.path.isdir(source_dir_for_final_copy):
                print(f"Copying final sorted outputs from {source_dir_for_final_copy} to {final_building_output_dir}")
                # Ensure final_building_output_dir is empty or handle appropriately
                if os.path.exists(final_building_output_dir) and os.listdir(final_building_output_dir):
                     print(f"Warning: Final output directory {final_building_output_dir} is not empty. Overwriting may occur.")
                # else: # This was already created earlier
                #    ensure_dir_exists(final_building_output_dir)

                for item in os.listdir(source_dir_for_final_copy):
                    s_item = os.path.join(source_dir_for_final_copy, item)
                    d_item = os.path.join(final_building_output_dir, item)
                    if os.path.isdir(s_item):
                        shutil.copytree(s_item, d_item, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s_item, d_item)
                print(f"Final sorted outputs for BLD_ID '{actual_target_bld_id}' are in: {final_building_output_dir}")
            else:
                print(f"Error: Expected sorted directory {source_dir_for_final_copy} not found after sorting.")
        else:
            print(f"Output sorting failed for BLD_ID '{actual_target_bld_id}'.")

        # --- 9. Cleanup ---
        if CLEANUP_TEMP_FULL_SAMPLING_DIR:
            if full_sampled_images_dir and os.path.exists(full_sampled_images_dir):
                print(f"Cleaning up full sampling directory: {full_sampled_images_dir}")
                try:
                    shutil.rmtree(full_sampled_images_dir)
                except Exception as e_cleanup:
                    print(f"Warning: Could not cleanup {full_sampled_images_dir}: {e_cleanup}")
        
        # The rest of temp_processing_dir_base (active_panos_dir, blurred, intermediate, etc.)
        # will be cleaned up automatically when the `with tempfile.TemporaryDirectory()` block exits.
        print(f"Temporary processing directory {temp_processing_dir_base} will be removed.")

    print(f"\nStandalone deployment script finished for BLD_ID '{actual_target_bld_id}'.")
    print(f"Final outputs should be in: {final_building_output_dir}")


if __name__ == "__main__":
    # Modify the process_facade call slightly or ensure it handles GDF override if we want to avoid temp geojson
    # For now, the temp GeoJSON approach is used.
    # A small modification to process_facade to accept a GeoDataFrame directly would be cleaner:
    # def process_building_footprints(..., footprint_geojson_path=None, _footprint_gdf_override=None, ...):
    # if _footprint_gdf_override is not None:
    #     gdf_buildings_wgs84 = _footprint_gdf_override
    # elif footprint_geojson_path:
    #     gdf_buildings_wgs84 = gpd.read_file(footprint_geojson_path) ...
    # else:
    #     # error
    # This change is not made here to keep existing scripts unmodified as per initial request style.
    deploy_single_building_standalone()