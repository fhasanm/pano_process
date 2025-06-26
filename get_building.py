#!/usr/bin/env python3
"""
get_building.py – Targeted Image Extractor for a Specific Building
-------------------------------------------------------------------
This script generates specific image outputs (rotated panoramas and/or
cube faces) for a single target building. It can start from either a
video file or a directory of pre-sampled panoramas.

Workflow:
1.  Input Source:
    - If the input is a VIDEO: The script will sample the entire video into
      a temporary directory to get panoramic frames and their metadata.
      This temporary directory is automatically deleted upon completion.
    - If the input is a DIRECTORY of sampled photos: The script uses the
      existing images and their corresponding 'mapillary_image_description.json'.

2.  Target Identification:
    - The script identifies the target building using a Building ID (BLD_ID)
      or by finding the closest building in the GeoJSON to a given
      latitude and longitude.

3.  Filtering & Processing:
    - It filters the panoramas to select only those near the target building.
    - If the camera offset is not pre-configured, it runs an interactive
      measurement using one of the relevant, filtered panoramas.
    - It then runs the necessary processing stages (façade matching, rotation,
      and cube extraction) on this small subset of images.

4.  Output Generation (Configurable):
    - Based on boolean flags, the script will generate specified outputs
      (rotated panoramas, cube faces, or both) with metadata.

How to Use:
1.  Configure the "USER CONFIGURATION" section below.
2.  Set `KNOWN_PANO_ZERO_OFFSET_DEG` to a number if you know the offset, or set it
    to `None` to trigger the interactive measurement window.
3.  Run the script: `python path/to/get_building.py`
"""
import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from tqdm import tqdm

# Make sure the library modules can be found
try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:
    print("Error: geopandas or shapely not found. Please run 'pip install geopandas shapely'.")
    sys.exit(1)

# Import library modules
import config
from utils import ensure_dir_exists, find_closest_building_by_latlon, calculate_distance_meters
import sample
import offset 
import process_facade
import rotate
import extract_cube
import sort

def get_building_images():
    # ========================== USER CONFIGURATION ==========================
    # --- 1. Input Source ---
    INPUT_SOURCE_PATH = Path(r"D:\Fuad\pano_process\output\VID_20250327_120940_20250507131706\01_sampled_images")
    INPUT_IS_VIDEO = False

    # --- 2. Target Building Specification (Use ONE of these) ---
    TARGET_BLD_ID = None
    TARGET_LATITUDE = 34.194344300490954
    TARGET_LONGITUDE = -118.1477737109565
    MAX_DIST_FOR_LATLON_MATCH_M = 75.0

    # --- 3. Desired Outputs (Set at least one to True) ---
    GET_ROTATED_PANOS = True
    GET_CUBE_FACES = True
    
    # --- 4. Processing Parameters ---
    # Set to a number (e.g., 180.0) if the offset is known.
    # Set to `None` if you want the script to run the interactive measurement.
    KNOWN_PANO_ZERO_OFFSET_DEG = None

    MAX_PANO_DISTANCE_TO_TARGET_BLD_M = 100.0
    CUBE_FACES_TO_EXTRACT = ["front"]

    # --- 5. Data & Output Paths ---
    BUILDING_FOOTPRINTS_GEOJSON_PATH = Path("./data/2023_Buildings_with_DINS_data.geojson")
    BUILDING_ATTRIBUTES_CSV_PATH = Path("./data/2023_Buildings_with_DINS_data.csv")
    FINAL_OUTPUT_DIR = Path("./building_specific_output")
    # ======================= END OF USER CONFIGURATION =======================

    print("Getting Building Images...")
    ensure_dir_exists(FINAL_OUTPUT_DIR)

    # --- Step 1: Validate Inputs and Identify Target Building ---
    if not BUILDING_FOOTPRINTS_GEOJSON_PATH.is_file():
        print(f"Error: Building footprints GeoJSON not found at '{BUILDING_FOOTPRINTS_GEOJSON_PATH}'.")
        return

    try:
        gdf_all_buildings = gpd.read_file(BUILDING_FOOTPRINTS_GEOJSON_PATH).to_crs(epsg=4326)
        id_col = 'BLD_ID' if 'BLD_ID' in gdf_all_buildings.columns else 'id'
    except Exception as e:
        print(f"Error reading GeoJSON: {e}"); return

    actual_target_bld_id, target_building_centroid = None, None
    if TARGET_BLD_ID:
        target_series = gdf_all_buildings[gdf_all_buildings[id_col].astype(str).strip() == str(TARGET_BLD_ID).strip()]
        if not target_series.empty:
            actual_target_bld_id = str(TARGET_BLD_ID).strip()
            target_building_centroid = target_series.iloc[0].geometry.centroid
    elif TARGET_LATITUDE and TARGET_LONGITUDE:
        bld_id_match, bld_lat, bld_lon, _ = find_closest_building_by_latlon(TARGET_LATITUDE, TARGET_LONGITUDE, gdf_all_buildings, MAX_DIST_FOR_LATLON_MATCH_M)
        if bld_id_match:
            actual_target_bld_id = bld_id_match
            target_building_centroid = Point(bld_lon, bld_lat)
    
    if not actual_target_bld_id:
        print("Error: Could not identify target building. Exiting."); return
    
    print(f"Identified Target Building ID: {actual_target_bld_id}")
    safe_bld_id_dirname = actual_target_bld_id.replace(os.sep, "_").replace(" ", "_")
    building_final_output_dir = FINAL_OUTPUT_DIR / safe_bld_id_dirname
    ensure_dir_exists(building_final_output_dir)

    # --- Step 2: Prepare Input Data in a Temporary Directory ---
    with tempfile.TemporaryDirectory(prefix="getbuilding_") as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Using temporary processing directory: {temp_path}")

        full_sampled_images_dir, full_mapillary_meta_json = None, None
        if INPUT_IS_VIDEO:
            if not INPUT_SOURCE_PATH.is_file(): print(f"Error: Input video not found at '{INPUT_SOURCE_PATH}'."); return
            print(f"Sampling video '{INPUT_SOURCE_PATH.name}'...")
            full_sampled_images_dir, full_mapillary_meta_json = sample.sample_video_by_distance(
                video_path=str(INPUT_SOURCE_PATH), base_output_dir=str(temp_path),
                distance_m=config.DEFAULT_SAMPLING_DISTANCE_M, mapillary_tools_path=config.MAPILLARY_TOOLS_PATH
            )
            if not full_sampled_images_dir: print("Video sampling failed."); return
        else:
            if not INPUT_SOURCE_PATH.is_dir(): print(f"Error: Input directory not found at '{INPUT_SOURCE_PATH}'."); return
            print(f"Using existing photos from '{INPUT_SOURCE_PATH.name}'.")
            full_sampled_images_dir = str(INPUT_SOURCE_PATH)
            full_mapillary_meta_json = str(INPUT_SOURCE_PATH / "mapillary_image_description.json")

        if not Path(full_mapillary_meta_json).is_file():
            print(f"Error: Metadata JSON not found at '{full_mapillary_meta_json}'."); return

        # --- Step 3: Filter panoramas to create a relevant subset for processing ---
        print(f"Filtering panoramas within {MAX_PANO_DISTANCE_TO_TARGET_BLD_M}m of target building...")
        with open(full_mapillary_meta_json, 'r') as f: all_pano_metadata = json.load(f)
        
        active_panos_dir = temp_path / "active_panos"
        ensure_dir_exists(active_panos_dir)
        filtered_pano_records = []

        for pano_meta in tqdm(all_pano_metadata, desc="Filtering Panos"):
            try:
                pano_lat, pano_lon = float(pano_meta["MAPLatitude"]), float(pano_meta["MAPLongitude"])
                dist_to_target = calculate_distance_meters(pano_lat, pano_lon, target_building_centroid.y, target_building_centroid.x)
                if dist_to_target <= MAX_PANO_DISTANCE_TO_TARGET_BLD_M:
                    source_pano_path = Path(full_sampled_images_dir) / Path(pano_meta["filename"]).name
                    if source_pano_path.is_file():
                        shutil.copy2(source_pano_path, active_panos_dir)
                        # Update record to point to the new location in the temp active dir
                        pano_meta["filename"] = str(active_panos_dir / source_pano_path.name)
                        filtered_pano_records.append(pano_meta)
            except (KeyError, ValueError, TypeError): continue
        
        if not filtered_pano_records:
            print(f"No panoramas found viewing the target building '{actual_target_bld_id}'. Exiting."); return
        
        filtered_mapillary_meta_json = active_panos_dir / "filtered_mapillary_description.json"
        with open(filtered_mapillary_meta_json, 'w') as f: json.dump(filtered_pano_records, f, indent=2)
        print(f"Found {len(filtered_pano_records)} relevant panoramas.")

        # --- Step 4: Determine Camera Offset ---
        pano_zero_offset_for_rotation = KNOWN_PANO_ZERO_OFFSET_DEG
        if pano_zero_offset_for_rotation is None:
            print("\n--- Pano zero offset not known. Running interactive measurement... ---")
            measured_offset = offset.measure_yaw_offset_interactively(
                panoramas_image_dir=str(active_panos_dir),
                mapillary_image_description_json_path=str(filtered_mapillary_meta_json)
            )
            if measured_offset is not None:
                pano_zero_offset_for_rotation = measured_offset
            else:
                print("Warning: Interactive offset measurement failed. Falling back to default (180).")
                pano_zero_offset_for_rotation = 180.0
        print(f"Using camera offset for rotation: {pano_zero_offset_for_rotation:.2f}°")

        # --- Step 5: Process Façades and Rotate ---
        target_gdf = gdf_all_buildings[gdf_all_buildings[id_col] == actual_target_bld_id]
        temp_target_geojson = temp_path / "target_building.geojson"
        target_gdf.to_file(temp_target_geojson, driver="GeoJSON")

        facade_matches_csv = process_facade.process_building_footprints(
            mapillary_image_description_json_path=str(filtered_mapillary_meta_json),
            footprint_geojson_path=str(temp_target_geojson),
            base_output_dir=str(temp_path),
            max_distance_to_building_m=MAX_PANO_DISTANCE_TO_TARGET_BLD_M,
            frontal_view_tolerance_deg=config.DEFAULT_FRONTAL_TOL_DEG
        )
        if not facade_matches_csv or not Path(facade_matches_csv).exists(): print("Façade processing failed."); return

        rotation_stage_output_dir = temp_path / "rotation_stage"
        _, rotated_panos_meta_json = rotate.process_all_rotations(
             general_output_dir=str(rotation_stage_output_dir),
             original_panos_input_dir=str(active_panos_dir), # Use the filtered "active" panos
             mapillary_meta_json_path=str(filtered_mapillary_meta_json),
             footprint_geojson_path=str(temp_target_geojson),
             max_facade_distance_m=MAX_PANO_DISTANCE_TO_TARGET_BLD_M,
             frontal_tolerance_deg=config.DEFAULT_FRONTAL_TOL_DEG,
             rotation_distance_cutoff_m=config.DEFAULT_ROTATION_DIST_CUTOFF_M,
             user_measured_pano_offset_deg=pano_zero_offset_for_rotation,
             target_bld_id_for_facades=actual_target_bld_id
        )
        if not rotated_panos_meta_json: print("Panorama rotation failed."); return

        # --- Step 6: Generate Final Outputs ---
        if GET_ROTATED_PANOS:
            print("Generating rotated panorama outputs...")
            final_rotated_panos_dir = building_final_output_dir / "rotated_panoramas"
            ensure_dir_exists(final_rotated_panos_dir)
            shutil.copy2(rotated_panos_meta_json, final_rotated_panos_dir)
            source_rotated_images_dir = Path(rotated_panos_meta_json).parent / "rotated_image_files"
            if source_rotated_images_dir.is_dir():
                shutil.copytree(source_rotated_images_dir, final_rotated_panos_dir, dirs_exist_ok=True)
                print(f"Saved rotated panoramas to: {final_rotated_panos_dir}")
        
        if GET_CUBE_FACES:
            print("Generating cube face outputs...")
            extraction_stage_dir = temp_path / "extraction_stage"
            source_rotated_images_dir = Path(rotated_panos_meta_json).parent / "rotated_image_files"
            faces_to_get = CUBE_FACES_TO_EXTRACT if CUBE_FACES_TO_EXTRACT else config.ALL_POSSIBLE_CUBE_FACES
            _, cube_faces_meta_json = extract_cube.extract_cubemap_faces(
                rotated_panoramas_meta_json_path=rotated_panos_meta_json,
                base_output_dir=str(extraction_stage_dir),
                faces_to_extract=faces_to_get
            )
            if not cube_faces_meta_json: print("Cube face extraction failed."); return

            final_cube_faces_dir = building_final_output_dir / "sorted_cube_faces"
            sort.sort_cube_faces_by_building(
                cube_faces_metadata_json_path=cube_faces_meta_json,
                building_damage_csv_path=str(BUILDING_ATTRIBUTES_CSV_PATH),
                final_sorted_dir=str(final_cube_faces_dir),
                move_files=False
            )
            print(f"Saved sorted cube faces to: {final_cube_faces_dir}")
            
    print(f"\nScript finished. Final outputs for building '{actual_target_bld_id}' are in: {building_final_output_dir}")

if __name__ == "__main__":
        get_building_images()
