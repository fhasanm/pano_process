#!/usr/bin/env python3
"""
offset.py – Interactively measure yaw offset.
Allows user to click on a panorama to determine the camera's optical axis world heading
relative to the vehicle's TrueHeading by clicking a point on the vehicle (typically rear).
"""
import os
import json
import random
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def _get_user_click_for_offset(image_path: Path, true_heading: float):
    """
    Displays a panorama, prompts user to click a reference point on the vehicle
    (typically directly backward), and calculates the PANO_ZERO_OFFSET.

    Args:
        image_path: Path to the equirectangular panorama image.
        true_heading: The TrueHeading of the vehicle (forward direction) when the panorama was captured.

    Returns:
        The calculated PANO_ZERO_OFFSET in degrees. This is the value to add
        to the vehicle's TrueHeading to get the world heading of the panorama's
        center (0-degree yaw line).
        Normalized to [-180, 180].
    """
    try:
        img = plt.imread(str(image_path))
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

    h, w = img.shape[:2]
    degrees_per_pixel = 360.0 / w

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(img)
    ax.axvline(w / 2, linestyle="--", linewidth=1, color="white", alpha=0.7, label="Pano Center (0° Yaw)")
    
    # Instruction updated to reflect clicking a backward point.
    instruction_text = (
        f"Vehicle TrueHeading (FORWARD): {true_heading:.2f}°\n"
        f"Image: {image_path.name}\n\n"
        "1. Identify a point directly BACKWARD on the vehicle (e.g., rear antenna, center of rear bumper).\n"
        "2. Click on that point in the image.\n"
        "3. Close this window to continue."
    )
    fig.text(0.5, 0.98, instruction_text, ha="center", va="top", fontsize=10, color="black",
             bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8))
    
    ax.set_title("Measure Camera Yaw Offset by Clicking Vehicle's REAR Point")
    plt.legend()
    
    try:
        if "DISPLAY" not in os.environ and not hasattr(sys, 'ps1'):
             print("Warning: Matplotlib is likely running in a headless environment. Interactive plot may not display.")
             print("Cannot measure offset interactively. Please run in an environment with a display server.")
             plt.close(fig)
             return 0.0 # Default or error indication

        click_points = plt.ginput(1, timeout=-1) # timeout=-1 waits indefinitely
    except Exception as e:
        print(f"Could not get GUI input, possibly due to environment: {e}. Returning default offset 0.0.")
        plt.close(fig)
        return 0.0 # Default offset if ginput fails

    plt.close(fig)

    if not click_points:
        print("No point selected. Cannot calculate offset.")
        return None
        
    x_click, _ = click_points[0]

    # --- CORRECTED OFFSET CALCULATION ---
    # dx_pixels: Pixel difference of the clicked (REAR) point from the panorama center.
    # Positive if click is to the right of center.
    dx_pixels = x_click - (w / 2)

    # dx_degrees_pano_coord: Angular position of the clicked (REAR) point in the panorama's
    # coordinate system (0 degrees is pano center). Positive if to the right.
    dx_degrees_pano_coord = dx_pixels * degrees_per_pixel

    # vehicle_rear_world_heading: The actual world heading of the point the user clicked
    # (the vehicle's rear). true_heading is the vehicle's FORWARD direction.
    vehicle_rear_world_heading = (true_heading + 180.0) % 360.0

    # camera_optical_axis_world_heading: The world heading of the panorama's center line (0-degree yaw).
    # If the vehicle's REAR (at vehicle_rear_world_heading) appears at dx_degrees_pano_coord
    # in the panorama's frame, then the panorama's center world heading is:
    # vehicle_rear_world_heading - dx_degrees_pano_coord
    camera_optical_axis_world_heading = (vehicle_rear_world_heading - dx_degrees_pano_coord + 360.0) % 360.0

    # PANO_ZERO_OFFSET: The value to add to the vehicle's FORWARD TrueHeading to get the
    # world heading of the panorama's center.
    # Offset = Camera Optical Axis World Heading - Vehicle's FORWARD TrueHeading
    # Normalized to the range [-180, 180]
    offset = ((camera_optical_axis_world_heading - true_heading + 180.0) % 360.0) - 180.0
    
    print(f"\n--- Offset Calculation (Clicked REAR point) ---")
    print(f"Image: {image_path.name}")
    print(f"Image width: {w}px, Degrees per pixel: {degrees_per_pixel:.3f}")
    print(f"Clicked X-coordinate (REAR reference): {x_click:.2f}px")
    print(f"Center X-coordinate of Panorama: {w/2:.2f}px")
    print(f"Pixel difference of click from center (dx_pixels): {dx_pixels:.2f}px")
    print(f"Angular position of clicked REAR point in pano (dx_degrees_pano_coord): {dx_degrees_pano_coord:.2f}°")
    print(f"Vehicle TrueHeading (FORWARD): {true_heading:.2f}°")
    print(f"Calculated Vehicle REAR World Heading: {vehicle_rear_world_heading:.2f}°")
    print(f"Calculated Camera Optical Axis World Heading (Pano Center): {camera_optical_axis_world_heading:.2f}°")
    print(f"Calculated PANO_ZERO_OFFSET: {offset:+.2f}°")
    print(f"(This means the panorama's center is oriented {offset:+.2f}° from the vehicle's FORWARD heading)")
    print("--------------------------------------------------\n")
    
    return offset

def measure_yaw_offset_interactively(
    panoramas_image_dir: str, 
    mapillary_image_description_json_path: str
) -> float:
    """
    Allows the user to interactively measure the camera yaw offset.
    Selects a random panorama, displays it, and prompts the user to click
    a reference point on the vehicle (typically directly backward).

    Args:
        panoramas_image_dir: Base directory containing the sampled equirectangular panoramas (JPGs).
                             Mapillary tools often creates subdirectories here named after the video.
        mapillary_image_description_json_path: Path to the JSON file from mapillary_tools
                                               (e.g., 'mapillary_image_description.json')
                                               which includes TrueHeading for each image.

    Returns:
        The measured yaw offset (PANO_ZERO_OFFSET) in degrees, or 0.0 if measurement fails or is skipped.
    """
    if not os.path.isdir(panoramas_image_dir):
        print(f"Error: Panoramas base directory not found: {panoramas_image_dir}")
        return 0.0
    if not os.path.isfile(mapillary_image_description_json_path):
        print(f"Error: Mapillary image description JSON not found: {mapillary_image_description_json_path}")
        return 0.0

    try:
        with open(mapillary_image_description_json_path, 'r') as f:
            pano_metadata_list = json.load(f)
    except Exception as e:
        print(f"Error reading metadata JSON {mapillary_image_description_json_path}: {e}")
        return 0.0

    if not pano_metadata_list:
        print(f"No panorama metadata found in JSON file: {mapillary_image_description_json_path}")
        return 0.0

    heading_map = {} 

    # print(f"Processing {len(pano_metadata_list)} records from {mapillary_image_description_json_path}...") # Optional: for verbosity
    for p_meta in pano_metadata_list:
        try:
            json_filename_str = p_meta.get('filename')
            if not json_filename_str:
                continue

            path_from_json = Path(json_filename_str)

            if path_from_json.exists() and path_from_json.is_file():
                true_heading_value = None
                if 'MAPCompassHeading' in p_meta and \
                   isinstance(p_meta['MAPCompassHeading'], dict) and \
                   'TrueHeading' in p_meta['MAPCompassHeading']:
                    true_heading_value = p_meta['MAPCompassHeading']['TrueHeading']
                elif 'CompassHeading' in p_meta and \
                     isinstance(p_meta['CompassHeading'], dict) and \
                     'TrueHeading' in p_meta['CompassHeading']: 
                    true_heading_value = p_meta['CompassHeading']['TrueHeading']

                if true_heading_value is not None:
                    try:
                        heading_map[str(path_from_json)] = float(true_heading_value)
                    except (ValueError, TypeError):
                        pass
        except KeyError:
            pass 
        except Exception: # Catch any other unexpected errors with a record
            pass


    if not heading_map:
        print(f"No image files with valid TrueHeading found based on the paths in '{mapillary_image_description_json_path}'.")
        print("Possible reasons:")
        print("1. The 'filename' entries in the JSON do not point to existing image files.")
        print("2. The existing image files listed in the JSON do not have a valid 'MAPCompassHeading.TrueHeading' field.")
        print("Please verify the content of the JSON and the existence/permissions of the image files.")
        return 0.0

    valid_image_paths_with_heading = list(heading_map.keys())
    
    chosen_image_path_str = random.choice(valid_image_paths_with_heading)
    chosen_image_path_obj = Path(chosen_image_path_str) 
    true_heading_for_chosen_image = heading_map[chosen_image_path_str]

    print(f"\nStarting interactive yaw offset measurement...")
    # print(f"Found {len(valid_image_paths_with_heading)} images with heading data to choose from.") # Optional: for verbosity
    print(f"Using randomly selected panorama: {chosen_image_path_obj.name}")
    # print(f"Full path: {chosen_image_path_str}") # Optional: for verbosity
    print(f"Vehicle TrueHeading (FORWARD) for this image: {true_heading_for_chosen_image:.2f}°")

    measured_offset = _get_user_click_for_offset(chosen_image_path_obj, true_heading_for_chosen_image)

    if measured_offset is None:
        print("Offset measurement failed or was cancelled by the user.")
        return 0.0 

    print(f"Interactive measurement complete. Measured PANO_ZERO_OFFSET: {measured_offset:+.2f}°")
    return measured_offset