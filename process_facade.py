
#!/usr/bin/env python3
"""
process_facade.py – Identifies building façades near panoramas.
Reads panorama metadata and a GeoJSON of building footprints.
For each panorama, finds nearby buildings and determines which building edge
is most "frontal" to the camera.
Outputs a CSV file detailing these panorama-building matches, including the
desired yaw for the panorama to face the façade directly.
"""
import os
import json
import csv
import warnings
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
from utils import (
    ensure_dir_exists,
    great_circle_bearing,
    angular_difference,
    find_front_edge,
    calculate_edge_normal_bearing
)

def process_building_footprints(
    mapillary_image_description_json_path: str,
    footprint_geojson_path: str,
    base_output_dir: str,
    max_distance_to_building_m: float,
    frontal_view_tolerance_deg: float
):
    """
    Enumerates frontal façades for each panorama.

    Args:
        mapillary_image_description_json_path: Path to the JSON file from mapillary_tools
                                               (e.g., 'mapillary_image_description.json').
        footprint_geojson_path: Path to the GeoJSON file containing building footprints.
                                Must include 'BLD_ID' and other properties like 'HEIGHT', 'ELEV', etc.
        base_output_dir: Base directory for pipeline outputs. The CSV will be saved here.
        max_distance_to_building_m: Maximum distance (meters) from camera to building centroid
                                    to consider a building.
        frontal_view_tolerance_deg: Maximum angle (degrees) between the camera-to-façade-midpoint
                                    ray and the outward normal of the façade edge.

    Returns:
        Path to the output CSV file containing panorama-façade matches, or None if an error occurs.
    """
    output_csv_dir = os.path.join(base_output_dir, "03_intermediate_data")
    ensure_dir_exists(output_csv_dir)
    output_csv_path = os.path.join(output_csv_dir, "pano_building_facade_matches.csv")

    if not os.path.exists(mapillary_image_description_json_path):
        print(f"Error: Mapillary metadata JSON not found at {mapillary_image_description_json_path}")
        return None
    if not os.path.exists(footprint_geojson_path):
        print(f"Error: Building footprint GeoJSON not found at {footprint_geojson_path}")
        return None

    try:
        gdf_buildings_wgs84 = gpd.read_file(footprint_geojson_path)
        if gdf_buildings_wgs84.crs.to_epsg() != 4326:
             gdf_buildings_wgs84 = gdf_buildings_wgs84.to_crs(epsg=4326)
        
        # For distance calculations, project to a suitable planar CRS (e.g., Web Mercator or local UTM)
        # Using Web Mercator (EPSG:3857) as a common default for global data.
        # A local UTM zone would be more accurate for distance if data is localized.
        gdf_buildings_metric = gdf_buildings_wgs84.to_crs(epsg=3857)
    except Exception as e:
        print(f"Error reading or reprojecting GeoJSON {footprint_geojson_path}: {e}")
        return None

    try:
        with open(mapillary_image_description_json_path, 'r') as f:
            panoramas_metadata = json.load(f)
    except Exception as e:
        print(f"Error reading panorama metadata {mapillary_image_description_json_path}: {e}")
        return None

    if not panoramas_metadata:
        print("No panorama metadata loaded. Cannot process façades.")
        return None

    output_rows = []
    header = [
        "pano_filename", "pano_abs_path", "pano_latitude", "pano_longitude", "pano_true_heading",
        "BLD_ID", "bld_height", "bld_elevation", "bld_source_dataset", "bld_capture_date", "bld_status",
        "distance_to_centroid_m", "desired_camera_yaw_to_facade",
        "bld_centroid_lon", "bld_centroid_lat",
        "facade_edge_lon1", "facade_edge_lat1", "facade_edge_lon2", "facade_edge_lat2"
    ]

    print(f"Processing {len(panoramas_metadata)} panoramas to find facade matches...")
    with warnings.catch_warnings(): # Suppress UserWarning about geographic CRS from geopandas distance
        warnings.filterwarnings("ignore", category=UserWarning, message="Geometry is in a geographic CRS.*")

        for pano_meta in tqdm(panoramas_metadata, desc="Processing Facades"):
            try:
                pano_lat = float(pano_meta["MAPLatitude"])
                pano_lon = float(pano_meta["MAPLongitude"])
                pano_true_heading = float(pano_meta.get("MAPCompassHeading", {}).get("TrueHeading", 0.0))
                # mapillary_image_description.json 'filename' is often absolute
                pano_abs_path = pano_meta["filename"] 
                pano_filename = os.path.basename(pano_abs_path)


                # Create a GeoSeries for the camera point in metric CRS for distance calculation
                camera_point_wgs84 = Point(pano_lon, pano_lat)
                camera_point_metric = gpd.GeoSeries([camera_point_wgs84], crs="EPSG:4326").to_crs(epsg=3857)[0]

                # Calculate distances from camera to all building centroids (in metric)
                # Using .geometry access to avoid warning if gdf_buildings_metric has no active geometry column set by name
                distances_to_centroids = gdf_buildings_metric.geometry.centroid.distance(camera_point_metric)
                
                # Filter buildings that are within max_distance_to_building_m
                # gdf_buildings_metric['dist_to_cam'] = distances_to_centroids
                # nearby_buildings_metric = gdf_buildings_metric[gdf_buildings_metric['dist_to_cam'] <= max_distance_to_building_m]
                
                # Efficiently get indices of nearby buildings
                nearby_indices = distances_to_centroids[distances_to_centroids <= max_distance_to_building_m].index


                for bld_idx in nearby_indices:
                    building_metric = gdf_buildings_metric.loc[bld_idx]
                    building_wgs84 = gdf_buildings_wgs84.loc[bld_idx] # Original WGS84 for lat/lon geometry
                    
                    building_polygon_wgs84 = building_wgs84.geometry
                    if not isinstance(building_polygon_wgs84, gpd.array.GeometryDtype.type): # Check if it's a shapely geometry
                         if hasattr(building_polygon_wgs84, 'iloc') and len(building_polygon_wgs84) > 0:
                            building_polygon_wgs84 = building_polygon_wgs84.iloc[0] # if it's a GeoSeries
                         elif not hasattr(building_polygon_wgs84, 'geom_type'): # if not a shapely geometry at all
                            print(f"Warning: Building {bld_idx} has unexpected geometry type {type(building_polygon_wgs84)}. Skipping.")
                            continue


                    bld_centroid_wgs84 = building_polygon_wgs84.centroid
                    bld_centroid_lon, bld_centroid_lat = bld_centroid_wgs84.x, bld_centroid_wgs84.y
                    
                    # Bearing from camera to building centroid
                    bearing_cam_to_bld_centroid = great_circle_bearing(pano_lat, pano_lon, bld_centroid_lat, bld_centroid_lon)

                    # Find the "front" edge of this building relative to the camera
                    facade_edge_coords = find_front_edge(building_polygon_wgs84, pano_lat, pano_lon, bearing_cam_to_bld_centroid)

                    if not facade_edge_coords:
                        continue # No suitable edge found

                    (edge_lon1, edge_lat1), (edge_lon2, edge_lat2) = facade_edge_coords
                    
                    # Calculate midpoint of the identified facade edge
                    facade_mid_lon = (edge_lon1 + edge_lon2) / 2
                    facade_mid_lat = (edge_lat1 + edge_lat2) / 2
                    
                    # Bearing from camera to the facade midpoint (this is the desired yaw)
                    desired_camera_yaw = great_circle_bearing(pano_lat, pano_lon, facade_mid_lat, facade_mid_lon)
                    
                    # Calculate the outward normal of this facade edge
                    facade_normal_bearing = calculate_edge_normal_bearing(edge_lon1, edge_lat1, edge_lon2, edge_lat2)
                    
                    # Check if the facade is "frontal" enough
                    # Angle between camera-to-midpoint ray and facade's outward normal
                    angle_diff_ray_normal = angular_difference(desired_camera_yaw, facade_normal_bearing)

                    if angle_diff_ray_normal <= frontal_view_tolerance_deg:
                        # This facade is considered frontal enough, record it
                        row = {
                            "pano_filename": pano_filename,
                            "pano_abs_path": pano_abs_path,
                            "pano_latitude": pano_lat,
                            "pano_longitude": pano_lon,
                            "pano_true_heading": pano_true_heading,
                            "BLD_ID": str(building_wgs84.get("BLD_ID", building_wgs84.get("id", f"Unknown_{bld_idx}"))), # Handle missing BLD_ID
                            "bld_height": building_wgs84.get("HEIGHT", ""),
                            "bld_elevation": building_wgs84.get("ELEV", ""),
                            "bld_source_dataset": building_wgs84.get("SOURCE", ""),
                            "bld_capture_date": building_wgs84.get("DATE_", ""), # Original script used DATE_
                            "bld_status": building_wgs84.get("STATUS", ""),
                            "distance_to_centroid_m": distances_to_centroids[bld_idx], # Use pre-calculated metric distance
                            "desired_camera_yaw_to_facade": desired_camera_yaw,
                            "bld_centroid_lon": bld_centroid_lon,
                            "bld_centroid_lat": bld_centroid_lat,
                            "facade_edge_lon1": edge_lon1,
                            "facade_edge_lat1": edge_lat1,
                            "facade_edge_lon2": edge_lon2,
                            "facade_edge_lat2": edge_lat2,
                        }
                        output_rows.append(row)
            except Exception as e_pano:
                print(f"Error processing panorama {pano_meta.get('filename', 'Unknown')}: {e_pano}")
                # import traceback
                # traceback.print_exc() # For more detailed debugging if needed

    if not output_rows:
        print("No facade matches found for any panorama.")
    else:
        try:
            with open(output_csv_path, "w", newline="", encoding="utf-8") as fp_csv:
                writer = csv.DictWriter(fp_csv, fieldnames=header)
                writer.writeheader()
                writer.writerows(output_rows)
            print(f"✅ Facade processing complete. Matches saved to → {output_csv_path}")
        except Exception as e_csv:
            print(f"Error writing CSV output to {output_csv_path}: {e_csv}")
            return None
            
    return output_csv_path