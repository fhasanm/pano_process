# utils.py
import os
from math import radians, degrees, atan2, sin, cos, sqrt
import numpy as np
from pyproj import Geod
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Geodesic object for distance and bearing calculations
geod = Geod(ellps="WGS84")

def ensure_dir_exists(dir_path):
    """Creates a directory if it doesn't already exist."""
    os.makedirs(dir_path, exist_ok=True)

def great_circle_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates the initial bearing (forward azimuth) from point 1 to point 2
    using great-circle navigation.
    Returns bearing in degrees (0-360).
    """
    phi1, phi2 = map(radians, (lat1, lat2))
    lambda1, lambda2 = map(radians, (lon1, lon2))
    delta_lambda = lambda2 - lambda1

    y = sin(delta_lambda) * cos(phi2)
    x = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(delta_lambda)
    theta = atan2(y, x)
    return (degrees(theta) + 360) % 360

def angular_difference(a1, a2):
    """
    Calculates the shortest angle between two angles (in degrees).
    Result is in range [0, 180].
    """
    diff = abs(a1 - a2) % 360
    return diff if diff <= 180 else 360 - diff

def signed_angular_difference(a1, a2):
    """
    Calculates the signed shortest angle from a1 to a2 (in degrees).
    a2 - a1, result in range [-180, 180].
    Positive if a2 is clockwise from a1, negative if counter-clockwise.
    """
    diff = (a2 - a1 + 180) % 360 - 180
    return diff

def get_cartesian_coordinates_from_latlon(lat, lon, center_lat, center_lon):
    """
    Approximate cartesian coordinates (x, y) in meters from lat/lon
    relative to a center point. This is a simplified planar projection.
    """
    R = 6371000  # Earth radius in meters
    x = R * (lon - center_lon) * cos(radians(center_lat))
    y = R * (lat - center_lat)
    return x, y

def drop_z(pt):
    """Return tuple (x,y) from a point that might have a Z coordinate."""
    return pt[0], pt[1]

def find_front_edge(polygon: Polygon, cam_lat: float, cam_lon: float, cam_to_centroid_bearing: float):
    """
    Identifies the edge of a polygon whose midpoint faces the camera most directly.
    Returns the coordinates of the best edge ((lon1, lat1), (lon2, lat2)) or None.
    """
    best_edge_coords, min_angle_diff_to_centroid_ray = None, 360.0
    
    # Handle both Polygon and MultiPolygon
    rings = []
    if polygon.geom_type == "Polygon":
        rings.append(polygon.exterior)
        for interior in polygon.interiors:
            rings.append(interior)
    elif polygon.geom_type == "MultiPolygon":
        for poly_part in polygon.geoms:
            rings.append(poly_part.exterior)
            for interior in poly_part.interiors:
                rings.append(interior)
    else:
        return None # Not a Polygon or MultiPolygon

    for ring in rings:
        coords = list(ring.coords)
        for i in range(len(coords) - 1):
            p1_lon, p1_lat = drop_z(coords[i])
            p2_lon, p2_lat = drop_z(coords[i+1])
            
            mid_lon, mid_lat = (p1_lon + p2_lon) / 2, (p1_lat + p2_lat) / 2
            
            # Bearing from camera to this edge's midpoint
            bearing_cam_to_midpoint = great_circle_bearing(cam_lat, cam_lon, mid_lat, mid_lon)
            
            # How much does this deviate from the direct line to the building centroid?
            # This helps select an edge on the "facing side" of the building.
            angle_diff = angular_difference(bearing_cam_to_midpoint, cam_to_centroid_bearing)
            
            if angle_diff < min_angle_diff_to_centroid_ray:
                min_angle_diff_to_centroid_ray = angle_diff
                best_edge_coords = ((p1_lon, p1_lat), (p2_lon, p2_lat))
                
    return best_edge_coords

def calculate_edge_normal_bearing(lon1: float, lat1: float, lon2: float, lat2: float):
    """
    Calculates the outward normal bearing for an edge defined by (lon1, lat1) -> (lon2, lat2).
    The normal points to the "left" of the edge direction, assuming CCW polygon winding for exterior.
    Returns bearing in degrees (0-360).
    """
    edge_bearing = great_circle_bearing(lat1, lon1, lat2, lon2)
    # For a CCW exterior ring, the outward normal is edge_bearing - 90 degrees.
    # For a CW interior ring, this would be inward. Assuming exterior for facade.
    normal_bearing = (edge_bearing - 90 + 360) % 360
    return normal_bearing

def calculate_distance_meters(lat1, lon1, lat2, lon2):
    """Calculate geodesic distance in meters."""
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2) # inv returns az1, az2, dist
    return distance

def find_closest_building_by_latlon(target_lat: float, target_lon: float, gdf_buildings: gpd.GeoDataFrame, tolerance_m: float = 50.0):
    """
    Finds the closest building ID from a GeoDataFrame to a target lat/lon.

    Args:
        target_lat: Latitude of the target point.
        target_lon: Longitude of the target point.
        gdf_buildings: GeoDataFrame with building geometries and 'BLD_ID'.
                       Must be in EPSG:4326.
        tolerance_m: Maximum distance in meters to consider a building.

    Returns:
        Tuple (BLD_ID, centroid_lat, centroid_lon, distance_m) or (None, None, None, None) if no building is found within tolerance.
    """
    if gdf_buildings.crs.to_epsg() != 4326:
        gdf_buildings = gdf_buildings.to_crs(epsg=4326)

    min_dist = float('inf')
    closest_bld_id = None
    closest_bld_centroid_lat = None
    closest_bld_centroid_lon = None

    for idx, building in gdf_buildings.iterrows():
        centroid = building.geometry.centroid
        bld_lat, bld_lon = centroid.y, centroid.x
        dist = calculate_distance_meters(target_lat, target_lon, bld_lat, bld_lon)

        if dist < min_dist:
            min_dist = dist
            closest_bld_id = building.get("BLD_ID", None) # Ensure BLD_ID column exists
            if closest_bld_id is None and "id" in building: # Try "id" if "BLD_ID" not found
                 closest_bld_id = building["id"]
            closest_bld_centroid_lat = bld_lat
            closest_bld_centroid_lon = bld_lon


    if closest_bld_id is not None and min_dist <= tolerance_m:
        return str(closest_bld_id), closest_bld_centroid_lat, closest_bld_centroid_lon, min_dist
    else:
        print(f"No building found within {tolerance_m}m of {target_lat}, {target_lon}. Closest was {min_dist:.2f}m away.")
        return None, None, None, None