# map_builder.py
import pandas as pd
from config import ZONE_COORDS

def is_flow_visible(source: str, target: str, selected_countries: list, direction: str) -> bool:
    if not selected_countries:
        return True
    if "Both" in direction:
        return source in selected_countries or target in selected_countries
    if "Export" in direction:
        return source in selected_countries
    if "Import" in direction:
        return target in selected_countries
    return False

def calculate_perpendicular_vector(zone1: str, zone2: str) -> tuple:
    coords1, coords2 = ZONE_COORDS[zone1], ZONE_COORDS[zone2]
    dx = coords2[0] - coords1[0]
    dy = coords2[1] - coords1[1]
    distance = (dx**2 + dy**2)**0.5
    if distance == 0:
        return 0, 0
    return -dy / distance, dx / distance

def generate_map_nodes(selected_countries: list) -> list:
    node_data = []
    for zone, coords in ZONE_COORDS.items():
        is_active = not selected_countries or zone in selected_countries
        color = [0, 114, 178, 255] if is_active else [140, 140, 140, 100]
        radius = 8000 if is_active else 2000
        node_data.append({"coords": coords, "color": color, "zone": zone, "radius": radius})
    return node_data

def generate_flow_arcs(hourly_flows: pd.DataFrame, selected_countries: list, flow_direction: str) -> list:
    arc_data = []
    arc_pairs = {} 
    
    for _, row in hourly_flows.iterrows():
        source, target, flow_value_mw = row['src'], row['tgt'], float(row['value'])
        if not is_flow_visible(source, target, selected_countries, flow_direction):
            continue
            
        if source in ZONE_COORDS and target in ZONE_COORDS and abs(flow_value_mw) > 10:
            pair_key = tuple(sorted([source, target]))
            if pair_key not in arc_pairs:
                arc_pairs[pair_key] = []
            arc_pairs[pair_key].append((source, target, flow_value_mw))

    for pair_key, flows in arc_pairs.items():
        perp_x, perp_y = calculate_perpendicular_vector(pair_key[0], pair_key[1]) if len(flows) > 1 else (0, 0)
        
        for index, (source, target, flow_value) in enumerate(flows):
            source_coords = list(ZONE_COORDS[source])
            target_coords = list(ZONE_COORDS[target])
            
            if len(flows) > 1:
                offset_multiplier = 0.3 if index == 0 else -0.3
                source_coords[0] += perp_x * offset_multiplier
                source_coords[1] += perp_y * offset_multiplier
                target_coords[0] += perp_x * offset_multiplier
                target_coords[1] += perp_y * offset_multiplier
                
            arc_data.append({
                "source": source_coords, "target": target_coords,
                "value": flow_value, "label": f"{source} → {target}: {flow_value:.0f} MW"
            })
            
    return arc_data