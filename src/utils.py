import math
from datetime import datetime, timedelta
from geopy.distance import geodesic


def calculate_geo_velocity(lat1, lon1, lat2, lon2, time_diff_seconds):
    if time_diff_seconds <= 0:
        return 0.0

    try:
        distance_km = geodesic((lat1, lon1), (lat2, lon2)).kilometers
        speed_kmh = distance_km / (time_diff_seconds / 3600)
        return round(speed_kmh, 2)
    except Exception:
        return 0.0


def parse_timestamp(ts):
    if isinstance(ts, datetime):
        return ts
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts}")


def is_off_hours(hour, start=22, end=6):
    if start > end:
        return hour >= start or hour < end
    return start <= hour < end


def format_risk_badge(score):
    if score >= 0.8:
        color = "#ff4444"
        label = "CRITICAL"
    elif score >= 0.6:
        color = "#ff8800"
        label = "HIGH"
    elif score >= 0.4:
        color = "#ffcc00"
        label = "MEDIUM"
    else:
        color = "#44cc44"
        label = "LOW"

    return (
        f'<span style="background-color:{color}; color:white; '
        f'padding:2px 8px; border-radius:4px; font-weight:bold; '
        f'font-size:0.85em;">{label} ({score:.2f})</span>'
    )


def generate_random_geo(center_lat, center_lon, radius_km=50):
    import random
    radius_deg = radius_km / 111.0
    lat = center_lat + random.uniform(-radius_deg, radius_deg)
    lon = center_lon + random.uniform(-radius_deg, radius_deg)
    return round(lat, 6), round(lon, 6)


def format_shap_explanation(feature_names, shap_values, top_n=5):
    pairs = sorted(
        zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True
    )[:top_n]

    parts = [f"{name} ({value:+.3f})" for name, value in pairs if abs(value) > 0.001]

    if not parts:
        return "No significant contributing features"

    return "Flagged due to " + " + ".join(parts)
