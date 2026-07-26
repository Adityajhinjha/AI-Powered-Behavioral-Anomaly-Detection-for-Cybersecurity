"""
Synthetic Access Log Data Generator.
Generates realistic access-log telemetry with 7 injected attack patterns
for training and evaluating the anomaly detection system.
"""
import os
import sys
import uuid
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    NUM_USERS, NUM_SERVICE_ACCOUNTS, NUM_EDGE_DEVICES,
    TOTAL_EVENTS, ANOMALY_RATE, DATA_WINDOW_DAYS,
    ATTACK_DISTRIBUTION, RAW_DATA_PATH,
)
from src.database import init_db, save_logs

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# ─── Geographic Locations (Major World Cities) ────────────────────────────────
CITY_COORDS = [
    ("New York", 40.7128, -74.0060),
    ("London", 51.5074, -0.1278),
    ("Tokyo", 35.6762, 139.6503),
    ("Mumbai", 19.0760, 72.8777),
    ("Sydney", -33.8688, 151.2093),
    ("Berlin", 52.5200, 13.4050),
    ("São Paulo", -23.5505, -46.6333),
    ("Toronto", 43.6532, -79.3832),
    ("Singapore", 1.3521, 103.8198),
    ("Dubai", 25.2048, 55.2708),
    ("San Francisco", 37.7749, -122.4194),
    ("Chicago", 41.8781, -87.6298),
    ("Paris", 48.8566, 2.3522),
    ("Seoul", 37.5665, 126.9780),
    ("Bangalore", 12.9716, 77.5946),
]

# ─── Resources / Endpoints ───────────────────────────────────────────────────
RESOURCES = [
    "/api/users", "/api/reports", "/api/settings", "/api/billing",
    "/api/dashboard", "/api/analytics", "/api/admin/config",
    "/api/admin/users", "/api/admin/logs", "/api/admin/security",
    "/files/documents", "/files/shared", "/files/confidential",
    "/files/exports", "/files/backups",
    "/db/production", "/db/staging", "/db/analytics",
    "/server/monitoring", "/server/deploy", "/server/ssh",
    "/network/firewall", "/network/vpn", "/network/dns",
    "/iot/sensor-data", "/iot/firmware-update", "/iot/control-panel",
]

AUTH_METHODS = ["password", "token", "certificate", "biometric"]
OS_VERSIONS = ["Windows 11", "Windows 10", "macOS 14", "macOS 13", "Ubuntu 22.04", "CentOS 8"]
PROTOCOLS = ["TLS 1.3", "TLS 1.2", "SSH", "HTTPS"]

COMMANDS_NORMAL = [
    ["login", "view_dashboard", "logout"],
    ["login", "read_file", "download", "logout"],
    ["login", "view_reports", "export_csv", "logout"],
    ["login", "check_email", "reply", "logout"],
    ["login", "update_profile", "logout"],
]

COMMANDS_SUSPICIOUS = [
    ["login", "enumerate_users", "access_admin", "download_db", "clear_logs"],
    ["login", "scan_network", "access_firewall", "modify_rules", "exfiltrate"],
    ["login", "escalate_privilege", "access_confidential", "bulk_download"],
    ["login", "install_backdoor", "create_admin_user", "modify_logs"],
]


def _create_entity_profiles():
    """Create realistic entity profiles with behavioral baselines."""
    entities = []

    # Regular users
    for i in range(NUM_USERS):
        city_name, lat, lon = random.choice(CITY_COORDS)
        typical_hour = random.gauss(10, 2)  # Most users work around 10 AM
        typical_hour = max(6, min(20, typical_hour))

        entities.append({
            "entity_id": f"user_{fake.user_name()}_{i:04d}",
            "entity_type": "user",
            "home_city": city_name,
            "home_lat": lat + random.uniform(-0.05, 0.05),
            "home_lon": lon + random.uniform(-0.05, 0.05),
            "typical_hour": typical_hour,
            "hour_std": random.uniform(1.0, 3.0),
            "typical_resources": random.sample(RESOURCES[:15], k=random.randint(3, 8)),
            "auth_method": random.choice(["password", "token", "biometric"]),
            "os": random.choice(OS_VERSIONS),
            "mac": fake.mac_address(),
            "protocol": random.choice(PROTOCOLS),
            "avg_session_duration": random.uniform(300, 7200),  # 5 min - 2 hours
            "first_seen_offset": random.randint(0, 60),  # Days before window start
        })

    # Service accounts
    for i in range(NUM_SERVICE_ACCOUNTS):
        city_name, lat, lon = random.choice(CITY_COORDS[:5])
        entities.append({
            "entity_id": f"svc_{fake.word()}_{i:03d}",
            "entity_type": "service_account",
            "home_city": city_name,
            "home_lat": lat,
            "home_lon": lon,
            "typical_hour": 12,  # 24/7 access
            "hour_std": 12,      # Uniform distribution
            "typical_resources": random.sample(RESOURCES[15:], k=random.randint(2, 5)),
            "auth_method": "certificate",
            "os": random.choice(["Ubuntu 22.04", "CentOS 8"]),
            "mac": fake.mac_address(),
            "protocol": "TLS 1.3",
            "avg_session_duration": random.uniform(60, 600),
            "first_seen_offset": random.randint(30, 90),
        })

    # Edge devices
    for i in range(NUM_EDGE_DEVICES):
        city_name, lat, lon = random.choice(CITY_COORDS)
        entities.append({
            "entity_id": f"device_{fake.lexify('???')}{i:03d}",
            "entity_type": "edge_device",
            "home_city": city_name,
            "home_lat": lat,
            "home_lon": lon,
            "typical_hour": 12,
            "hour_std": 12,
            "typical_resources": random.sample(RESOURCES[21:], k=random.randint(1, 3)),
            "auth_method": "certificate",
            "os": f"FirmwareOS {random.randint(1, 5)}.{random.randint(0, 9)}",
            "mac": fake.mac_address(),
            "protocol": random.choice(["TLS 1.2", "TLS 1.3"]),
            "avg_session_duration": random.uniform(30, 300),
            "first_seen_offset": random.randint(0, 45),
        })

    return entities


def _generate_normal_event(entity, base_time, day_offset):
    """Generate a single normal (benign) access event for an entity."""
    # Time: sampled from entity's habitual hours
    hour = max(0, min(23, int(random.gauss(entity["typical_hour"], entity["hour_std"]))))
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    ts = base_time + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)

    # Location: near home with small noise
    lat = entity["home_lat"] + random.uniform(-0.01, 0.01)
    lon = entity["home_lon"] + random.uniform(-0.01, 0.01)

    # Session duration: log-normal around entity's average
    duration = max(10, random.lognormvariate(
        np.log(entity["avg_session_duration"]), 0.3
    ))

    return {
        "log_id": str(uuid.uuid4()),
        "entity_id": entity["entity_id"],
        "entity_type": entity["entity_type"],
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "source_ip": fake.ipv4_private(),
        "geo_location": f"{lat:.6f},{lon:.6f}",
        "resource_accessed": random.choice(entity["typical_resources"]),
        "auth_method": entity["auth_method"],
        "auth_success": 1,
        "session_duration": round(duration, 1),
        "command_sequence": json.dumps(random.choice(COMMANDS_NORMAL)),
        "device_fingerprint": f"{entity['os']}|{entity['mac']}|{entity['protocol']}",
        "label": "normal",
    }


def _inject_brute_force(entity, base_time, day_offset):
    """Inject brute force attack: rapid failed auth attempts from one source."""
    events = []
    start_hour = random.randint(0, 23)
    start_ts = base_time + timedelta(days=day_offset, hours=start_hour)
    source_ip = fake.ipv4_public()
    num_attempts = random.randint(15, 50)

    for i in range(num_attempts):
        ts = start_ts + timedelta(seconds=random.randint(1, 300))  # Within 5 min window
        lat = entity["home_lat"] + random.uniform(-1, 1)
        lon = entity["home_lon"] + random.uniform(-1, 1)

        events.append({
            "log_id": str(uuid.uuid4()),
            "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"],
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": source_ip,
            "geo_location": f"{lat:.6f},{lon:.6f}",
            "resource_accessed": "/api/admin/security",
            "auth_method": "password",
            "auth_success": 0 if i < num_attempts - 1 else random.choice([0, 1]),
            "session_duration": round(random.uniform(0.5, 3.0), 1),
            "command_sequence": json.dumps(["login_attempt"]),
            "device_fingerprint": f"{random.choice(OS_VERSIONS)}|{fake.mac_address()}|HTTPS",
            "label": "brute_force",
        })

    return events


def _inject_impossible_travel(entity, base_time, day_offset):
    """Inject impossible travel: same entity logging in from distant locations within short time."""
    events = []
    city1_name, lat1, lon1 = random.choice(CITY_COORDS)
    # Choose a distant city (different from city1)
    distant_cities = [(n, la, lo) for n, la, lo in CITY_COORDS
                      if abs(la - lat1) > 20 or abs(lo - lon1) > 20]
    if not distant_cities:
        distant_cities = CITY_COORDS
    city2_name, lat2, lon2 = random.choice(distant_cities)

    hour = random.randint(8, 18)
    ts1 = base_time + timedelta(days=day_offset, hours=hour)
    ts2 = ts1 + timedelta(minutes=random.randint(10, 45))  # Within 45 min

    for ts, lat, lon, city in [(ts1, lat1, lon1, city1_name), (ts2, lat2, lon2, city2_name)]:
        events.append({
            "log_id": str(uuid.uuid4()),
            "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"],
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": fake.ipv4_public(),
            "geo_location": f"{lat:.6f},{lon:.6f}",
            "resource_accessed": random.choice(entity["typical_resources"]),
            "auth_method": entity["auth_method"],
            "auth_success": 1,
            "session_duration": round(random.uniform(300, 3600), 1),
            "command_sequence": json.dumps(random.choice(COMMANDS_NORMAL)),
            "device_fingerprint": f"{entity['os']}|{entity['mac']}|{entity['protocol']}",
            "label": "impossible_travel",
        })

    return events


def _inject_credential_stuffing(entities, base_time, day_offset):
    """Inject credential stuffing: many entity_ids from few source IPs with high failure."""
    events = []
    source_ips = [fake.ipv4_public() for _ in range(random.randint(1, 3))]
    target_entities = random.sample(entities, k=min(30, len(entities)))
    hour = random.randint(0, 23)

    for entity in target_entities:
        ts = base_time + timedelta(
            days=day_offset, hours=hour,
            minutes=random.randint(0, 30),
            seconds=random.randint(0, 59)
        )
        events.append({
            "log_id": str(uuid.uuid4()),
            "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"],
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": random.choice(source_ips),
            "geo_location": f"{random.uniform(-90, 90):.6f},{random.uniform(-180, 180):.6f}",
            "resource_accessed": "/api/users",
            "auth_method": "password",
            "auth_success": 1 if random.random() < 0.15 else 0,  # 85% fail rate
            "session_duration": round(random.uniform(0.5, 5.0), 1),
            "command_sequence": json.dumps(["login_attempt"]),
            "device_fingerprint": f"{random.choice(OS_VERSIONS)}|{fake.mac_address()}|HTTPS",
            "label": "credential_stuffing",
        })

    return events


def _inject_lateral_movement(entity, base_time, day_offset):
    """Inject lateral movement: entity accesses unusual breadth of resources."""
    events = []
    # Access many resources NOT in the entity's typical set
    unusual_resources = [r for r in RESOURCES if r not in entity["typical_resources"]]
    targets = random.sample(unusual_resources, k=min(15, len(unusual_resources)))
    hour = random.randint(10, 16)

    for i, resource in enumerate(targets):
        ts = base_time + timedelta(
            days=day_offset, hours=hour,
            minutes=i * random.randint(2, 10)
        )
        events.append({
            "log_id": str(uuid.uuid4()),
            "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"],
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": fake.ipv4_private(),
            "geo_location": f"{entity['home_lat']:.6f},{entity['home_lon']:.6f}",
            "resource_accessed": resource,
            "auth_method": entity["auth_method"],
            "auth_success": 1,
            "session_duration": round(random.uniform(30, 600), 1),
            "command_sequence": json.dumps(random.choice(COMMANDS_SUSPICIOUS)),
            "device_fingerprint": f"{entity['os']}|{entity['mac']}|{entity['protocol']}",
            "label": "lateral_movement",
        })

    return events


def _inject_device_spoofing(entity, base_time, day_offset):
    """Inject device spoofing: same device_id with mismatched fingerprint."""
    ts = base_time + timedelta(
        days=day_offset,
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    # Use a different OS and MAC than the entity's normal profile
    spoofed_os = random.choice([o for o in OS_VERSIONS if o != entity["os"]])
    spoofed_mac = fake.mac_address()

    return [{
        "log_id": str(uuid.uuid4()),
        "entity_id": entity["entity_id"],
        "entity_type": entity["entity_type"],
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "source_ip": fake.ipv4_public(),
        "geo_location": f"{entity['home_lat'] + random.uniform(-2, 2):.6f},"
                        f"{entity['home_lon'] + random.uniform(-2, 2):.6f}",
        "resource_accessed": random.choice(entity["typical_resources"]),
        "auth_method": entity["auth_method"],
        "auth_success": 1,
        "session_duration": round(random.uniform(300, 7200), 1),
        "command_sequence": json.dumps(random.choice(COMMANDS_NORMAL)),
        "device_fingerprint": f"{spoofed_os}|{spoofed_mac}|{entity['protocol']}",
        "label": "device_spoofing",
    }]


def _inject_low_and_slow(entity, base_time, day_offset):
    """Inject low-and-slow exfiltration: gradual off-hours access over weeks."""
    events = []
    num_days = random.randint(10, 25)

    for d in range(num_days):
        if random.random() > 0.6:  # Not every day — intermittent
            continue
        # Off-hours: 2-5 AM
        hour = random.randint(2, 5)
        ts = base_time + timedelta(days=day_offset + d, hours=hour, minutes=random.randint(0, 59))

        # Access slightly more resources each time
        unusual_resources = [r for r in RESOURCES if r not in entity["typical_resources"]]
        resource = random.choice(unusual_resources[:5 + d])  # Gradually broadening

        events.append({
            "log_id": str(uuid.uuid4()),
            "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"],
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": fake.ipv4_private(),
            "geo_location": f"{entity['home_lat']:.6f},{entity['home_lon']:.6f}",
            "resource_accessed": resource,
            "auth_method": entity["auth_method"],
            "auth_success": 1,
            "session_duration": round(random.uniform(60, 900), 1),
            "command_sequence": json.dumps(["login", "browse", "download_small", "logout"]),
            "device_fingerprint": f"{entity['os']}|{entity['mac']}|{entity['protocol']}",
            "label": "low_and_slow_exfiltration",
        })

    return events if events else [_generate_normal_event(entity, base_time, day_offset)]


def _inject_insider_drift(entity, base_time, day_offset):
    """Inject insider drift: legitimate entity slowly expanding resource footprint."""
    events = []
    num_days = random.randint(20, 30)
    unusual_resources = [r for r in RESOURCES if r not in entity["typical_resources"]]

    for d in range(num_days):
        if random.random() > 0.5:
            continue
        hour = int(random.gauss(entity["typical_hour"], entity["hour_std"]))
        hour = max(0, min(23, hour))
        ts = base_time + timedelta(days=day_offset + d, hours=hour, minutes=random.randint(0, 59))

        # Gradually include more unusual resources (drift)
        max_idx = min(len(unusual_resources), 2 + d // 5)
        resource = random.choice(unusual_resources[:max_idx]) if max_idx > 0 else random.choice(entity["typical_resources"])

        events.append({
            "log_id": str(uuid.uuid4()),
            "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"],
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": fake.ipv4_private(),
            "geo_location": f"{entity['home_lat'] + random.uniform(-0.01, 0.01):.6f},"
                            f"{entity['home_lon'] + random.uniform(-0.01, 0.01):.6f}",
            "resource_accessed": resource,
            "auth_method": entity["auth_method"],
            "auth_success": 1,
            "session_duration": round(random.uniform(300, 5400), 1),
            "command_sequence": json.dumps(random.choice(COMMANDS_NORMAL)),
            "device_fingerprint": f"{entity['os']}|{entity['mac']}|{entity['protocol']}",
            "label": "insider_drift",
        })

    return events if events else [_generate_normal_event(entity, base_time, day_offset)]


def generate_all():
    """
    Main entry point: generate the full synthetic dataset.
    Creates normal baseline events and injects all 7 attack patterns.
    Saves results to CSV (data/raw/) and SQLite (database/).
    """
    print("=" * 60)
    print("  SYNTHETIC DATA GENERATOR")
    print("=" * 60)

    # Step 1: Create entity profiles
    print("\n[1/4] Creating entity profiles...")
    entities = _create_entity_profiles()
    user_entities = [e for e in entities if e["entity_type"] == "user"]
    print(f"  -> {len(entities)} entities ({NUM_USERS} users, "
          f"{NUM_SERVICE_ACCOUNTS} service accounts, {NUM_EDGE_DEVICES} edge devices)")

    # Step 2: Generate normal events
    print("\n[2/4] Generating normal baseline events...")
    base_time = datetime(2025, 1, 1)
    num_normal = int(TOTAL_EVENTS * (1 - ANOMALY_RATE))

    normal_events = []
    for _ in range(num_normal):
        entity = random.choice(entities)
        day_offset = random.randint(0, DATA_WINDOW_DAYS - 1)
        normal_events.append(_generate_normal_event(entity, base_time, day_offset))

    print(f"  -> {len(normal_events)} normal events generated")

    # Step 3: Inject attacks
    print("\n[3/4] Injecting attack patterns...")
    num_anomaly = TOTAL_EVENTS - num_normal
    attack_events = []

    for attack_type, fraction in ATTACK_DISTRIBUTION.items():
        target_count = int(num_anomaly * fraction)
        print(f"  -> Injecting {attack_type}: ~{target_count} events")

        generated = 0
        while generated < target_count:
            entity = random.choice(user_entities)
            day_offset = random.randint(0, DATA_WINDOW_DAYS - 1)

            if attack_type == "brute_force":
                events = _inject_brute_force(entity, base_time, day_offset)
            elif attack_type == "impossible_travel":
                events = _inject_impossible_travel(entity, base_time, day_offset)
            elif attack_type == "credential_stuffing":
                events = _inject_credential_stuffing(user_entities, base_time, day_offset)
            elif attack_type == "lateral_movement":
                events = _inject_lateral_movement(entity, base_time, day_offset)
            elif attack_type == "device_spoofing":
                events = _inject_device_spoofing(entity, base_time, day_offset)
            elif attack_type == "low_and_slow_exfiltration":
                events = _inject_low_and_slow(entity, base_time, day_offset)
            elif attack_type == "insider_drift":
                events = _inject_insider_drift(entity, base_time, day_offset)
            else:
                continue

            # Only take what we need
            remaining = target_count - generated
            attack_events.extend(events[:remaining])
            generated += len(events[:remaining])

    print(f"  -> {len(attack_events)} total attack events injected")

    # Step 4: Combine, shuffle, and save
    print("\n[4/4] Combining and saving dataset...")
    all_events = normal_events + attack_events
    random.shuffle(all_events)
    df = pd.DataFrame(all_events)

    # Sort by timestamp for realistic chronological order
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Save to CSV
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"  -> CSV saved: {RAW_DATA_PATH}")

    # Save to SQLite
    init_db()
    save_logs(df)

    # Print summary
    print("\n" + "=" * 60)
    print("  GENERATION COMPLETE")
    print("=" * 60)
    print(f"\n  Total events:  {len(df)}")
    print(f"  Normal events: {len(df[df['label'] == 'normal'])} "
          f"({len(df[df['label'] == 'normal']) / len(df) * 100:.1f}%)")
    print(f"  Attack events: {len(df[df['label'] != 'normal'])} "
          f"({len(df[df['label'] != 'normal']) / len(df) * 100:.1f}%)")
    print("\n  Attack type distribution:")
    for label, count in df[df["label"] != "normal"]["label"].value_counts().items():
        print(f"    {label:30s} {count:5d} ({count / len(df) * 100:.2f}%)")

    return df


if __name__ == "__main__":
    generate_all()
