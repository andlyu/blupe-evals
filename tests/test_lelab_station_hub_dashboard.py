from scripts.lelab_station_hub import DATASET_EDITOR_HTML, HUB_DASHBOARD_HTML, RECORD_DATASET_HTML


def test_dashboard_targets_station_hub_api_only():
    assert "LeLab Station Dashboard" in HUB_DASHBOARD_HTML
    assert "getJson('/api/stations')" in HUB_DASHBOARD_HTML
    assert "/api/stations/${encodeURIComponent(station.id)}/camera/" in HUB_DASHBOARD_HTML
    assert "/record/start" in HUB_DASHBOARD_HTML
    assert "/teleop/claim" in HUB_DASHBOARD_HTML
    assert "/dataset?station=" in HUB_DASHBOARD_HTML
    assert "192.168." not in HUB_DASHBOARD_HTML
    assert "8091" not in HUB_DASHBOARD_HTML


def test_dataset_editor_targets_station_hub_dataset_api_only():
    assert "LeLab Dataset Editor" in DATASET_EDITOR_HTML
    assert "/api/stations/${encodeURIComponent(station)}/recordings" in DATASET_EDITOR_HTML
    assert "/api/stations/${encodeURIComponent(station)}/recording?name=" in DATASET_EDITOR_HTML
    assert "/api/stations/${encodeURIComponent(selectedStation())}/segments/save" in DATASET_EDITOR_HTML
    assert "/api/stations/${encodeURIComponent(selectedStation())}/segments/export" in DATASET_EDITOR_HTML
    assert "/api/stations/${encodeURIComponent(station)}/frame?name=" in DATASET_EDITOR_HTML
    assert "/api/episodes" not in DATASET_EDITOR_HTML
    assert "/api/busyboard/extract" not in DATASET_EDITOR_HTML


def test_record_dataset_page_uses_station_presets_and_native_recording_fields():
    assert "LeLab Station Recording" in RECORD_DATASET_HTML
    assert "/api/stations/${encodeURIComponent(station)}/recording-preset" in RECORD_DATASET_HTML
    assert "Dataset repo" in RECORD_DATASET_HTML
    assert "Station robot" in RECORD_DATASET_HTML
    assert "robotProfileSelect" in RECORD_DATASET_HTML
    assert "robot_profile" in RECORD_DATASET_HTML
    assert "Episode time" in RECORD_DATASET_HTML
    assert "Reset time" in RECORD_DATASET_HTML
    assert "Add Camera" in RECORD_DATASET_HTML
    assert "camera_configs" in RECORD_DATASET_HTML
    assert "/api/stations/${encodeURIComponent(station)}/record/start" in RECORD_DATASET_HTML
