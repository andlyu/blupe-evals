from scripts.split_lerobot_dataset_by_episode import _feature_subset


def test_feature_subset_normalizes_json_shape_lists_to_tuples():
    info = {
        "features": {
            "observation.state": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [6]},
            "observation.images.camera1": {
                "dtype": "image",
                "shape": [480, 640, 3],
            },
        }
    }

    features = _feature_subset(info, ["observation.images.camera1"])

    assert features["observation.state"]["shape"] == (6,)
    assert features["action"]["shape"] == (6,)
    assert features["observation.images.camera1"]["shape"] == (480, 640, 3)
