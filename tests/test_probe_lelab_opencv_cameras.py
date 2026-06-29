from scripts.probe_lelab_opencv_cameras import parse_args


def test_probe_opencv_cameras_defaults_to_three_cameras() -> None:
    args = parse_args([])

    assert args.camera == [0, 1, 2]


def test_probe_opencv_cameras_explicit_cameras_replace_defaults() -> None:
    args = parse_args(["--camera", "0", "--camera", "2"])

    assert args.camera == [0, 2]
