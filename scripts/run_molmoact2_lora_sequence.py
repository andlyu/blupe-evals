#!/usr/bin/env python3
"""Run the four MolmoAct2 SO-101 LoRA variants sequentially on one worker."""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    dataset_repo: str
    image_key: str


DEFAULT_VARIANTS = (
    Variant("so101-teleop-front-lora", "local/so101-molmoact2-teleop-front", "observation.images.front"),
    Variant("so101-teleop-side-lora", "local/so101-molmoact2-teleop-side", "observation.images.side"),
    Variant(
        "so101-teleop-intervention-front-lora",
        "local/so101-molmoact2-teleop-intervention-front",
        "observation.images.front",
    ),
    Variant(
        "so101-teleop-intervention-side-lora",
        "local/so101-molmoact2-teleop-intervention-side",
        "observation.images.side",
    ),
)


def _selected_variants(names: list[str]) -> list[Variant]:
    if not names:
        return list(DEFAULT_VARIANTS)
    by_name = {variant.name: variant for variant in DEFAULT_VARIANTS}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(f"unknown variants: {missing}; known variants: {sorted(by_name)}")
    return [by_name[name] for name in names]


def _run_variant(command: str, variant: Variant, dry_run: bool) -> None:
    env = os.environ.copy()
    env.update(
        {
            "RUN_NAME": variant.name,
            "DATASET_REPO": variant.dataset_repo,
            "IMAGE_KEY": variant.image_key,
        }
    )
    print(f"[molmoact2] RUN_NAME={variant.name}", flush=True)
    print(f"[molmoact2] DATASET_REPO={variant.dataset_repo}", flush=True)
    print(f"[molmoact2] IMAGE_KEY={variant.image_key}", flush=True)
    print(f"[molmoact2] command={command}", flush=True)
    if dry_run:
        return
    subprocess.run(command, shell=True, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        default=(
            "python train_molmoact2_lora.py "
            '--dataset "$DATASET_REPO" '
            '--image-key "$IMAGE_KEY" '
            '--run-name "$RUN_NAME" '
            "--lora-r 16 "
            "--lora-alpha 32 "
            "--bf16"
        ),
        help="Training command. Receives RUN_NAME, DATASET_REPO, and IMAGE_KEY.",
    )
    parser.add_argument("--variant", action="append", default=[], help="Run only this variant. Repeatable.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for variant in _selected_variants(args.variant):
        _run_variant(args.command, variant, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
