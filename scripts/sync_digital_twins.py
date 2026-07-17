#!/usr/bin/env python3
"""Copy best April 2026 digital-twin runs into the package bundle."""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

DEV_ROOT = Path(
    "/Users/roquemev/Library/CloudStorage/OneDrive-OregonHealth&ScienceUniversity/Documents/"
    "OHSU/Research/JDRF_Project/JDRFProject_DiabetesAISimulator/T2D-simulator"
)
DEV_TWINS = (
    DEV_ROOT
    / "data/T2HelpDigitalTwins_BayesianOptimizationIndividualApril2026"
    / "AllInputs"  # dev-only folder name; not used in package code
)
PKG_TWINS = (
    Path(__file__).resolve().parents[1]
    / "src/t2dsim_ai/models/DigitalTwins"
)
OGTT_SRC = DEV_ROOT / "src/OGTT/models/6compartments/ProductionModel/NeuralOGTT.pt"
OGTT_DST = (
    Path(__file__).resolve().parents[1]
    / "src/t2dsim_ai/models/OGTT_productionModel_6compartments.pt"
)


def read_best_folder(subject_dir: Path) -> Path | None:
    logs = list(subject_dir.glob("log*"))
    if not logs:
        return None
    rows = []
    with open(logs[0], "r", encoding="utf-8") as handle:
        for line in handle:
            row = ast.literal_eval(line.strip())
            rows.append((row["target"], row["params"]))
    best_target, best_params = max(rows, key=lambda item: item[0])
    suffix = (
        f"Hypo{round(best_params['hypo_penalization'], 3)}_"
        f"Hyper{round(best_params['hyper_penalization'], 3)}"
    )
    candidate = subject_dir / suffix / f"S{subject_dir.name}"
    if candidate.exists():
        return candidate
    return None


def copy_twin(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    model_files = sorted(src_dir.glob("model*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model weights in {src_dir}")
    shutil.copy2(model_files[-1], dst_dir / "model.pt")
    for name in ("scaler_inputsPop.pkl", "info.csv"):
        shutil.copy2(src_dir / name, dst_dir / name)


def main() -> None:
    if not DEV_TWINS.exists():
        raise FileNotFoundError(f"Dev twin directory not found: {DEV_TWINS}")

    synced = 0
    for subject_dir in sorted(DEV_TWINS.glob("021-*")):
        best = read_best_folder(subject_dir)
        if best is None:
            print(f"skip {subject_dir.name}: no best run")
            continue
        dst = PKG_TWINS / f"S{subject_dir.name}"
        copy_twin(best, dst)
        synced += 1
        print(f"synced {subject_dir.name} -> {dst.name}")

    if OGTT_SRC.exists():
        shutil.copy2(OGTT_SRC, OGTT_DST)
        print(f"updated OGTT weights -> {OGTT_DST}")

    print(f"Done. Synced {synced} digital twins.")


if __name__ == "__main__":
    main()
