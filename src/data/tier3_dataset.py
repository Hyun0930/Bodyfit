"""
Tier 3 평가셋 Dataset — 라벨 포함.

labels.json 구조:
{
  "squat/abc123_rep00": {"label": 1, "joints": ["knee"], "reason": "...", ...},
  ...
}

Usage:
    from src.data.tier3_dataset import Tier3Dataset
    dataset = Tier3Dataset(root="data/test", labels_path="data/test/labels.json")
    pose, body, label = dataset[0]
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = Path(os.environ["BODYFIT_DATA"]) if "BODYFIT_DATA" in os.environ else ROOT / "data"
TEST_DIR = _DATA_ROOT / "test"

EXERCISES = ["squat", "bench", "deadlift", "ohp"]


class Tier3Dataset(Dataset):
    """Tier 3 평가셋: (pose, body, label) 반환.

    Args:
        root: data/test 루트 디렉토리
        labels_path: labels.json 경로 (기본: root/labels.json)
        exercises: 로드할 종목 (기본: 전체 4종목)
    """

    def __init__(
        self,
        root: Path | str | None = None,
        labels_path: Path | str | None = None,
        exercises: list[str] | None = None,
    ):
        self.root = Path(root) if root else TEST_DIR
        labels_path = Path(labels_path) if labels_path else self.root / "labels.json"
        self.exercises = exercises or EXERCISES

        if not labels_path.exists():
            raise FileNotFoundError(f"labels.json 없음: {labels_path}\n"
                                    "label_tier3.py 실행 후 labels_draft.json → labels.json 으로 복사하세요.")

        self.labels: dict = json.loads(labels_path.read_text())
        self.samples: list[tuple[Path, int]] = []  # (npz_path, label)

        for exercise in self.exercises:
            ex_dir = self.root / exercise
            if not ex_dir.exists():
                continue
            for npz_path in sorted(ex_dir.glob("*.npz")):
                key = f"{exercise}/{npz_path.stem}"
                entry = self.labels.get(key)
                if entry is None:
                    continue
                label = entry.get("label", -1)
                if label not in (0, 1):  # 미라벨(-1) 또는 파싱 실패 스킵
                    continue
                self.samples.append((npz_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        npz_path, label = self.samples[idx]
        data = np.load(npz_path, allow_pickle=True)
        pose = torch.tensor(data["pose"], dtype=torch.float32)   # (64, 33, 3)
        body = torch.tensor(data["body"], dtype=torch.float32)   # (7,)
        return pose, body, label

    def label_counts(self) -> dict[str, int]:
        labels = [lbl for _, lbl in self.samples]
        return {"normal": labels.count(0), "abnormal": labels.count(1)}
