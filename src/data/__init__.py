import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

EXERCISES = ["squat", "bench", "deadlift", "ohp"]


class BodyFitDataset(Dataset):
    """
    data/processed/{exercise}/*.npz 에서 (pose, body) 쌍 로드.

    Args:
        exercises: 로드할 종목 목록 (기본: 전체 4종목)
        root: processed 디렉토리 경로 오버라이드
    """

    def __init__(
        self,
        exercises: list[str] | None = None,
        root: Path | None = None,
    ):
        base = root or PROCESSED_DIR
        targets = exercises or EXERCISES
        self.samples: list[Path] = []

        for ex in targets:
            self.samples.extend(sorted((base / ex).glob("*.npz")))

        if not self.samples:
            raise FileNotFoundError(f"No .npz files found under {base} for {targets}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.samples[idx], allow_pickle=True)
        pose = torch.from_numpy(data["pose"].astype(np.float32))   # (64, 33, 3)
        body = torch.from_numpy(data["body"].astype(np.float32))   # (7,)
        return pose, body

    def get_meta(self, idx: int) -> dict:
        data = np.load(self.samples[idx], allow_pickle=True)
        return json.loads(str(data["meta"]))

    def split(self, train_ratio: float = 0.9, seed: int = 42) -> tuple["Subset", "Subset"]:
        """재현 가능한 train/val 분리."""
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self))
        n_train = int(len(self) * train_ratio)
        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train:].tolist()
        return Subset(self, train_idx), Subset(self, val_idx)
