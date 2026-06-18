"""체크포인트 내용 확인."""
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="checkpoints/cvae/best.pt")
args = parser.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
print("keys:", list(ckpt.keys()))
print("threshold_95:", ckpt.get("threshold_95"))
