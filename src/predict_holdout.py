"""Produce ``predictions.csv`` for the public holdout set from the submitted
single model.

For every holdout photo: encode the three views, take the cosine top-K training
images by global descriptor, rerank that shortlist by the late-interaction
score (``local_rerank_weight``), and emit the top-1 match's coordinate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

import country_aware_retrieval_finetune as cv
from data import DEFAULT_DATA_CONFIG, ThreeViewTransform, read_json
from retrieval_model import GeoCPRegNetRetrieval

ROOT = Path(__file__).resolve().parents[1]
TRAINER = cv.TRAINER
COMMON = cv.COMMON

DATA_ROOT = Path("/var/tmp/luli38se-geomatch/data/geo_dataset")
DEFAULT_MODEL = Path(
    "/var/tmp/luli38se-geomatch/outputs/regnet_cp_full_data/model_final.pt"
)
DEFAULT_NORMALIZATION = ROOT / "artifacts/normalization/fold_0.json"
DEFAULT_OUTPUT = ROOT / "predictions.csv"
ASSET_ROOT = ROOT / "artifacts/fold_assignments"


class ImageFolderDataset(Dataset):
    def __init__(self, filenames: list[str], directory: Path, transform) -> None:
        self.filenames = filenames
        self.directory = directory
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict:
        path = self.directory / self.filenames[index]
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        global_view, left_view, right_view = self.transform(image)
        return {
            "filename": self.filenames[index],
            "global_view": global_view,
            "left_view": left_view,
            "right_view": right_view,
        }


@torch.inference_mode()
def encode(model, loader, device: torch.device) -> dict:
    descriptors, tokens, names = [], [], []
    for batch in loader:
        views = tuple(
            batch[name].to(device, non_blocking=True)
            for name in ("global_view", "left_view", "right_view")
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(*views)
        descriptors.append(
            torch.nn.functional.normalize(output["retrieval_descriptor"].float(), dim=1)
            .cpu()
            .numpy()
        )
        tokens.append(output["local_tokens"].float().cpu().numpy().astype(np.float16))
        names.extend(batch["filename"])
    return {
        "descriptor": np.concatenate(descriptors).astype(np.float32),
        "local_tokens": np.concatenate(tokens).astype(np.float16),
        "filename": np.asarray(names),
    }


def bank_frame() -> pd.DataFrame:
    frames = [
        pd.read_csv(ASSET_ROOT / f"fold_{fold}" / "validation_assignments.csv")
        for fold in range(5)
    ]
    rows = pd.concat(frames, ignore_index=True).drop_duplicates("filename")
    return rows.sort_values("filename").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    device = torch.device("cuda")

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    ema = checkpoint.get("ema")
    state = (
        ema["model"]
        if isinstance(ema, dict) and "model" in ema
        else checkpoint["model"]
    )
    model = GeoCPRegNetRetrieval(local_features=64, local_grid_size=4).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    data_config = read_json(DEFAULT_DATA_CONFIG)
    normalization = read_json(DEFAULT_NORMALIZATION)
    transform = ThreeViewTransform(
        normalization["mean"], normalization["std"], training=False, config=data_config
    )

    bank_rows = bank_frame()
    bank_dataset = ImageFolderDataset(
        bank_rows["filename"].astype(str).tolist(), args.data_root / "train", transform
    )
    holdout_names = sorted(
        p.name for p in (args.data_root / "holdout_public").iterdir()
    )
    holdout_dataset = ImageFolderDataset(
        holdout_names, args.data_root / "holdout_public", transform
    )

    def loader(dataset):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

    bank = encode(model, loader(bank_dataset), device)
    if not np.array_equal(
        bank["filename"].astype(str), bank_rows["filename"].astype(str).to_numpy()
    ):
        raise RuntimeError("bank encode order differs from the frame order")
    query = encode(model, loader(holdout_dataset), device)

    bank_coordinates = bank_rows[["lat", "lng"]].to_numpy(dtype=np.float64)
    top_k = int(config["evaluation"]["shortlist_top_k"])
    local_weight = float(config["evaluation"]["local_rerank_weight"])

    indices, scores = COMMON.matrix_topk(
        query["descriptor"], bank["descriptor"], top_k, device
    )
    reranked_indices, _, _ = TRAINER.rerank_with_local_tokens(
        query["local_tokens"],
        bank["local_tokens"],
        indices,
        scores,
        local_weight,
        device,
    )
    predicted = bank_coordinates[reranked_indices[:, 0]]

    frame = pd.DataFrame(
        {
            "filename": query["filename"],
            "pred_lat": predicted[:, 0],
            "pred_lng": predicted[:, 1],
        }
    )
    if len(frame) != 2400:
        raise RuntimeError(f"expected 2400 holdout rows, produced {len(frame)}")
    frame.to_csv(args.output, index=False)
    print(
        json.dumps(
            {
                "rows": len(frame),
                "output": str(args.output),
                "shortlist_top_k": top_k,
                "local_rerank_weight": local_weight,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
