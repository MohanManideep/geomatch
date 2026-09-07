"""Self-supervised (BYOL) pretraining of the RegNetY-400MF trunk on the
fold-``k`` training images only.

Rationale: with the true country given, retrieval still misses badly in Germany
(253 km) and France (244 km) even though a <=50 km training photo exists for
~100% of those rows -- the from-scratch classification-warmstarted backbone has
not learned features that discriminate location *within* a large visually
homogeneous country.  BYOL on the same fold-``k`` images (no labels, no external
data, no pretrained weights) gives the trunk a much sharper general visual
representation to start the supervised retrieval training from.

Output: ``backbone.pt`` -- a state dict keyed ``stem.*`` / ``stages.*`` that
loads directly into ``GeoCPRegNetY400MF`` / ``GeoCPRegNetRetrieval``.

Per-fold contract: fold-``k`` validation images are never opened here.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import regnet_y_400mf
from torchvision.transforms import InterpolationMode

from data import load_fold_assignments, read_json
from training import atomic_json, atomic_torch_save, set_seed, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = Path("/var/tmp/luli38se-geomatch/data/geo_dataset/train")
DEFAULT_OUTPUT = Path("/var/tmp/luli38se-geomatch/outputs/ssl_backbone")
FORMAT = "geomatch-regnet-cp-ssl-byol-backbone-v1"


class TwoViewDataset(Dataset):
    def __init__(self, filenames: list[str], image_dir: Path, transform) -> None:
        self.filenames = filenames
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        path = self.image_dir / self.filenames[index]
        with Image.open(path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        return self.transform(image), self.transform(image)


def build_transform(mean, std, size: int):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size,
                scale=(0.35, 1.0),
                ratio=(0.75, 1.3333),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.3),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


class BYOL(nn.Module):
    def __init__(self, projection_dim: int = 256, hidden: int = 2048) -> None:
        super().__init__()
        trunk = regnet_y_400mf(weights=None)
        self.feature_dim = trunk.fc.in_features
        trunk.fc = nn.Identity()
        self.stem = trunk.stem
        self.stages = trunk.trunk_output
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = mlp(self.feature_dim, hidden, projection_dim)
        self.predictor = mlp(projection_dim, hidden, projection_dim)

        self.target_stem = copy.deepcopy(self.stem)
        self.target_stages = copy.deepcopy(self.stages)
        self.target_projector = copy.deepcopy(self.projector)
        for module in (self.target_stem, self.target_stages, self.target_projector):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _encode(self, stem, stages, image: Tensor) -> Tensor:
        value = stem(image)
        for name in ("block1", "block2", "block3", "block4"):
            value = getattr(stages, name)(value)
        return self.pool(value).flatten(1)

    def online(self, image: Tensor) -> Tensor:
        return self.predictor(
            self.projector(self._encode(self.stem, self.stages, image))
        )

    @torch.no_grad()
    def target(self, image: Tensor) -> Tensor:
        return self.target_projector(
            self._encode(self.target_stem, self.target_stages, image)
        )

    @torch.no_grad()
    def update_target(self, tau: float) -> None:
        pairs = [
            (self.stem, self.target_stem),
            (self.stages, self.target_stages),
            (self.projector, self.target_projector),
        ]
        for online_module, target_module in pairs:
            for online_p, target_p in zip(
                online_module.parameters(), target_module.parameters()
            ):
                target_p.mul_(tau).add_(online_p.detach(), alpha=1.0 - tau)
            for online_b, target_b in zip(
                online_module.buffers(), target_module.buffers()
            ):
                target_b.copy_(online_b)

    def backbone_state_dict(self) -> dict:
        state = {}
        for key, value in self.stem.state_dict().items():
            state[f"stem.{key}"] = value.clone()
        for key, value in self.stages.state_dict().items():
            state[f"stages.{key}"] = value.clone()
        return state


def loss_fn(online_a, target_b, online_b, target_a) -> Tensor:
    def d(p, z):
        p = nn.functional.normalize(p, dim=1)
        z = nn.functional.normalize(z, dim=1)
        return 2.0 - 2.0 * (p * z).sum(dim=1)

    return (d(online_a, target_b) + d(online_b, target_a)).mean()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--base-tau", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=707011)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    device = torch.device("cuda")
    output = args.output / f"fold_{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed + args.fold * 1000)
    torch.backends.cudnn.benchmark = True

    normalization = read_json(ROOT / f"artifacts/normalization/fold_{args.fold}.json")
    training_rows, validation_rows = load_fold_assignments(args.fold)
    train_names = [str(v) for v in training_rows["filename"]]
    val_names = set(str(v) for v in validation_rows["filename"])
    if val_names & set(train_names):
        raise RuntimeError("fold split leak")
    transform = build_transform(
        normalization["mean"], normalization["std"], args.image_size
    )
    dataset = TwoViewDataset(train_names, args.images.resolve(), transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    model = BYOL().to(device)
    online_parameters = (
        list(model.stem.parameters())
        + list(model.stages.parameters())
        + list(model.projector.parameters())
        + list(model.predictor.parameters())
    )
    optimizer = torch.optim.AdamW(
        online_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * 5

    manifest = {
        "format": FORMAT,
        "fold": args.fold,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "lr": args.lr,
        "train_images": len(train_names),
        "base_tau": args.base_tau,
        "seed": args.seed,
    }
    atomic_json(output / "manifest.json", manifest)

    step = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for view_a, view_b in loader:
            view_a = view_a.to(device, non_blocking=True)
            view_b = view_b.to(device, non_blocking=True)
            if step < warmup_steps:
                lr_scale = (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_scale
            tau = 1.0 - (1.0 - args.base_tau) * (
                0.5 * (1.0 + math.cos(math.pi * step / total_steps))
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                online_a = model.online(view_a)
                online_b = model.online(view_b)
                target_a = model.target(view_a)
                target_b = model.target(view_b)
                loss = loss_fn(online_a, target_b, online_b, target_a)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_parameters, 5.0)
            optimizer.step()
            model.update_target(tau)
            step += 1
            epoch_loss += float(loss.detach()) * len(view_a)
            n += len(view_a)
        mean_loss = epoch_loss / n
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.perf_counter() - started
            print(
                f"ssl fold={args.fold} epoch={epoch}/{args.epochs} loss={mean_loss:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} tau={tau:.4f} elapsed={elapsed/60:.1f}m",
                flush=True,
            )
        if epoch % 20 == 0 or epoch == args.epochs:
            payload = {
                "format": FORMAT,
                "fold": args.fold,
                "epoch": epoch,
                "backbone": model.backbone_state_dict(),
                "manifest": manifest,
                "final_loss": mean_loss,
            }
            atomic_torch_save(output / "backbone.pt", payload)
            atomic_torch_save(output / f"backbone_epoch_{epoch:03d}.pt", payload)
    atomic_json(
        output / "done.json",
        {
            "format": FORMAT,
            "fold": args.fold,
            "final_loss": mean_loss,
            "epochs": args.epochs,
            "backbone_sha256": sha256_file(output / "backbone.pt"),
        },
    )
    print(f"wrote {output/'backbone.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
