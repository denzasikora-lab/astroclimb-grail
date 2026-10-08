#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract deduplicated frozen object embeddings.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("joint", "text", "vision"), required=True)
    parser.add_argument("--name", required=True, help="Short output name, for example clip or dino")
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path")
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional immutable Hugging Face model revision for reproducible extraction.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Tokenizer limit. Omit to use the checkpoint's declared model_max_length.",
    )
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    parser.add_argument(
        "--image-preprocess",
        choices=("native", "letterbox"),
        default="native",
        help="Use white square padding for crop-based encoders; keep native aspect for NaFlex",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalized_fp16(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values /= np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-8, None)
    return values.astype(np.float16)


def tensor_from_output(
    output: object, *, pooling: str, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    pooler = getattr(output, "pooler_output", None)
    if pooler is not None and pooling == "cls":
        return pooler
    hidden = output.last_hidden_state
    if pooling == "cls":
        return hidden[:, 0]
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def move_batch(batch: object, device: torch.device) -> object:
    if isinstance(batch, dict):
        return {key: value.to(device) for key, value in batch.items()}
    return batch.to(device)


class ManualImageNetProcessor:
    """Torchvision-free processor for ImageNet-normalized vision checkpoints."""

    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(
        self, *, images: list[Image.Image], return_tensors: str
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("ManualImageNetProcessor supports return_tensors='pt' only")
        tensors: list[torch.Tensor] = []
        for image in images:
            resized = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            values = np.asarray(resized, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(values).permute(2, 0, 1)
            tensors.append((tensor - self.mean) / self.std)
        return {"pixel_values": torch.stack(tensors)}


@torch.inference_mode()
def encode_texts(
    texts: list[str],
    *,
    model: object,
    processor: object,
    device: torch.device,
    batch_size: int,
    max_length: int | None,
    pooling: str,
    joint: bool,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_text = texts[start : start + batch_size]
        tokenization = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if max_length is not None:
            tokenization["max_length"] = max_length
        inputs = (
            processor(text=batch_text, **tokenization)
            if joint
            else processor(batch_text, **tokenization)
        )
        inputs = move_batch(inputs, device)
        if joint:
            encoded = tensor_from_output(model.get_text_features(**inputs), pooling="cls")
        else:
            output = model(**inputs)
            encoded = tensor_from_output(
                output,
                pooling=pooling,
                attention_mask=inputs.get("attention_mask"),
            )
        blocks.append(encoded.float().cpu().numpy())
        if start % (batch_size * 20) == 0:
            print(f"text {start:,}/{len(texts):,}", flush=True)
    return normalized_fp16(np.concatenate(blocks))


@torch.inference_mode()
def encode_images(
    hashes: list[str],
    image_dir: Path,
    *,
    model: object,
    processor: object,
    device: torch.device,
    batch_size: int,
    pooling: str,
    joint: bool,
    image_preprocess: str,
    letterbox_side: int,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for start in range(0, len(hashes), batch_size):
        batch_hashes = hashes[start : start + batch_size]
        images: list[Image.Image] = []
        for digest in batch_hashes:
            with Image.open(image_dir / f"{digest}.jpg") as source:
                image = source.convert("RGB")
                if image_preprocess == "letterbox":
                    canvas = Image.new("RGB", (letterbox_side, letterbox_side), "white")
                    left = (letterbox_side - image.width) // 2
                    top = (letterbox_side - image.height) // 2
                    canvas.paste(image, (left, top))
                    image = canvas
                images.append(image)
        inputs = processor(images=images, return_tensors="pt")
        inputs = move_batch(inputs, device)
        if joint:
            encoded = tensor_from_output(model.get_image_features(**inputs), pooling="cls")
        else:
            output = model(**inputs)
            encoded = tensor_from_output(output, pooling=pooling)
        blocks.append(encoded.float().cpu().numpy())
        if start % (batch_size * 20) == 0:
            print(f"image {start:,}/{len(hashes):,}", flush=True)
    return normalized_fp16(np.concatenate(blocks))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = args.cache_dir / "manifest.json"
    if not cache_manifest_path.exists():
        raise FileNotFoundError(f"Missing object-cache manifest: {cache_manifest_path}")
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if cache_manifest.get("image_failures"):
        raise RuntimeError("Object cache contains image decode failures and is unsafe to embed")
    if cache_manifest.get("status") not in (None, "complete"):
        raise RuntimeError(
            f"Object cache status is {cache_manifest.get('status')!r}, expected 'complete'"
        )
    device = choose_device(args.device)
    started = time.monotonic()
    print(f"Loading {args.model} on {device}")
    pretrained_kwargs = {"revision": args.revision} if args.revision else {}
    model = AutoModel.from_pretrained(args.model, **pretrained_kwargs).to(device).eval()
    processor_backend = "transformers"
    if args.kind == "joint":
        processor = AutoProcessor.from_pretrained(args.model, **pretrained_kwargs)
    elif args.kind == "text":
        processor = AutoTokenizer.from_pretrained(args.model, **pretrained_kwargs)
    else:
        try:
            processor = AutoImageProcessor.from_pretrained(args.model, **pretrained_kwargs)
        except ImportError:
            processor = ManualImageNetProcessor(
                image_size=int(getattr(model.config, "image_size", 224))
            )
            processor_backend = "manual_imagenet_torchvision_free"

    manifest: dict[str, object] = {
        "kind": args.kind,
        "name": args.name,
        "model": args.model,
        "requested_revision": args.revision,
        "device": str(device),
        "pooling": args.pooling,
        "image_preprocess": args.image_preprocess,
        "processor_backend": processor_backend,
        "max_length": args.max_length,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "cache_manifest": cache_manifest,
    }

    def write_embedding_artifacts(
        embeddings: np.ndarray,
        hashes: pd.DataFrame,
        modality: str,
    ) -> None:
        if hashes["hash"].astype(str).duplicated().any():
            raise ValueError(f"Duplicate hashes in {modality} cache mapping")
        if len(hashes) != len(embeddings):
            raise ValueError(f"{modality} embedding and mapping row counts differ")
        array_path = args.output_dir / f"emb_{args.name}_{modality}.npy"
        array_tmp = args.output_dir / f"emb_{args.name}_{modality}.npy.tmp"
        with array_tmp.open("wb") as handle:
            np.save(handle, embeddings)
        array_tmp.replace(array_path)
        hash_path = args.output_dir / f"hashes_{args.name}_{modality}.parquet"
        hash_tmp = args.output_dir / f"hashes_{args.name}_{modality}.parquet.tmp"
        hashes[["hash"]].to_parquet(hash_tmp, index=False)
        hash_tmp.replace(hash_path)
        digest = hashlib.sha256()
        for value in hashes["hash"].astype(str):
            digest.update(value.encode("ascii"))
            digest.update(b"\n")
        manifest[f"{modality}_mapping_sha256"] = digest.hexdigest()

    if args.kind in ("joint", "text"):
        text_frame = pd.read_parquet(args.cache_dir / "texts.parquet")
        embeddings = encode_texts(
            text_frame["text"].astype(str).tolist(),
            model=model,
            processor=processor,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
            joint=args.kind == "joint",
        )
        write_embedding_artifacts(embeddings, text_frame, "text")
        manifest["text_shape"] = list(embeddings.shape)

    if args.kind in ("joint", "vision"):
        image_frame = pd.read_parquet(args.cache_dir / "image_hashes.parquet")
        embeddings = encode_images(
            image_frame["hash"].astype(str).tolist(),
            args.cache_dir / "images",
            model=model,
            processor=processor,
            device=device,
            batch_size=args.batch_size,
            pooling=args.pooling,
            joint=args.kind == "joint",
            image_preprocess=args.image_preprocess,
            letterbox_side=int(cache_manifest.get("image_side", 384)),
        )
        write_embedding_artifacts(embeddings, image_frame, "image")
        manifest["image_shape"] = list(embeddings.shape)

    manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
    provenance = args.output_dir / f"provenance_{args.name}.json"
    provenance_tmp = args.output_dir / f"provenance_{args.name}.json.tmp"
    provenance_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    provenance_tmp.replace(provenance)
    print(manifest)


if __name__ == "__main__":
    main()
