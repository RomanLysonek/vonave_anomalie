"""Run the optional DAVID-style multivariate demand autoencoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from framework import Config, load_raw
from systemic_autoencoder import AutoencoderConfig, fit_score_systemic_autoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/anomaly_autoencoder")
    parser.add_argument("--window", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.02)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train, _ = load_raw(Config())
    ae_cfg = AutoencoderConfig(
        window=args.window,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        evt_alpha=args.alpha,
    )
    scores, metadata, model = fit_score_systemic_autoencoder(train, ae_cfg)
    scores.to_csv(output / "systemic_autoencoder_scores.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, output / "model.pt")
    with open(output / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
