"""Finetune bge-small on mined pairs (Colab GPU). Saves the model to --out.

python scripts/finetune_embedding.py --pairs data/pairs.jsonl --out artifacts/bge-ft
"""

from __future__ import annotations

import argparse

from frag.train.finetune_embedding import DEFAULT_BASE_MODEL, load_pairs, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", default="artifacts/bge-ft")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    out = train(
        load_pairs(args.pairs),
        out_dir=args.out,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print({"saved": out})


if __name__ == "__main__":
    main()
