from __future__ import annotations

import argparse

from solarres_sr import TrainConfig, benchmark_models, candidate_order, list_available_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiple solar SR models and rank the best fit.")
    parser.add_argument("--models", nargs="*", default=["diffusion_sr", "rcan", "edsr"], choices=list_available_models())
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--discriminator-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-mode", default="auto", choices=["auto", "grayscale", "solar_features", "rgb"])
    parser.add_argument("--capacity", default="auto", choices=["auto", "tiny", "base", "large"])
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--max-val-batches", type=int, default=200)
    parser.add_argument("--diffusion-eval-steps", type=int, default=8)
    parser.add_argument("--diffusion-metric-batches", type=int, default=2)
    parser.add_argument("--adv-weight", type=float, default=5e-3)
    parser.add_argument("--target-range", default="zero_one", choices=["zero_one", "minus_one_one", "none"])
    parser.add_argument("--output-activation", default="auto", choices=["auto", "identity", "sigmoid", "tanh"])
    parser.add_argument("--apply-clahe", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--all-diffusion-centric", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = candidate_order(diffusion_centric=True) if args.all_diffusion_centric else args.models
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        discriminator_learning_rate=args.discriminator_learning_rate,
        weight_decay=args.weight_decay,
        input_mode=args.input_mode,
        capacity=args.capacity,
        device=args.device,
        num_workers=args.num_workers,
        max_val_batches=args.max_val_batches,
        diffusion_eval_steps=args.diffusion_eval_steps,
        diffusion_metric_batches=args.diffusion_metric_batches,
        adv_weight=args.adv_weight,
        target_range=args.target_range,
        output_activation=args.output_activation,
        apply_clahe=args.apply_clahe,
        save_root=args.save_root,
        strict_gpu=not args.allow_cpu_fallback,
    )
    summaries = benchmark_models(config, model_names=models, dataset_root=args.dataset_root)
    print("Leaderboard:")
    for index, summary in enumerate(summaries, start=1):
        print(
            f"{index}. {summary.model_name} | score={summary.selection_score:.3f} | "
            f"PSNR={summary.best_psnr:.3f} | SSIM={summary.best_ssim:.4f} | {summary.fit_diagnosis}"
        )


if __name__ == "__main__":
    main()
