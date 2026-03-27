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
    parser.add_argument("--optimizer", default="auto", choices=["auto", "adam", "adamw"])
    parser.add_argument("--scheduler", default="auto", choices=["auto", "plateau", "cosine", "none"])
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.99)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--disable-ema-eval", action="store_true")
    parser.add_argument("--input-mode", default="auto", choices=["auto", "grayscale", "solar_features", "rgb"])
    parser.add_argument("--capacity", default="auto", choices=["auto", "tiny", "base", "large"])
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--max-train-batches", type=int, default=-1)
    parser.add_argument("--max-val-batches", type=int, default=200)
    parser.add_argument("--diffusion-eval-steps", type=int, default=8)
    parser.add_argument("--diffusion-metric-batches", type=int, default=2)
    parser.add_argument("--diffusion-sampler", default="ddpm", choices=["ddpm", "deterministic_fast"])
    parser.add_argument("--disable-diffusion-eval-stabilization", action="store_true")
    parser.add_argument("--disable-diffusion-clamp-pred-x0", action="store_true")
    parser.add_argument("--adv-weight", type=float, default=5e-3)
    parser.add_argument("--loss-pixel-mode", default="charbonnier", choices=["charbonnier", "l1", "mse"])
    parser.add_argument("--loss-pixel-weight", type=float, default=1.0)
    parser.add_argument("--loss-ssim-weight", type=float, default=0.15)
    parser.add_argument("--loss-edge-weight", type=float, default=0.1)
    parser.add_argument("--loss-fft-weight", type=float, default=0.0)
    parser.add_argument("--target-range", default="zero_one", choices=["zero_one", "minus_one_one", "none"])
    parser.add_argument("--output-activation", default="auto", choices=["auto", "identity", "sigmoid", "tanh"])
    parser.add_argument("--train-hr-patch-size", type=int, default=192)
    parser.add_argument("--disable-leakage-filter", action="store_true")
    parser.add_argument("--disable-profile-timing", action="store_true")
    parser.add_argument("--apply-clahe", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--all-diffusion-centric", action="store_true")
    parser.add_argument("--selection-metric", default="score", choices=["score", "psnr", "ssim", "bicubic_gap_psnr"])
    parser.add_argument("--verbose-debug", action="store_true")
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
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        min_learning_rate=args.min_learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        ema_decay=args.ema_decay,
        use_ema_for_eval=not args.disable_ema_eval,
        input_mode=args.input_mode,
        capacity=args.capacity,
        device=args.device,
        num_workers=args.num_workers,
        max_train_batches=args.max_train_batches if args.max_train_batches > 0 else None,
        max_val_batches=args.max_val_batches,
        diffusion_eval_steps=args.diffusion_eval_steps,
        diffusion_metric_batches=args.diffusion_metric_batches,
        diffusion_sampler=args.diffusion_sampler,
        diffusion_stabilize_eval=not args.disable_diffusion_eval_stabilization,
        diffusion_clamp_pred_x0=not args.disable_diffusion_clamp_pred_x0,
        adv_weight=args.adv_weight,
        loss_pixel_mode=args.loss_pixel_mode,
        loss_pixel_weight=args.loss_pixel_weight,
        loss_ssim_weight=args.loss_ssim_weight,
        loss_edge_weight=args.loss_edge_weight,
        loss_fft_weight=args.loss_fft_weight,
        target_range=args.target_range,
        output_activation=args.output_activation,
        train_hr_patch_size=args.train_hr_patch_size if args.train_hr_patch_size > 0 else None,
        deduplicate_splits_by_hr_hash=not args.disable_leakage_filter,
        profile_timing=not args.disable_profile_timing,
        apply_clahe=args.apply_clahe,
        save_root=args.save_root,
        strict_gpu=not args.allow_cpu_fallback,
        selection_metric=args.selection_metric,
        verbose_debug=args.verbose_debug,
    )
    summaries = benchmark_models(config, model_names=models, dataset_root=args.dataset_root)
    print("Leaderboard:")
    for index, summary in enumerate(summaries, start=1):
        print(
            f"{index}. {summary.model_name} | {summary.selected_metric}={summary.selected_metric_value:.3f} | "
            f"PSNR={summary.best_psnr:.3f} | SSIM={summary.best_ssim:.4f} | Bicubic={summary.bicubic_psnr:.3f} | "
            f"RMSE={summary.best_rmse:.4f} | Corr={summary.best_correlation:.4f} | "
            f"{summary.fit_diagnosis}"
        )


if __name__ == "__main__":
    main()
