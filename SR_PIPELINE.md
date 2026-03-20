# SolarRes SR Suite

This repo now includes a reusable training package in `solarres_sr/` so you can compare `SRCNN`, `RLFB+ESA`, `EDSR`, `RCAN`, `SRGAN`, `ESRGAN`, and `Diffusion SR` on the same dataset pipeline.

## Why this helps

- `Diffusion SR` is the default recommendation path for solar-detail recovery.
- `EDSR` and `RCAN` are added as strong non-GAN baselines.
- Capacity is chosen automatically from dataset size when `--capacity auto` is used.
- Validation-based checkpointing, LR scheduling, gradient clipping, augmentation, and early stopping reduce overfitting and underfitting mistakes.
- `benchmark_sr_models.py` ranks models by validation score instead of choosing by guesswork.

## Train one model

```bash
python train_sr.py --model diffusion_sr --epochs 50 --batch-size 4
python train_sr.py --model rcan --epochs 50 --batch-size 4
python train_sr.py --model edsr --epochs 50 --batch-size 4
```

## Benchmark the main candidates

```bash
python benchmark_sr_models.py --models diffusion_sr rcan edsr --epochs 40 --batch-size 4
```

## Useful options

- `--input-mode auto`
  Uses each model's recommended default. Solar-specific models default to `solar_features`.
- `--capacity auto`
  Picks `tiny`, `base`, or `large` from training set size.
- `--max-val-batches N`
  Speeds up experiments when you want a quick comparison run.
- `--apply-clahe`
  Switches the solar preprocessing to CLAHE instead of log scaling.

## Output

Each run saves:

- `config.json`
- `history.json`
- `summary.json`
- `best_model.pt`

under `checkpoints/sr_suite/` or the `--save-root` you provide.
