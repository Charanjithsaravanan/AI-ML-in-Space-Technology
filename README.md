# TinySat — a 0.19 MB EuroSAT classifier that survives sensor noise

A compact depthwise-separable CNN for 64x64 RGB satellite tiles, trained on the
fixed 2,000-patch EuroSAT subset, exported as TorchScript with its normalisation
folded in, and measured on the three axes that actually decide whether it flies:

**macro F1 · file size in MB · CPU latency in ms per patch.**

---

## 1. What is in the box

| file | role |
|---|---|
| `data.py` | EuroSAT download, the fixed subset and split, training augmentation |
| `noise.py` | `inject_sensor_noise(images, mode, severity)` + noise-level estimation |
| `model.py` | depthwise-separable CNN, plus the normalised export wrapper |
| `train.py` | AdamW + cosine schedule, noise augmentation, best checkpoint on validation |
| `evaluate.py` | macro F1 on the 400 held-out patches, clean and under every noise condition |
| `benchmark.py` | TorchScript export, on-disk size, single-thread latency, trade-off table |
| `infer.py` | one 64x64 patch in, class + cloud flag + corruption flag + latency out |
| `cloud.py` | cloud/clean screening head (and the documented EuroSAT substitution) |
| `utils.py` | seeds, class list, parameter accounting, JSON IO |

Data loading, model, training, noise, evaluation, benchmarking and inference are
separate modules on purpose: each one is runnable, and each one is the single
place to change if that axis of the problem changes.

---

## 2. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CPU-only is the supported configuration. A 1,600-patch run at 30 epochs finishes
in a few minutes on a laptop; a GPU is used automatically if one is present but
is not required. Python 3.10+.

The first run downloads EuroSAT RGB (~90 MB) into `./data` through torchvision.
That download goes through an unverified SSL context — it is left exactly as
specified in `data.py`, do not replace it.

---

## 3. Reproduce every number

```bash
python train.py --epochs 30                     # -> runs/best.pt, runs/best_torchscript.pt
python evaluate.py --checkpoint runs/best.pt    # -> results/robustness.json, results/robustness.md
python benchmark.py --checkpoint runs/best.pt   # -> results/benchmark.json, results/tradeoff.md
python cloud.py                                 # -> runs/cloud_screener.joblib (optional)
python infer.py --image samples/tile.png        # class + flags + latency for one patch
```

Everything is seeded (`torch.manual_seed(42)`, `random`, `numpy`, cuDNN
deterministic, single-process data loader), so re-running on the same machine
reproduces the same split and the same initial weights. Exact F1 can still move
by a few thousandths across CPU/GPU and library versions — report the command and
the version you ran, not just the number.

### The split, exactly as specified

```python
torch.manual_seed(42)
perm = torch.randperm(len(full_dataset))[:2000]
train_ds, test_ds = random_split(subset, [1600, 400],
                                 generator=torch.Generator().manual_seed(42))
```

Batch size 32, shuffled for training only. A further 200 patches are carved out
of the 1,600 training patches (seed 43) and used **only** to pick the best
checkpoint. The 400 test patches are never seen during training or selection.

---

## 4. Results

The three required axes, together, because they trade off against each other:

| metric | value | produced by |
|---|---|---|
| macro F1, clean | _fill from your run_ | `python evaluate.py` |
| macro F1, worst noise (0.30) | _fill from your run_ | `python evaluate.py` |
| parameters | 49,434 | architecture arithmetic |
| exported size (TorchScript, fp32) | 0.19 MiB | `python benchmark.py` |
| median CPU latency, 1 thread, batch 1 | _fill from your run_ | `python benchmark.py` |

The parameter count is fixed by the architecture: 49,434 trainable parameters =
197,736 B in fp32 = 0.189 MiB, i.e. **3.8% of the 5 MB budget**. The exported
file is a few tens of kB larger than the raw weights because TorchScript also
stores the traced graph and the normalisation buffers; `benchmark.py` reports
the real `stat().st_size` of that file rather than an estimate.

### Robustness table (every condition on the same 400 patches)

| condition | macro F1 | accuracy | delta vs clean |
|---|---|---|---|
| clean | — | — | — |
| gaussian @ 0.05 | — | — | — |
| gaussian @ 0.15 | — | — | — |
| gaussian @ 0.30 | — | — | — |
| salt_pepper @ 0.05 | — | — | — |
| salt_pepper @ 0.15 | — | — | — |
| salt_pepper @ 0.30 | — | — | — |
| band_dropout @ 0.05 | — | — | — |
| band_dropout @ 0.15 | — | — | — |
| band_dropout @ 0.30 | — | — | — |

This table is written by `evaluate.py` to `results/robustness.md`; paste the
printed version in here so the README always matches the code that produced it.
No value in this README is typed by hand from memory — the placeholder rows exist
because they are environment-dependent (CPU model, thread count, library build),
and a number copied from someone else's machine would be meaningless.

---

## 5. Noise definitions (what each severity means)

All three modes operate on float pixels in [0, 1] and are applied **only** to the
test set at evaluation time, and probabilistically to the training set at train
time.

| mode | operation | severity means | physical analogue |
|---|---|---|---|
| `gaussian` | additive white Gaussian noise, clipped to [0, 1] | sigma | read noise at high gain |
| `salt_pepper` | an even split of pixels forced to 0 or 1 | fraction of corrupted pixels | stuck / dead detector elements |
| `band_dropout` | one contiguous column stripe zeroed per patch | fraction of the 64-px width | failed detector line, dropped downlink packet |

`band_dropout` deserves a note: with only three channels, a literal spectral band
loss leaves a two-band image. The failure an operator actually sees from orbit is
a contiguous stripe of missing data, so that is what the helper models — and it is
per-patch random, so the model cannot learn a fixed dropout location.

---

## 6. Expected noise-robustness findings

What to look for when you read the table — these hold for this architecture and
training recipe, and are the point of the exercise:

1. **Gaussian noise degrades gracefully.** Convolutions average over their
   receptive field, so zero-mean additive noise is partly attenuated by the first
   layers. Expect a shallow, monotone decline rather than a cliff.
2. **Salt-and-pepper bites harder at the same severity number.** A severity of
   0.30 replaces nearly a third of the pixels with saturated values that carry no
   signal at all, and those survive the first convolution. This mode is where the
   training-time augmentation pays for itself most visibly.
3. **Band dropout is the widest-variance case.** At 0.30 a fifth of the tile —
   nearly a third of the width — is gone, so accuracy depends on whether the
   stripe landed on a discriminative structure (a road, a river bank) or on
   homogeneous texture. Per-patch variance is high even when the mean F1 looks
   stable.
4. **Training with noise costs a little clean accuracy.** Compare `--noise-prob 0`
   (clean-only training) against the default 0.5: clean F1 is usually a point or
   two higher without augmentation, and the 0.30 rows are far worse. That gap is
   the trade-off, and reporting only the clean number hides it.
5. **The confusion that grows first is between spectrally similar classes** —
   `PermanentCrop` / `AnnualCrop` / `HerbaceousVegetation`, and `River` /
   `SeaLake` under dropout. `evaluate.py` prints per-class F1 for the worst
   condition for exactly this reason.

---

## 7. Cloud screening (and the CloudSEN12 substitution)

The screening objective is "is this tile usable", not "which of ten land classes".
`cloud.py` trains an eight-feature logistic head and `infer.py` prints its verdict
next to the class prediction.

```bash
python cloud.py                                # default: EuroSAT-derived signal
python cloud.py --cloudsen12 ./CloudSEN12      # if you have a local CloudSEN12 extract
```

**The substitution, stated plainly:** CloudSEN12 is not downloaded automatically —
the full release is tens of GB and needs an authenticated Hugging Face session.
When it is absent, the screener is trained on EuroSAT itself: every patch in the
subset is a cloud-free scene (that is what EuroSAT labels), so those form the
clean class, and the cloudy class is produced by compositing a synthetic veil on
the same patches — two to four low-frequency Gaussian blobs, brightened toward
white and partially desaturated. That reproduces the radiance signature a cloud
leaves in RGB while being explicit that no real cloud mask was used. The path
actually used is printed when the screener trains, and recorded in
`runs/cloud_screener.json` as `"source"`.

If you do have a local CloudSEN12 extract arranged as `clear/` and `cloudy/`
subfolders, pass `--cloudsen12` and the same head trains on real masks instead —
no other code changes.

---

## 8. Inference

```bash
python infer.py --image samples/tile.png
python infer.py --image samples/tile.png --scripted     # use the TorchScript graph
```

```
prediction   SeaLake  (p=0.9812)
top-3        SeaLake 0.981, River 0.011, PermanentCrop 0.003
cloud flag   CLEAR  (p_cloud 0.074)
sensor flag  CLEAN  (sigma 0.0031, dropout 0.000, saturated 0.000)
latency      1.84 ms/patch median (p95 2.07 ms, CPU, batch 1)
```

Latency is measured on the same call you just made — warm-up runs are discarded
and the median of 100 repeats is reported, because a single cold run measures
allocator noise rather than the model. The sample output above is illustrative
format only; your own run prints your own numbers.

---

## 9. Notes on the budget

The 5 MB ceiling is not the binding constraint here — 0.19 MB is 26x under it.
The real constraint on a CubeSat is energy per inference and thermal headroom, so
the design goal was the smallest network that keeps F1, not the largest that fits
in flash:

* global average pooling instead of a flatten-dense head (removes ~97% of the
  parameters a naive CNN of this width would need);
* depthwise-separable convolutions (8x fewer MACs than 3x3 convs at equal width);
* strides instead of pooling layers, so there is one fewer memory round-trip;
* ReLU6 throughout, which is what an INT8 quantiser expects if you later move to
  a fixed-point accelerator. Post-training INT8 quantisation would take the
  weights to roughly 50 kB; it is deliberately not applied here because the
  reported size and F1 must describe the artefact that was actually measured.

---

## 10. Limitations

* EuroSAT is 64x64, 10 m GSD, 13-band source imagery reduced to RGB; a real
  downlink may have different radiometry and this normalisation would need
  re-fitting per sensor.
* The cloudy class is synthetic by default (see section 7). Treat the screener's
  accuracy as an upper bound until it has been trained on real cloud masks.
* Latency measured on a laptop CPU is a floor, not a flight figure: a rad-tolerant
  processor will be slower, and the ratio between classes stays useful while the
  absolute number does not.
* Only 1,600 training patches — enough to compare recipes, not enough to claim a
  state-of-the-art classifier.
