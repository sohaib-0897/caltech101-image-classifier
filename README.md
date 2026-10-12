# Pet breed classifier (Oxford-IIIT Pet)

I trained a small CNN and an ImageNet-pretrained MobileNetV3-small to classify pet breeds on a laptop CPU. This project includes saved splits, training logs, error analysis and an image-upload API built with FastAPI.

## problem and dataset choice

The `torchvision.datasets.Caltech101(download=True)` downloader returned HTTP 404. I switched to `torchvision.datasets.OxfordIIITPet(download=True)`; the checkpoint predicts pet breeds. The download record is in [outputs/dataset_download.json](outputs/dataset_download.json).

Oxford-IIIT Pet fits this task: it has 37 breeds with about 200 images per breed, enough to train a small model on a CPU. The classes are fine-grained and can look alike, including Birman/Ragdoll and American Pit Bull Terrier/Staffordshire Bull Terrier. Its near-balanced class distribution makes imbalance handling a modest part of the project.

For a stratified 70/15/15 split, I combined the supplied Pet partitions, fixed seed 42 and saved 5,144 train, 1,102 validation and 1,103 test paths in [outputs/splits.csv](outputs/splits.csv). Paths are relative to `datasets/`. The checksum normalizes line endings so Windows and Linux checkouts keep the same split. Exact pixel duplicates are removed before splitting; duplicate groups with conflicting labels are excluded altogether.

## data findings

The inventory contains 7,349 labeled images across 37 breeds. Labels are zero-based integer IDs paired with torchvision's class names. Class counts range from 184 to 200, with median 200 and max/min ratio 1.087. Among the labeled images, there are 3 RGBA files and 0 grayscale files. Pillow decoded every file: there were 0 unreadable files, 0 exact pixel duplicate groups and 0 conflicting duplicate-label groups.

Widths range from 114 to 3,264 pixels; heights range from 103 to 2,606. Median width and height are 500 and 375. The saved [class counts](outputs/class_distribution.png), [image sizes](outputs/image_sizes.png), [inventory](outputs/image_inventory.csv) and [full report](outputs/exploration.json) record these checks.

In the [label contact sheet](outputs/label_review.png), Birman and Ragdoll examples share coat and face patterns, while the selected bulldog/terrier breeds have similar shapes. The classes overlap visually. That small review turned up no confirmed mislabeled examples, so I kept the provided labels. It does not cover near-duplicates or a full breed-label audit.

## preprocessing and imbalance

Images are converted to RGB with EXIF orientation applied. Validation/test images are resized to 128 x 128 and normalized with the ImageNet mean and standard deviation stored in the checkpoint. Training adds random resized crops, horizontal flips and light color jitter. Augmentation stays out of validation, testing and inference so their scores are repeatable. Small inputs reduce CPU work at the cost of fine detail and aspect ratio.

Macro-F1 drives model selection so each breed contributes equally; I report accuracy alongside it. On the best unweighted MobileNet settings, I compared ordinary cross-entropy with inverse-frequency class-weighted loss. Weights come only from training labels, and everything else, including the seed, stays fixed. Validation macro-F1 changed from 0.6828 to 0.6774 (difference -0.0054), so the final checkpoint keeps ordinary loss. This comparison covers one nearly balanced split and seed; variability across runs is unmeasured. The controls and weights are in [outputs/imbalance_comparison.json](outputs/imbalance_comparison.json).

## models and tuning

The from-scratch baseline is a CNN with four convolution blocks, batch normalization, global average pooling and dropout, trained for 8 epochs. It is small enough for CPU training and shows what this budget achieves without pretrained features.

With only 129 to 140 training images per breed, [ImageNet-pretrained MobileNetV3-small](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.mobilenet_v3_small.html) gives me a useful starting point. I trained its classifier head for 2 epochs, then fine-tuned the selected tail bottlenecks and final projection for 3 epochs at one tenth of the head learning rate. That lower rate changes pretrained features gradually. Backbone batch-normalization statistics stay frozen because the batches are small.

The search changes one setting at a time, without a tuning framework. Each run keeps its best checkpoint by validation macro-F1:

| run | learning rate | batch | tail bottlenecks |
|---|---:|---:|---:|
| mobile_1 | 0.001 | 32 | 1 |
| mobile_2 | 0.003 | 32 | 1 |
| mobile_3 | 0.001 | 16 | 1 |
| mobile_4 | 0.001 | 32 | 2 |

The weighting comparison left `mobile_4` as the selected run. All 6 completed runs used 128-pixel inputs, small batches and 4 CPU threads on a 13th Gen Intel(R) Core(TM) i5-13420H. Training plus validation took 76.4 minutes, excluding downloads, exploration, test evaluation and Docker builds. Few epochs and a small search kept this within the laptop budget. I left data-loader worker processes off to avoid Windows process overhead. Package versions and the machine are recorded in [outputs/environment.json](outputs/environment.json).

## results

The baseline and selected model were evaluated together in one final test pass. The [test audit](outputs/test_evaluation.json) records it, and the evaluator refuses another pass in that output folder. Only those two models have test scores below. Every image is included before applying the uncertainty threshold.

| run | weighted loss | val accuracy | val macro-F1 | test accuracy | test macro-F1 | test top-5 |
|---|---|---:|---:|---:|---:|---:|
| baseline | no | 15.43% | 0.1263 | 16.77% | 0.1463 | 48.50% |
| mobile_1 | no | 67.42% | 0.6709 | - | - | - |
| mobile_2 | no | 63.61% | 0.6383 | - | - | - |
| mobile_3 | no | 67.51% | 0.6749 | - | - | - |
| mobile_4 | no | 68.51% | 0.6828 | 67.63% | 0.6745 | 92.93% |
| weighted | yes | 68.06% | 0.6774 | - | - | - |

Every training run is in the [run CSV](outputs/run_log.csv); the [epoch logs](outputs/runs/) contain losses, scores and times. Full test metrics are in [outputs/test_metrics.json](outputs/test_metrics.json). For CPU latency, I used one thread and a warm model, with 100 repetitions on validation images. These times exclude HTTP overhead; p95 values are in the metrics JSON.

| model | checkpoint (MiB) | forward median (ms) | decode + preprocessing + forward median (ms) |
|---|---:|---:|---:|
| baseline | 0.33 | 3.99 | 12.60 |
| mobile_4 | 6.04 | 18.51 | 27.41 |

## error analysis

The largest two-way confusions were British Shorthair/Russian Blue with 13 errors, Birman/Ragdoll with 12, and American Pit Bull Terrier/Staffordshire Bull Terrier with 11. The [pair plot](outputs/confused_pairs.png) shows them; [outputs/confused_pairs.json](outputs/confused_pairs.json) records the counts and directions.

Staffordshire Bull Terrier was the weakest class by recall: precision 47.37%, recall 32.14%, with 28 test images. Beagle recall was 33.33% across 30 images. The [per-class CSV](outputs/final_per_class.csv) records every breed's precision and recall. Top-5 accuracy (92.93%) is higher than top-1 accuracy (67.63%); the correct breed often remains in the candidate list.

The [error grid](outputs/misclassified.png) shows the most confident mistakes. Several shaggy dogs were predicted as Havanese, and grey British Shorthair cats as Russian Blue. Their displayed probabilities round to 98-100%, so the threshold accepts those wrong labels. Accepted test accuracy was 88.85%; meeting the validation target did not guarantee the same accuracy on test. I kept the validation threshold.

## inference and how to run it

I tested Python 3.12.9. Use Python 3.10-3.12 and a virtual environment. For CPU installation:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m src.predict image.jpg --top-k 5
```

The small final checkpoint is committed at `checkpoints/final.pt`. It contains the weights, class names, preprocessing, dataset name, seed, split checksum and confidence threshold. `load_model()` caches it; `predict(image)` accepts a path, bytes or a Pillow image and returns top-k labels with softmax probabilities. Non-RGB images are converted to RGB and unreadable inputs raise a clear error.

Validation selected a confidence threshold of 0.85, seeking at least 90% accepted accuracy and 20% coverage. The rule keeps the largest coverage that meets the target. If none does, it keeps the highest accepted accuracy while maintaining minimum coverage. It met the target: validation accepted accuracy was 90.02% at 50.91% coverage. On test, accepted accuracy was 88.85% at 52.04% coverage. Predictions below the threshold return `"uncertain"` with the top-k probabilities still included. These probabilities have not been calibrated.

```sh
python -m uvicorn app:app --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/health
curl -F "file=@image.jpg" "http://127.0.0.1:8000/predict?top_k=5"
```

On Windows use `curl.exe` for the upload command. `MODEL_PATH` can point to another final checkpoint. The API rejects empty, unreadable and oversized uploads with HTTP errors.

```sh
docker build -t image-classifier .
docker run --rm -p 8000:8000 image-classifier
```

The container was built and checked with the final checkpoint, including both endpoints. It uses a slim Python image, CPU-only torch and an unprivileged user. I also ran 14 tests covering splitting, checkpoint loading, fine-tuning, preprocessing, confidence behavior, uploads and the test-use lock. Run them with `python -m pytest -q`.

To reproduce training, use fresh artifact folders so the shipped test results stay fixed. The split command recreates the same rows from the recorded seed and inventory:

```sh
python scripts/explore_data.py --dataset pets --output outputs/reproduction
python -m src.data --output outputs/reproduction
python -m src.train --run baseline --model cnn --epochs 8 --output outputs/reproduction --checkpoints checkpoints/reproduction
python -m src.train --search --output outputs/reproduction --checkpoints checkpoints/reproduction
python -m src.train --compare-imbalance --output outputs/reproduction --checkpoints checkpoints/reproduction
python -m src.evaluate --output outputs/reproduction --checkpoints checkpoints/reproduction
```

To try Caltech first, omit `--dataset pets`; the exploration script falls back if its download fails. I trained on CPU only and did not run GPU training. Seeds stay fixed; unsupported CUDA deterministic operations warn, as described in [PyTorch's deterministic-operations documentation](https://docs.pytorch.org/docs/2.8/generated/torch.use_deterministic_algorithms.html).

## limitations

This covers one seed, one split and short training. I have not measured run-to-run variation, and the small CNN may need more training or capacity.

The data checks cover exact decoded-pixel duplicates and a small label sample. Near-duplicates, repeated animals and broader label ambiguity were not audited.

The results cover Pet breeds because Caltech's download failed. I have not tested other image domains or unknown labels; the validation threshold is an abstention rule within this dataset.
