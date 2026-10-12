# image classification

## problem and dataset choice

I built an image classifier with a small training budget and an image-upload API. I first tried `torchvision.datasets.Caltech101(download=True)`. Its downloader returned HTTP 404, so I used `torchvision.datasets.OxfordIIITPet(download=True)` as the requested fallback. This checkpoint predicts pet breeds. The download record is in [outputs/dataset_download.json](outputs/dataset_download.json).

I combined the supplied Pet partitions and made my own stratified 70/15/15 split. I fixed seed 42 and saved 5,144 train, 1,102 validation and 1,103 test paths in [outputs/splits.csv](outputs/splits.csv). Paths are relative to `datasets/`. The split checksum normalizes line endings so a Windows/Linux checkout keeps the same split. Exact pixel duplicates are removed before splitting; conflicting duplicate labels are excluded altogether.

## data findings

I found 7,349 labeled images and 37 breeds. Labels are zero-based integer IDs paired with torchvision's class names. Class counts range from 184 to 200, with median 200 and max/min ratio 1.087. There are 3 RGBA files and 0 grayscale files among the labeled images. Pillow decoded all of them: I found 0 unreadable files, 0 exact pixel duplicate groups and 0 conflicting duplicate-label groups.

Widths range from 114 to 3,264 pixels; heights range from 103 to 2,606. Median width and height are 500 and 375. I saved [class counts](outputs/class_distribution.png), [image sizes](outputs/image_sizes.png), the [inventory](outputs/image_inventory.csv) and the [full report](outputs/exploration.json).

I reviewed the [label contact sheet](outputs/label_review.png). Birman and Ragdoll examples share coat and face patterns, and the selected bulldog/terrier breeds have similar shapes. These are visually overlapping classes. I found no confirmed mislabeled examples in that small review; I kept the provided labels. This check does not cover near-duplicates or a full breed-label audit.

## preprocessing and imbalance

I convert images to RGB, apply EXIF orientation, resize validation/test images to 128 x 128, and use the ImageNet mean and standard deviation stored in the checkpoint. Training uses random resized crops, horizontal flips and light color jitter. I keep augmentation out of validation, testing and inference so their scores are repeatable. The small input reduces CPU work, with a tradeoff in fine detail and aspect ratio.

I use macro-F1 as the selection metric so each breed contributes equally, alongside accuracy. I compared inverse-frequency class-weighted cross-entropy against ordinary cross-entropy on the best unweighted MobileNet settings. Weights come only from training labels. I kept ordinary loss for the final checkpoint because it had higher validation macro-F1. Everything else, including the seed, is held fixed. Validation macro-F1 changed from 0.6828 to 0.6774 (difference -0.0054). I measured this on one nearly balanced split and seed; variability across runs is unmeasured. The controls and weights are in [outputs/imbalance_comparison.json](outputs/imbalance_comparison.json).

## models and tuning

I used a CNN with four convolution blocks, batch normalization, global average pooling and dropout as a from-scratch reference. I trained it for 8 epochs. It is small enough for CPU training and shows what this budget achieves without pretrained features.

I used [ImageNet-pretrained MobileNetV3-small](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.mobilenet_v3_small.html) because the split has only 129 to 140 training images per breed. I trained the classifier head for 2 epochs, then fine-tuned the requested tail bottlenecks and final projection for 3 epochs at one tenth of the head learning rate. I used the lower backbone rate to change pretrained features gradually. I kept backbone batch-normalization statistics frozen because the batches are small.

I changed one setting at a time and chose the best checkpoint of each run by validation macro-F1. The search has no tuning framework:

| run | learning rate | batch | tail bottlenecks |
|---|---:|---:|---:|
| mobile_1 | 0.001 | 32 | 1 |
| mobile_2 | 0.003 | 32 | 1 |
| mobile_3 | 0.001 | 16 | 1 |
| mobile_4 | 0.001 | 32 | 2 |

I selected `mobile_4` after the weighting comparison. All 6 completed runs used 128-pixel inputs, small batches and 4 CPU threads on a 13th Gen Intel(R) Core(TM) i5-13420H. Total training plus validation took 76.4 minutes, excluding downloads, exploration, test evaluation and Docker builds. I used few epochs and a small search to stay inside the laptop budget. I used no data-loader worker processes to avoid Windows process overhead. Package versions and the machine are recorded in [outputs/environment.json](outputs/environment.json).

## results

I evaluated the baseline and selected model together in one final test pass. The test audit is [outputs/test_evaluation.json](outputs/test_evaluation.json); the evaluator refuses another pass in that output folder. Test columns are filled for those two models. All scores below include every image before applying the uncertainty threshold.

| run | weighted loss | val accuracy | val macro-F1 | test accuracy | test macro-F1 | test top-5 |
|---|---|---:|---:|---:|---:|---:|
| baseline | no | 15.43% | 0.1263 | 16.77% | 0.1463 | 48.50% |
| mobile_1 | no | 67.42% | 0.6709 | - | - | - |
| mobile_2 | no | 63.61% | 0.6383 | - | - | - |
| mobile_3 | no | 67.51% | 0.6749 | - | - | - |
| mobile_4 | no | 68.51% | 0.6828 | 67.63% | 0.6745 | 92.93% |
| weighted | yes | 68.06% | 0.6774 | - | - | - |

The [run CSV](outputs/run_log.csv) includes every training run, and [epoch logs](outputs/runs/) show losses, scores and times. Full test metrics are in [outputs/test_metrics.json](outputs/test_metrics.json). CPU latency uses one thread and a warm model, with 100 repetitions on validation images. It excludes HTTP overhead; p95 values are in the metrics JSON.

| model | checkpoint (MiB) | forward median (ms) | decode + preprocessing + forward median (ms) |
|---|---:|---:|---:|
| baseline | 0.33 | 3.99 | 12.60 |
| mobile_4 | 6.04 | 18.51 | 27.41 |

## error analysis

I found 13 errors between British Shorthair and Russian Blue, 12 between Birman and Ragdoll, and 11 between American Pit Bull Terrier and Staffordshire Bull Terrier. These are the largest two-way confusions in [the pair plot](outputs/confused_pairs.png). The counts and directions are in [outputs/confused_pairs.json](outputs/confused_pairs.json).

Staffordshire Bull Terrier was the weakest class by recall: precision 47.37%, recall 32.14%, with 28 test images. Beagle recall was 33.33% across 30 images. The [per-class CSV](outputs/final_per_class.csv) records every breed's precision and recall. Top-5 accuracy (92.93%) is higher than top-1 accuracy (67.63%); the correct breed often remains in the candidate list.

I put the most confident mistakes in [the error grid](outputs/misclassified.png). Several shaggy dogs were predicted as Havanese, and grey British Shorthair cats were predicted as Russian Blue. The displayed probabilities round to 98-100% despite the wrong labels, so the threshold accepts those cases. Accepted test accuracy was 88.85%; the validation target did not guarantee the same accuracy on test. I kept the threshold chosen on validation.

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

I chose confidence threshold 0.85 on validation, seeking at least 90% accepted accuracy and 20% coverage. I choose the largest coverage that meets the target; if none does, I choose the highest accepted accuracy while keeping the minimum coverage. The policy met that target: validation accepted accuracy was 90.02% at 50.91% coverage. On test, accepted accuracy was 88.85% at 52.04% coverage. Below the threshold I return `"uncertain"` while retaining the top-k probabilities. These probabilities have not been calibrated.

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

I built and checked this container with the final checkpoint, including both endpoints. It uses a slim Python image, CPU-only torch and an unprivileged user. I also ran 14 tests for splitting, checkpoint loading, fine-tuning, preprocessing, confidence behavior, uploads and the test-use lock. Run them with `python -m pytest -q`.

To reproduce training, use fresh artifact folders so the shipped test results stay fixed. The split command recreates the same rows from the recorded seed and inventory:

```sh
python scripts/explore_data.py --dataset pets --output outputs/reproduction
python -m src.data --output outputs/reproduction
python -m src.train --run baseline --model cnn --epochs 8 --output outputs/reproduction --checkpoints checkpoints/reproduction
python -m src.train --search --output outputs/reproduction --checkpoints checkpoints/reproduction
python -m src.train --compare-imbalance --output outputs/reproduction --checkpoints checkpoints/reproduction
python -m src.evaluate --output outputs/reproduction --checkpoints checkpoints/reproduction
```

For the original dataset attempt, omit `--dataset pets`; the exploration script tries Caltech first and falls back if its download fails. TODO: run the same training commands on a free Colab GPU with CUDA-enabled torch and `--device cuda`. I have no GPU results. Seeds stay fixed; unsupported CUDA deterministic operations warn, as described in [PyTorch's deterministic-operations documentation](https://docs.pytorch.org/docs/2.8/generated/torch.use_deterministic_algorithms.html).

## limitations

I used one seed, one split and short training. I have not measured run-to-run variation, and the small CNN may need more training or capacity.

I checked exact decoded-pixel duplicates and a small label sample. Near-duplicates, repeated animals and broader label ambiguity were not audited.

I tested Pet breeds because Caltech's download failed. I have not tested other image domains or unknown labels; the validation threshold is an abstention rule within this dataset.
