# Oxford-IIIT Pet Breed Classification with Deep Learning

This project compares CNN training strategies for classifying images of 10 cat and dog breeds from the Oxford-IIIT Pet Dataset. The experiments use PyTorch and compare training ResNet-18 from scratch, transfer learning with feature extraction, progressive fine-tuning, data augmentation, and a lightweight MobileNetV3-Small model.

## Project Goals

- Compare training from scratch against transfer learning on a limited dataset.
- Evaluate Feature Extraction versus Progressive Fine-Tuning.
- Measure the effect of data augmentation on fine-tuning.
- Compare ResNet-18 with MobileNetV3-Small.
- Evaluate model performance, training time, and trainable parameter counts.
- Inspect errors through misclassification examples and Grad-CAM visualization.

## Dataset

The notebook downloads the `trainval` split through `torchvision.datasets.OxfordIIITPet` and selects the following 10 breeds:

**Cats:** Abyssinian, Bengal, Birman, Bombay, British Shorthair  
**Dogs:** Beagle, Boxer, Chihuahua, Pug, Saint Bernard

The selected subset contains **996 images**, split using stratified sampling with random seed `42`:

| Subset | Images | Approx. Split |
|---|---:|---:|
| Training | 697 | 70% |
| Validation | 149 | 15% |
| Test | 150 | 15% |
| **Total** | **996** | **100%** |

The test set is created by re-splitting the selected samples from the dataset's `trainval` split; it is not the original official test split.

## Experiments

| ID | Model | Strategy | Trainable Components |
|---|---|---|---|
| B1 | ResNet-18 | From Scratch | Entire model; random initialization |
| B2 | ResNet-18 pretrained on ImageNet | Feature Extraction | Classifier head (`fc`) only |
| B3 | ResNet-18 pretrained on ImageNet | Progressive Fine-Tuning | `layer4` and `fc` |
| B4 | ResNet-18 pretrained on ImageNet | Fine-Tuning with Data Augmentation | `layer4` and `fc` |
| B5 | MobileNetV3-Small pretrained on ImageNet | Feature Extraction with Data Augmentation | Final classifier layer only |

### Training Configuration

- **Loss:** `CrossEntropyLoss`
- **Optimizer:** `AdamW`
- **Weight decay:** `1e-2`
- **Maximum epochs:** `25`
- **Early stopping:** patience `4`, minimum improvement delta `1e-4`, monitored on validation loss
- **Batch size:** `32`
- **Random seed:** `42`
- **Input size:** `224 × 224`
- **Device:** CUDA GPU if available; otherwise CPU

For B3 and B4, the learning rate is `1e-4` for `layer4` and `1e-3` for `fc`. Other trainable classifier heads use learning rate `1e-3`.

### Preprocessing and Augmentation

All images are resized, center-cropped to `224 × 224`, converted to tensors, and normalized using the ImageNet mean and standard deviation associated with the ResNet-18 pretrained weights.

The augmented training pipeline used by B4 and B5 adds:

- `RandomResizedCrop(224, scale=(0.8, 1.0))`
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(degrees=15)`
- `ColorJitter(brightness=0.2, contrast=0.2)`

Validation and test images use deterministic preprocessing without random augmentation.

## Results

The following values are from the executed notebook. Training times were recorded in a Google Colab environment with an NVIDIA Tesla T4 GPU, so timing may vary on other hardware.

| Experiment | Trainable Parameters | Learning Rate | Epochs Run | Training Time (s) | Best Validation Accuracy | Test Accuracy | Test Macro F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| B1 — From Scratch | 11,181,642 | `1e-3` | 17 | 92.13 | 42.95% | 48.00% | 46.99% |
| B2 — Feature Extraction | 5,130 | `1e-3` (`fc`) | 25 | 125.34 | 95.30% | 93.33% | 93.27% |
| B3 — Progressive Fine-Tuning | 8,398,858 | `layer4: 1e-4`, `fc: 1e-3` | 12 | 57.91 | 97.32% | 94.67% | **94.67%** |
| B4 — Fine-Tuning + Augmentation | 8,398,858 | `layer4: 1e-4`, `fc: 1e-3` | 7 | 48.27 | 95.30% | 94.67% | 94.61% |
| B5 — MobileNetV3-Small | 10,250 | `1e-3` (classifier) | 14 | 91.48 | 94.63% | 90.67% | 90.63% |

**Best Test Macro F1:** B3 (Progressive Fine-Tuning), at **94.67%**. B3 also ties B4 for the highest Test Accuracy at 94.67%.

These results come from one run with seed `42`; they should not be interpreted as estimates of performance variance across runs.

## Analysis and Visualization

The notebook includes:

- Training and validation loss curves for all experiment conditions.
- Training and validation accuracy curves.
- A consolidated experiment metrics table.
- Visualization of six misclassified test images from B4, including true label, predicted label, and prediction confidence.
- A Grad-CAM visualization using the `layer4` block of B4.

Note that the notebook's failure analysis and Grad-CAM are run on B4, while B3 has the highest Test Macro F1 in the recorded comparison.

## Requirements

The notebook uses Python and the following libraries:

- PyTorch
- Torchvision
- NumPy
- pandas
- Matplotlib
- scikit-learn
- OpenCV (`opencv-python`)

Install the dependencies with:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn opencv-python
```

For GPU training, install a PyTorch build compatible with the CUDA version available in your environment. The notebook can also run on CPU, but training may take longer. An internet connection is needed on the first run to download the dataset and pretrained model weights.

## How to Run

1. Clone or download this repository.
2. Install the dependencies listed above.
3. Open `Deep_Learning_Oxford_IIIT_Pet.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
4. Run the notebook cells in order, from environment setup through log export.
5. Wait for the Oxford-IIIT Pet dataset and pretrained weights to download on the first run.
6. Review the learning curves, summary metrics, failure analysis, and Grad-CAM output.

The notebook saves `experiment_metrics.csv` and `experiment_metrics.json` to the current working directory when the export cell is executed.

## Output Files

| File | Description |
|---|---|
| `Deep_Learning_Oxford_IIIT_Pet.ipynb` | Notebook containing dataset preparation, model training, evaluation, visualizations, and export code |
| `experiment_metrics.csv` | Summary metrics for experiments B1–B5 |
| `experiment_metrics.json` | Summary metrics and per-epoch loss, accuracy, and timing history |
| `data/` | Dataset download directory created by the notebook |

The notebook displays plots inline; it does not automatically export all plots or trained model weights as separate files.

## Reproducibility Notes

- The notebook sets random seed `42` for Python, NumPy, and PyTorch.
- cuDNN deterministic mode is enabled and benchmarking is disabled.
- The dataset split uses stratification and `random_state=42`.
- Training metrics and timings may still differ across software versions, devices, and execution environments.

## Limitations

- Only 10 breeds and 996 images are used.
- The test set is derived from the dataset's `trainval` split.
- Results are based on one random seed and one recorded run per experiment.
- Training times depend on the execution hardware.
- B5 differs from ResNet-18 experiments in both architecture and training setup, since it uses Feature Extraction with Data Augmentation.
- Deployment metrics such as inference latency, memory usage, and throughput are not measured.
