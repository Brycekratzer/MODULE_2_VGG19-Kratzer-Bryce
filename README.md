# MODULE_2_VGG19_Pruning2-Kratzer-Bryce

## Overview

This repository fine-tunes an existing pre-trained VGG19 model from PyTorch's model zoo. One baseline model is trained on the CIFAR-10 dataset, which features 32x32 pixel images of various objects. We then prune using an unstructured approach on the fine-tuned model and evaluate the performance from both software and hardware perspectives.

The model uses a cross-entropy loss function with a 0.0001 learning rate. The model underwent 10 epochs of training. At the end of each epoch, F1, recall, and precision scores were calculated along with the average validation loss to track overall progress between epochs.

## File Strucutre

```bash
MODULE_1_VGG19-KRATZER-BRYCE/
├── VGG19_NoteBooks/                  # Notebooks for training/evaluating
│   ├── VGG19.ipynb             
│   └── Visuals.ipynb
├── Model_Metrics/                          # Outputs from final test for both models
│   ├── Inference_Metrics/
│   │   ├── test_history_NO_PRUNE.csv
│   │   ├── test_history_unstructured_prune_low_setting.csv
│   │   ├── test_history_unstructured_prune_medium_setting.csv
│   │   └── test_history_unstructured_prune_high_setting.csv
│   └── Training_Metrics/
│       ├── training_outputst.txt
│       └── training_history_CIFAR10.csv
├── Metrics/                                # Visuals of Model performance after pruning
│   ├── latency.png
│   ├── accuracy(h).png
│   ├── accuracy(n,l,m).png
│   └── throughput.png
├── README.md
└── requirement.txt
```

## Setup Steps

1) At the root directory run `pip install -r requirement.txt`

2) Go to the `./VGG19_NoteBooks` directory

3) The directory has a notebook that documents the following:

    - Importing needed Libraries
    - Importing Model
    - Importing Dataset
    - Splitting Dataset
    - Creating Functions for
        - Calculating Precision, Recall, and F1 scores
        - Single Epoch execution
        - Validation execution
    - Initializing Loss Function
    - Pruning Methodology
    - Testing & Hardware Analysis

## Testing Results

From the results, we can conclude that while pruning does speed up inference times through reduced latency and increased throughput, accuracy is sacrificed. No pruning of the fine-tuned model yielded the best results, while the lightly pruned model yielded the second-best results.

The highest prune model had the lowest latency and the highest throughput but achieved the worst results compared to the other prune types.

**Test Set Details:**

- CIFAR-10: 5,000 test images

### Hardware Performance

The highest pruned model demonstrated superior hardware efficiency on a per-image basis, while the model with no pruning demonstrated superior accuracy across all metrics:

No Pruning:

- **Latency**: 0.0732 seconds per image
- **Throughput**: 13.35 samples/second
- **F1 Score**: .9011
- **Accuracy**: 90.42%

Light Pruning:

- **Latency**: 0.0705 seconds per image
- **Throughput**: 13.93 samples/second
- **F1 Score**: .8997
- **Accuracy**: 90.26%

Medium Pruning:

- **Latency**: 0.0719 seconds per image
- **Throughput**: 13.67 samples/second
- **F1 Score**: .8590
- **Accuracy**: 86.36%

High Pruning:

- **Latency**: 0.0681 seconds per image
- **Throughput**: 14.43 samples/second
- **F1 Score**: .7265
- **Accuracy**: 73.88%

### Trade-off Analysis

The pruning experiments reveal a clear accuracy-performance trade-off:

**Performance Gains**:

High pruning achieves a 7.0% reduction in latency (0.0732s → 0.0681s) and an 8.1% increase in throughput (13.35 → 14.43 samples/second) compared to the un-pruned model

Light pruning offers a more balanced approach with 3.7% latency reduction while maintaining near-baseline accuracy (90.26% vs 90.42%)

**Accuracy Costs**:

- **Light pruning**: Minimal impact (-0.16 percentage points in accuracy)
- **Medium pruning**: Moderate degradation (-4.06 percentage points in accuracy)
- **High pruning**: Significant loss (-16.54 percentage points in accuracy, dropping to 73.88%)
