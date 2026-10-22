# Radio Signal Object Detection (Kaggle Competition)

Full project documentation and error analysis (in Romanian): [Smarandoiu_Eduard-Stefan_231_doc.pdf](./Smarandoiu_Eduard-Stefan_231_doc.pdf)[cite: 1]

## Overview and Dataset

The goal of this project is to classify grayscale radio signal images (`128x55` resolution) into 5 classes, where the class label corresponds to the number of vertical signal lines (objects) visible against heavy background noise[cite: 1, 2]. The dataset contains 15,500 training images and 5,500 test images in PNG format. 

Following the main competition constraint, I did not use any external data or pretrained models[cite: 1]. Everything in this repository—from the k-NN classifier and multi-scale Local Binary Patterns (LBP) feature extractor to the custom Convolutional Neural Network—was written from scratch using `NumPy`, `PyTorch`, and `scikit-learn` (used strictly for SVM and Naive Bayes)[cite: 1, 4, 5].

When inspecting the data early on, computing the mean image across each class revealed five nearly identical gray rectangles sharing the same horizontal black artifact line in the middle[cite: 2]. Because overall pixel intensity does not separate the classes, any method relying on raw pixel comparisons was bound to fail; the actual signal lies in the fine local structure and line count[cite: 2].

## Classical Baselines (80/20 Split, Seed 42)

Before moving to deep learning, I tested several classical approaches to establish a baseline[cite: 3]:

1. **Multinomial Naive Bayes:** Tested on both raw pixels and discretized LBP features across 3 to 11 bins[cite: 4]. Accuracy stayed around `0.22` (barely above the `0.20` random guessing threshold), which was expected since neighboring pixels along a signal line are heavily correlated and violate the feature independence assumption[cite: 4].
2. **k-Nearest Neighbors (from scratch):** Implemented in `NumPy` using Manhattan (L1) and Euclidean (L2) distances[cite: 4]. On raw pixels it scored below `0.21` due to the curse of dimensionality, while on LBP features it reached `0.370` for `k=7` with L1 distance[cite: 4, 5].
3. **Multi-scale LBP + RBF SVM:** I implemented an LBP extractor from scratch across 5 radii (1 to 5) and concatenated the resulting 8-bit neighborhood histograms into a 1,280-dimensional feature vector per image, normalized with `StandardScaler`[cite: 2, 5]. Tuning the RBF kernel parameters (`C=5` and increasing `gamma` to `1.5 * scale`, which alone added almost 6 percentage points) pushed validation accuracy to `0.531`[cite: 5, 6]. While local texture clearly carries signal, LBP only describes texture rather than counting objects, creating a performance ceiling around 53%[cite: 5, 6].

## Custom Residual CNN with Channel Attention

To learn spatial filters directly from the images, I built a custom residual CNN in PyTorch tailored to the geometry and noise profile of the dataset[cite: 7]:

* **Asymmetric Input Kernel (`5x3`):** Since the spectrograms are much taller (`128`) than wide (`55`), a rectangular filter captures more context along the vertical axis where the signal lines develop[cite: 2, 7].
* **Residual Blocks & Squeeze-and-Excitation (SE):** The backbone scales channels from `32` to `256` using skip connections to keep gradients stable, alternating stride-1 and stride-2 blocks before global pooling[cite: 7]. After each block, a Squeeze-and-Excitation module recalibrates channel weights between 0 and 1, allowing the network to suppress filters that only pick up background noise[cite: 7].
* **Preprocessing & Augmentation:** Each fold was z-score normalized using only its own training split statistics to avoid data leakage[cite: 3]. During training, I applied vertical translations (up to 5%), Mixup (`alpha = 0.3`), and Random Erasing / Cutout (`p = 0.5`) so the model would not rely on a single region of the spectrogram[cite: 3].
* **Training & Ensembling:** Trained with Cross-Entropy loss and label smoothing (`0.05`), `AdamW` with weight decay, a `OneCycle` learning rate schedule, automatic mixed precision (AMP), and gradient clipping[cite: 7]. I used 5-fold stratified cross-validation, ensembling the 5 models by averaging their predicted probabilities along with 9-direction weighted Test-Time Augmentation (TTA)[cite: 7, 8].

## Results

| Approach / Configuration | Validation / OOF Accuracy |
| :--- | :---: |
| Multinomial Naive Bayes (raw pixels / LBP) | `0.2274` / `0.2265`[cite: 4] |
| Custom k-NN on LBP (`k=7`, L1 distance) | `0.3700`[cite: 5] |
| Multi-scale LBP (5 radii) + RBF SVM (`C=5`, `gamma=1.5*scale`) | `0.5310`[cite: 5, 6] |
| Custom CNN (4 folds, 35 epochs, no TTA) | `0.7360`[cite: 8] |
| Custom CNN (4 folds, 40 epochs, 5-dir equal TTA) | `0.7420`[cite: 8] |
| **Custom CNN (5 folds, 55 epochs, 9-dir weighted TTA)** | **`0.7530` (OOF) / `0.7800` (Public Leaderboard)**[cite: 8] |

## Key Takeaways from the Confusion Matrix

Evaluating the out-of-fold predictions across all 15,500 training images shows a recall of `0.95` on Class 1, gradually dropping to `0.74`, `0.70`, `0.67`, and `0.65` on Class 5 as overlapping lines become harder to count in heavy noise[cite: 8, 9]. More importantly, even though the model was trained with standard multi-class Softmax rather than regression, almost all misclassifications occur between adjacent classes (for instance, Class 5 is mostly confused with Class 4, rarely with Class 1), showing that the network implicitly learned the ordinal structure of the counting problem[cite: 3, 8]. A natural next step would be testing ordinal classification and class-specific weighting for higher object counts[cite: 3, 9].
