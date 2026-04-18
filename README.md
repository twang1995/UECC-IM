# UECC-IM2

UECC-IM2 is a MATLAB implementation of an updatable ensemble classifier-chain framework for imbalanced multi-label classification. The current codebase combines label-specific feature selection, binary/trinary label modeling, difficulty-aware minority oversampling, and ensemble voting in a 5-fold evaluation pipeline.

## Overview

Multi-label class imbalance is more complicated than standard binary imbalance because different labels may have different imbalance severities, and label dependencies can affect both prediction quality and resampling behavior. In this implementation, UECC-IM2 follows a chain-based workflow and updates the training process label by label and round by round.

At a high level, the current implementation does the following:

1. **Label-wise feature selection**  
   For each target label, the model performs feature selection using the original features together with the remaining labels as auxiliary inputs.

2. **Binary and trinary modeling**  
   For each label, the code trains a standard binary classifier and also builds a trinary-style decomposition using a selected front label. The trinary branch is implemented through `Train_3classes.m` and `Predict_3classes.m`.

3. **Model choice during training**  
   The implementation compares binary and trinary training behavior with MCC-based selection logic. When the trinary branch is better and the label imbalance is sufficiently strong, the trinary prediction path is used for that label in that ensemble round.

4. **Difficulty-aware minority oversampling**  
   Minority examples are scored according to prediction difficulty. These samples are bucketed into bins, and the next ensemble round reuses minority samples with a diversity-oriented oversampling strategy.

5. **Ensemble voting**  
   Multiple rounds are aggregated by voting to produce the final multi-label prediction.

The current main script runs **5-fold cross-validation**, stores fold-level predictions, and reports the following final metrics:

- `MacroF1_each_class`
- `MCC`
- `MacroF1_minority`
- `MacroF1_majority`
- `ExAcc`

## Repository structure

A typical directory layout is:

```text
UECC-IM2/
├── BaseClassifierPredict.m
├── BaseClassifierTrain.m
├── cal_IR.m
├── label_diversity_oversampling.m
├── Predict_3classes.m
├── Train_3classes.m
├── select_best_front_preference.m
├── run_uecc_im2.m
├── data/
├── imbalance_characterization/
├── label_order/
├── liblinear/
└── metrics/
```

### Main files and folders

- `run_uecc_im2.m`  
  Main entry script. It loads the dataset, performs 5-fold evaluation, trains UECC-IM2, and prints fold-level and final averaged metrics.

- `BaseClassifierTrain.m`, `BaseClassifierPredict.m`  
  Base binary classification wrapper functions.

- `Train_3classes.m`, `Predict_3classes.m`  
  Trinary-branch training and inference functions.

- `label_diversity_oversampling.m`  
  Diversity-aware minority oversampling used between ensemble rounds.

- `select_best_front_preference.m`  
  Front-label selection for the trinary branch.

- `cal_IR.m`  
  Computes label imbalance ratio.

- `metrics/`  
  Metric functions such as `MCC`, `MacroF1_each_class`, `MacroF1_minority`, `MacroF1_majority`, and `exAccu_basedMinority`.

- `imbalance_characterization/`  
  Helper functions related to imbalance characterization.

- `label_order/`  
  Helper functions for label-order related computations.

- `liblinear/`  
  LIBLINEAR dependency used by the base classifier wrapper.

- `data/`  
  Dataset files and the corresponding fold split files.

## Requirements

- MATLAB
- LIBLINEAR available in the repository under `liblinear/`
- Dataset `.mat` files and fold split files placed under `data/`

## Input data format

The main script expects:

1. A dataset file:
   - `data/<dataset_name>.mat`
2. A fold file:
   - `data/<dataset_name>_5folds.mat`

The dataset file should contain a matrix named `num`.

The script then splits `num` into:
- feature matrix `x`
- label matrix `y`

For the default configuration in `run_uecc_im2.m`, the code uses:

```matlab
x = num(:, 1:19);
y = num(:, 20:end);
```

which corresponds to the `mlc_flags` setting in the current script.

The script also contains commented split examples for several other datasets. To switch datasets, update:
- `dataname`
- the feature/label column split in the dataset section

## How to run

From MATLAB, open the repository root and run:

```matlab
run_uecc_im2
```

The script will:

1. add required paths
2. load the selected dataset and fold indices
3. run 5-fold cross-validation
4. print fold-level metrics
5. print final averaged metrics across the 5 folds

## Reported metrics

The current code reports these final metrics:

- **MacroF1_each_class**: multi-label Macro-F1 computed from class-wise F1 aggregation
- **MCC**: average label-wise Matthews correlation coefficient
- **MacroF1_minority**: average minority-oriented F1 across labels
- **MacroF1_majority**: average majority-oriented F1 across labels
- **ExAcc**: example-based accuracy computed by treating the minority class of each label as the positive class

`ExAcc` is computed by:

```matlab
exAccu_basedMinority(y_test, final_output)
```

## Notes on the current implementation

- The current script uses `rng(1)` for reproducibility.
- The default evaluation setting is 5-fold cross-validation.
- The current code keeps several internal containers to preserve the original execution behavior.
- The present implementation uses `ensemble_times = 3` and `bins_number = 10` in the main script.
- The final result displayed at the end of the script is the mean of the first row of each fold’s stored metric matrix.

## Example output

The script prints per-fold values and then final averages such as:

```text
Final MacroF1_each_class = ...
Final MCC = ...
Final MacroF1_minority = ...
Final MacroF1_majority = ...
Final ExAcc = ...
```

## Citation / description

If you use this repository, please cite the corresponding UECC-IM2 paper or thesis chapter once it is publicly available.

## Disclaimer

This repository reflects the current research code version prepared for experiment reproduction and GitHub release. It is intended for academic use and further extension rather than as a polished software package.
