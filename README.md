# UECC-IM2

MATLAB implementation of **UECC-IM2**, an enhanced Updatable Ensemble Classifier Chains framework for class-imbalance mitigation in multi-label classification.

UECC-IM2 combines label-wise feature selection, binary/trinary classifier generation, model selection, difficulty-aware diversity-driven oversampling, and progressive ensemble updating. The current entry script is `run_uecc_im2.m`.

## Method Overview

For each cross-validation fold, UECC-IM2 performs the following steps:

1. Loads the dataset and predefined training/test indices.
2. Converts the multi-label targets from `{0,1}` to `{-1,+1}`.
3. Selects a label-specific feature subset for each target label.
4. Trains two candidate models for each label:
   - a binary classifier;
   - a trinary classifier constructed using a selected FRONT label.
5. Selects the trinary model only when:
   - its training MCC is higher than that of the binary model; and
   - the original imbalance ratio of the target label is greater than `1.5`.
6. Uses the selected model to:
   - generate train/test predictions;
   - update the predicted-label feature slots used by subsequent classifiers;
   - estimate the difficulty of minority-class training samples.
7. Divides minority samples into difficulty bins and allocates oversampling according to the accumulated difficulty in each bin.
8. Applies label-diversity-based sample selection within each bin.
9. Updates the training indices for the next ensemble member.
10. Aggregates all ensemble predictions through majority voting.
11. Reports macro-level and minority/majority-oriented evaluation metrics.

## Repository Structure

```text
UECC-IM2/
├── data/
│   ├── <dataset>.mat
│   └── <dataset>_5folds.mat
├── label order/
│   └── label-order-related functions
├── metrics/
│   └── evaluation metric functions
├── liblinear/
│   └── MATLAB interface of LIBLINEAR
├── imbalance_characterization/
│   └── imbalance-related utility functions
├── BaseClassifierTrain.m
├── BaseClassifierPredict.m
├── cal_IR.m
├── IRLbl.m
├── Copy_of_scumble_between_labels.m
├── select_best_front_preference.m
├── featureSelection_improved.m
├── run_uecc_im2.m
└── README.md
```

The main script also calls the following functions. They must be available in the repository, one of the added subdirectories, or the parent directory:

```text
conditional_entropy_sorting
Train_3classes
Predict_3classes
label_diversity_oversampling
MCC
MacroF1_each_class
MacroF1_minority
MacroF1_majority
exAccu_basedMinority
```

## Requirements

- MATLAB
- MATLAB-compatible LIBLINEAR interface
- Dataset files prepared in the format described below
- All dependent functions listed above

Place the LIBLINEAR MATLAB files in:

```text
./liblinear/
```

The main script adds the following paths automatically:

```matlab
addpath('./data/');
addpath('./metrics/');
addpath('../');
addpath('./imbalance_characterization/');
addpath('./label order/');
addpath('./liblinear/');
```

Make sure these paths are valid relative to the working directory from which `run_uecc_im2.m` is executed.

## Dataset Format

Each dataset requires two MATLAB files.

### 1. Data file

```text
<dataname>.mat
```

The file must contain a variable named `num`:

```matlab
load([dataname, '.mat'], 'num');
```

`num` must store the feature matrix and label matrix in the following order:

```text
num = [features, labels]
```

For example, the current `flags` configuration is:

```matlab
x = num(:, 1:19);
y = num(:, 20:end);
```

The script also provides column ranges for several datasets:

```matlab
% enron:             x = num(:, 1:1001); y = num(:, 1002:end);
% emotions:          x = num(:, 1:72);   y = num(:, 73:end);
% birds:             x = num(:, 1:260);  y = num(:, 261:end);
% scene:             x = num(:, 1:294);  y = num(:, 295:end);
% yeast:             x = num(:, 1:103);  y = num(:, 104:end);
% CAL500:            x = num(:, 1:68);   y = num(:, 69:end);
% bibtex after FS:   x = num(:, 1:184);  y = num(:, 185:end);
```

Input labels are expected to use `{0,1}`. The script converts `0` to `-1` before training and evaluation.

### 2. Cross-validation file

```text
<dataname>_5folds.mat
```

The file must contain:

```matlab
train_ind
test_ind
```

Both variables are expected to be five-element cell arrays. Each cell contains the row indices for one training or test fold.

## Running the Code

### Step 1: Prepare dependencies

Confirm that:

- LIBLINEAR is available under `./liblinear/`;
- all helper functions are on the MATLAB path;
- the dataset and fold files are stored under `./data/`.

### Step 2: Configure the dataset

Edit the basic settings in `run_uecc_im2.m`:

```matlab
dataname = 'mlc_flags';
```

Then set the correct feature and label column ranges:

```matlab
x = num(:, 1:19);
y = num(:, 20:end);
```

### Step 3: Run the experiment

From the repository directory, execute:

```matlab
run_uecc_im2
```

The random seed is fixed by:

```matlab
rng(1);
```

This preserves reproducibility for operations that depend on MATLAB's random-number generator.

## Main Hyperparameters

The current implementation uses the following settings.

| Parameter | Current value | Role |
|---|---:|---|
| `ensemble_times` | `3` | Number of progressively updated ensemble members |
| `bins_number` | `10` | Number of minority-sample difficulty bins |
| `bin_length` | `1 / bins_number` | Width of each difficulty interval |
| model-selection IR threshold | `1.5` | Enables trinary-model selection only for imbalanced labels |
| oversampling IR threshold | `3` | Separates the two oversampling-budget rules |
| random seed | `1` | Controls reproducibility |

These values are experimental hyperparameters and should be examined through sensitivity analysis when the method is evaluated on new datasets.

## Label-Wise Feature Selection

For each target label, the script constructs an auxiliary input space from:

```text
original features + all other labels
```

The function:

```matlab
featureSelection_improved
```

returns the selected indices for the current target label. The selected subset is stored in:

```matlab
x_fs{fold, label}
```

When the trinary candidate is constructed, the selected FRONT label is added to the current label-specific input subset when necessary.

## Binary and Trinary Candidate Models

For each ensemble member and each target label, the script trains:

```matlab
classifier2
```

as the binary candidate and:

```matlab
classifier_3_1
classifier_3_2
```

as the trinary candidate.

The FRONT label is selected using:

```matlab
front_label = select_best_front_preference(...);
```

The candidate model is chosen according to training MCC:

```matlab
if MCC_train3 > MCC_train2 && IR_origin > 1.5
    use the trinary candidate
else
    use the binary candidate
end
```

The selected predictions are appended to the extended feature space and may therefore be used by subsequent label-wise classifiers.

## Difficulty-Aware Diversity-Driven Oversampling

For a minority-class sample, the script defines its difficulty from the selected model's probability assigned to the true minority class:

```matlab
difficulty = 1 - probability_of_true_minority_class;
```

The difficulty range is divided into ten bins. The oversampling budget assigned to each bin is proportional to the accumulated difficulty of the minority samples in that bin.

Within each bin, samples are selected by:

```matlab
label_diversity_oversampling
```

The selected sample indices are appended to the label-specific training index set for the next ensemble member. The implementation duplicates indices rather than generating synthetic feature vectors.

## Ensemble Prediction

Each ensemble member produces one complete multi-label prediction matrix. The final output is obtained by summing the member predictions and applying sign-based majority voting:

```matlab
final_output(final_output >= 0) = 1;
final_output(final_output < 0) = -1;
```

With an odd number of ensemble members, this gives an unambiguous majority decision. The current configuration uses three members.

## Evaluation Metrics

The script reports the following metrics:

- `MacroF1_each_class`
- mean label-wise `MCC`
- mean `MacroF1_minority`
- mean `MacroF1_majority`
- `exAccu_basedMinority`

The final five-fold averages are stored in:

```matlab
stats_avg
```

with the following order:

```text
[MacroF1_each_class,
 mean MCC,
 mean MacroF1_minority,
 mean MacroF1_majority,
 exAccu_basedMinority]
```

Additional experiment outputs include:

| Variable | Description |
|---|---|
| `stats_cell` | Fold-level metric matrices |
| `pred_cell` | Fold-level prediction matrices |
| `x_fs` | Label-specific selected feature indices |
| `MCC_train2_matrix` | Training MCC of binary candidates |
| `MCC_train3_matrix` | Training MCC of trinary candidates |
| `IR_update_matrix` | Label-wise imbalance ratios after index updating |

## Current Implementation Notes

The following points describe the present behavior of `run_uecc_im2.m`.

1. The script computes a SCUMBLE-based label ordering but subsequently resets:

   ```matlab
   label_order = 1:q;
   ```

   Therefore, the current execution uses the original label order.

2. The variables:

   ```matlab
   alf
   training_data_ratio
   ```

   are retained in the script but are not used by the main training and evaluation process.

3. Although several result containers are initialized for ten repetitions, the current loop is:

   ```matlab
   for experiment_times = 1:1
   ```

   Therefore, each fold is executed once.

4. `pred_cell` is initialized as a `5 × 10` cell array, and the single prediction matrix from each fold is copied into all ten cells of that fold.

5. The script expects all helper functions and directories referenced by `addpath` to exist. Missing functions or unresolved relative paths will stop execution.

6. The normalization:

   ```matlab
   average_prob_each_interval = ...
       average_prob_each_interval ./ sum(average_prob_each_interval);
   ```

   assumes that the total accumulated minority-sample difficulty is nonzero. A zero-sum safeguard should be added when adapting the code to datasets containing degenerate labels or empty candidate bins.

## Adapting the Code to a New Dataset

To run UECC-IM2 on another dataset:

1. Save the combined feature-label matrix as `num`.
2. Create five-fold `train_ind` and `test_ind` cell arrays.
3. Store both `.mat` files under `./data/`.
4. Update `dataname`.
5. Update the feature/label column split.
6. Verify that every label contains both classes in each training fold.
7. Check that all metric functions use the same `{-1,+1}` encoding.
8. Reassess the ensemble size, bin count, and IR thresholds through sensitivity analysis.

## Citation

When using this implementation in academic work, cite the associated study:

> Tielin Wang et al. *Updatable Ensemble Classifier Chains for Joint Class-Imbalance Mitigation in Multi-Label Classification*.

Publication metadata should be added here after the final bibliographic record becomes available.

## License

No license file is specified in the current repository structure. Add an explicit `LICENSE` file before public redistribution so that permitted use, modification, and distribution are clearly defined.
