%% Result-equivalent cleaned version
% This script is cleaned for readability, but preserves the original
% execution order, feature-selection behavior, random seed, output shapes,
% and prediction logic.

addpath('./data/');
addpath('./metrics/');
addpath('../');
addpath('./imbalance_characterization/');
addpath('./label order/');
addpath('./liblinear/');

rng(1);

%% Basic settings
dataname = 'mlc_flags';
alf = 1.5;
training_data_ratio = 0.8; 

load([dataname, '_5folds.mat'], 'train_ind', 'test_ind');
load([dataname, '.mat'], 'num');

%% Dataset split selection
% flags
x = num(:, 1:19);
y = num(:, 20:end);

% To switch datasets later, replace the two lines above with the desired split.
% Examples kept here for convenience:
% enron:                 x = num(:, 1:1001); y = num(:, 1002:end);
% emotions:              x = num(:, 1:72);   y = num(:, 73:end);
% birds:                 x = num(:, 1:260);  y = num(:, 261:end);
% scene:                 x = num(:, 1:294);  y = num(:, 295:end);
% yeast:                 x = num(:, 1:103);  y = num(:, 104:end);
% CAL500:                x = num(:, 1:68);   y = num(:, 69:end);
% bibtex (after FS):     x = num(:, 1:184);  y = num(:, 185:end);

%% Storage
pred_cell = cell(5, 10);
stats_cell = cell(5, 1);

alf = [num2str(alf), '.00'];
alf = alf(1:3); 

ylabels = size(y, 2);
q = size(y, 2);

x = sparse(x);

echo_max = 5;
x_fs = cell(echo_max, ylabels);

%% 5-fold cross-validation
for echo = 1:echo_max

    %% Label-order related calculations (preserved exactly)
    label_order = linspace(1, q, q);

    y_temp = y(:, label_order);
    y_train_temp = y_temp(train_ind{echo}, :);
    x_train_temp = x(train_ind{echo}, :); 

    IR_origin_LO = zeros(1, q);
    for count = 1:q
        IR_origin_LO(1, count) = cal_IR(y_train_temp(:, count));
    end

    front_labels_arr = zeros(1, q); 
    [~, condi_entropy_values] = conditional_entropy_sorting(y_train_temp); 

    scumble_matrix = zeros(q, q);
    for idx1 = 1:q
        for idx2 = 1:q
            scumble_matrix(idx1, idx2) = Copy_of_scumble_between_labels(y_train_temp, idx1, idx2);
        end
    end
    scumble_sum_arr = sum(scumble_matrix);
    [~, label_order] = sort(scumble_sum_arr, 'descend'); %#ok<ASGLU>

    label_order = linspace(1, q, q);
    y_temp = y(:, label_order);
    y_temp(y_temp == 0) = -1;

    %% Per-fold experiment containers (preserved shape)
    MCC_experiment = zeros(1, 10);
    MACROF1_experiment = zeros(1, 10);
    MacroF1_minority_experiment = zeros(1, 10);
    MacroF1_majority_experiment = zeros(1, 10);
    exAcc_experiment = zeros(1, 10);

    for experiment_times = 1:1

        disp("echo = " + echo);
        disp("experiment times = " + experiment_times);

        %% Train / test split
        y_train = y_temp(train_ind{echo}, :);
        x_train = x(train_ind{echo}, :);
        x_test = x(test_ind{echo}, :);
        y_test = y_temp(test_ind{echo}, :);
        m_train = size(y_train, 1);
        m_test = size(y_test, 1); 

        %% Label-wise feature selection (kept exactly as original behavior)
        for idx_FS = 1:q
            y_train_else = y_train;
            y_train_else(:, idx_FS) = [];

            selected_feature_idx = featureSelection_improved([x_train, y_train_else], y_train(:, idx_FS));
            selected_feature_idx(find(selected_feature_idx == size(x, 2) + idx_FS)) = []; %#ok<FNDSB>

            if size(selected_feature_idx, 1) ~= 1
                selected_feature_idx = selected_feature_idx';
            end

            x_fs{echo, idx_FS} = selected_feature_idx;
        end

        %% Hyperparameters
        ensemble_times = 3;
        bins_number = 10;
        bin_length = 1 / bins_number;

        %% Ensemble prediction containers
        ensemble_pre_output = cell(1, ensemble_times);
        for count = 1:ensemble_times
            ensemble_pre_output{1, count} = zeros(size(y_test));
        end

        %% Training index containers
        train_ensemble_idx = cell(ensemble_times, q);
        for count = 1:q
            train_ensemble_idx{1, count} = linspace(1, size(y_train, 1), size(y_train, 1));
        end

        %% Difficulty-distribution containers
        minority_diff_distribution_train2_cell = cell(1, q);
        minority_diff_distribution_test2_cell = cell(1, q);
        majority_diff_distribution_train2_cell = cell(1, q);
        majority_diff_distribution_test2_cell = cell(1, q);

        minority_diff_distribution_train3_cell = cell(1, q);
        minority_diff_distribution_test3_cell = cell(1, q);
        majority_diff_distribution_train3_cell = cell(1, q);
        majority_diff_distribution_test3_cell = cell(1, q);

        for count = 1:q
            minority_diff_distribution_train2_cell{1, count} = zeros(ensemble_times, bins_number);
            minority_diff_distribution_test2_cell{1, count} = zeros(ensemble_times, bins_number);
            majority_diff_distribution_train2_cell{1, count} = zeros(ensemble_times, bins_number);
            majority_diff_distribution_test2_cell{1, count} = zeros(ensemble_times, bins_number);

            minority_diff_distribution_train3_cell{1, count} = zeros(ensemble_times, bins_number);
            minority_diff_distribution_test3_cell{1, count} = zeros(ensemble_times, bins_number);
            majority_diff_distribution_train3_cell{1, count} = zeros(ensemble_times, bins_number);
            majority_diff_distribution_test3_cell{1, count} = zeros(ensemble_times, bins_number);
        end

        %% Original IR
        IR_origin_arr = zeros(1, q);
        for count = 1:q
            IR_origin_arr(1, count) = cal_IR(y_train(:, count));
        end

        %% Metrics used internally for model selection
        MCC_train2_matrix = zeros(ensemble_times, q);
        MCC_train3_matrix = zeros(ensemble_times, q);
        IR_update_matrix = zeros(ensemble_times, q);

        %% Extend features with q predicted-label slots
        [n, d] = size(x_train);
        n_test = size(x_test, 1);
        x_train = [x_train, zeros(n, q)];
        x_test = [x_test, zeros(n_test, q)];

        %% ECC loop
        for ensemble_count = 1:ensemble_times
            for back_label = 1:q
                fprintf('.');

                if sum(y_train(:, back_label) == -1) >= sum(y_train(:, back_label) == 1)
                    maj_value_back_label = -1;
                    min_samples_num = sum(y_train(:, back_label) == 1);
                else
                    maj_value_back_label = 1;
                    min_samples_num = sum(y_train(:, back_label) == -1);
                end

                %% Binary classifier
                classifier2 = BaseClassifierTrain( y_train(train_ensemble_idx{ensemble_count, back_label}, back_label), x_train(train_ensemble_idx{ensemble_count, back_label}, x_fs{echo, back_label}) );

                [y_pred_train2, prob_train2] = BaseClassifierPredict(x_train(:, x_fs{echo, back_label}), classifier2);
                y_pred_train2(y_pred_train2 == 0) = -1;

                IR_update_matrix(ensemble_count, back_label) = cal_IR(y_train(train_ensemble_idx{ensemble_count, back_label}, back_label));
                MCC_train2_matrix(ensemble_count, back_label) = MCC(y_train(:, back_label), y_pred_train2);

                [y_pred_test2, ~] = BaseClassifierPredict(x_test(:, x_fs{echo, back_label}), classifier2);
                y_pred_test2(y_pred_test2 == 0) = -1;

                %% Trinary classifier
                y_train_logi = y_train;
                y_train_logi(y_train_logi == -1) = 0;
                x_train_temp = x_train;
                y_train_temp = y_train;
                x_train_temp = x_train_temp(:, 1:end-q);
                y_train_temp(y_train_temp == -1) = 0;
                y_train_temp = logical(y_train_temp);

                front_label = select_best_front_preference(x_train_temp, y_train_temp, back_label);

                if ~ismember(size(x, 2) + front_label, x_fs{echo, back_label})
                    x_fs{echo, back_label} = [x_fs{echo, back_label}, size(x, 2) + front_label];
                end

                [classifier_3_1, classifier_3_2] = Train_3classes(y_train(train_ensemble_idx{ensemble_count, back_label}, :), x_train(train_ensemble_idx{ensemble_count, back_label}, x_fs{echo, back_label}), back_label, front_label);

                [y_pred_train3, prob_train3] = Predict_3classes(classifier_3_1, classifier_3_2, x_train(:, x_fs{echo, back_label}));
                [y_pred_test3, ~] = Predict_3classes(classifier_3_1, classifier_3_2, x_test(:, x_fs{echo, back_label}));

                MCC_train3_matrix(ensemble_count, back_label) = MCC(y_train(:, back_label), y_pred_train3);

                %% Model choice preserved exactly
                if MCC_train3_matrix(ensemble_count, back_label) > MCC_train2_matrix(ensemble_count, back_label) && IR_origin_arr(back_label) > 1.5
                    prob_train = prob_train3;
                    y_pred_train = y_pred_train3;
                    y_pred_test = y_pred_test3;

                    ensemble_pre_output{1, ensemble_count}(:, back_label) = y_pred_test3;
                else
                    prob_train = prob_train2;
                    y_pred_train = y_pred_train2;
                    y_pred_test = y_pred_test2;

                    ensemble_pre_output{1, ensemble_count}(:, back_label) = y_pred_test2;
                end

                x_train(:, d + back_label) = y_pred_train;
                x_test(:, d + back_label) = y_pred_test;

                %% Update over-sampling indices
                average_prob_each_interval = zeros(1, bins_number);

                col_ind = (3 + maj_value_back_label) / 2;


                samples_ind_each_interval = cell(1, bins_number);

                for ind_count = 1:m_train
                    diff = 0;
                    if y_train(ind_count, back_label) ~= maj_value_back_label
                        diff = 1 - prob_train(ind_count, col_ind);
                        interval_ind_cur_samp = ceil(diff / bin_length);
                        if interval_ind_cur_samp == 0
                            interval_ind_cur_samp = 1;
                        end

                        minority_diff_distribution_train2_cell{1, back_label}(ensemble_count, interval_ind_cur_samp) = ...
                            minority_diff_distribution_train2_cell{1, back_label}(ensemble_count, interval_ind_cur_samp) + 1;

                        average_prob_each_interval(1, interval_ind_cur_samp) = ...
                            average_prob_each_interval(1, interval_ind_cur_samp) + diff;

                        samples_ind_each_interval{interval_ind_cur_samp} = [samples_ind_each_interval{interval_ind_cur_samp}, ind_count];
                    end
                end

                if IR_update_matrix(ensemble_count, back_label) > 3
                    average_prob_each_interval(1, [6 7 8]) = average_prob_each_interval(1, [6 7 8]) .* 2;
                else
                    average_prob_each_interval(1, [6 7 8]) = average_prob_each_interval(1, [6 7 8]) .* 1;
                end

                if IR_origin_arr(back_label) > 3 && ensemble_count <= ensemble_times
                    minority_over_sam_number = ceil((0.5 * m_train - 1.5 * min_samples_num) / ensemble_times);
                end
                if IR_origin_arr(back_label) <= 3 && ensemble_count <= ensemble_times
                    minority_over_sam_number = fix((m_train - 2 * min_samples_num) / ensemble_times);
                end

                average_prob_each_interval = average_prob_each_interval ./ sum(average_prob_each_interval);
                average_prob_each_interval = average_prob_each_interval .* minority_over_sam_number;
                average_prob_each_interval = round(average_prob_each_interval);
                minority_over_sam_number = sum(average_prob_each_interval); 

                minority_sample_indexes_repeat = [];
                for bins_count = 1:bins_number
                    if average_prob_each_interval(bins_count) ~= 0
                        number = average_prob_each_interval(bins_count);
                        alphabet = samples_ind_each_interval{bins_count};

                        ind_selected_each_bin = label_diversity_oversampling(x_train, y_train, back_label, alphabet, number);
                        minority_sample_indexes_repeat = [minority_sample_indexes_repeat, ind_selected_each_bin];
                    end
                end

                if ensemble_count < ensemble_times
                    train_ensemble_idx{ensemble_count + 1, back_label} = [train_ensemble_idx{ensemble_count, back_label}, minority_sample_indexes_repeat];
                end
            end
            fprintf('\n');
        end

        %% Ensemble voting
        final_output = ensemble_pre_output{1};
        if ensemble_times > 1
            for count = 2:ensemble_times
                final_output = final_output + ensemble_pre_output{count};
            end
        end

        final_output(final_output >= 0) = 1;
        final_output(final_output < 0) = -1;

        %% Final evaluation
        MCC_final_output_arr = zeros(1, q);
        f1_minority_final_output_arr = zeros(1, q);
        f1_majority_final_output_arr = zeros(1, q);

        for count = 1:q
            MCC_final_output_arr(1, count) = MCC(y_test(:, count), final_output(:, count));
            f1_minority_final_output_arr(1, count) = MacroF1_minority(y_test(:, count), final_output(:, count));
            f1_majority_final_output_arr(1, count) = MacroF1_majority(y_test(:, count), final_output(:, count));
        end

        macroF1_each_class_value = MacroF1_each_class(y_test, final_output);
        exAcc_value = exAccu_basedMinority(y_test, final_output);

        disp("MacroF1_each_class = " + macroF1_each_class_value);
        disp("MCC = " + mean(MCC_final_output_arr));
        disp("MacroF1_minority = " + mean(f1_minority_final_output_arr));
        disp("MacroF1_majority = " + mean(f1_majority_final_output_arr));
        disp("ExAcc = " + exAcc_value);

        MCC_experiment(1, experiment_times) = mean(MCC_final_output_arr);
        MACROF1_experiment(1, experiment_times) = macroF1_each_class_value;
        MacroF1_minority_experiment(1, experiment_times) = mean(f1_minority_final_output_arr);
        MacroF1_majority_experiment(1, experiment_times) = mean(f1_majority_final_output_arr);
        exAcc_experiment(1, experiment_times) = exAcc_value;

        pred_cell{echo, experiment_times} = final_output;
        for count_temp_pred_cell = 1:10
            pred_cell{echo, count_temp_pred_cell} = final_output;
        end
    end

    %% Preserve original output layout
    MCC_experiment = MCC_experiment';
    MACROF1_experiment = MACROF1_experiment';
    MacroF1_minority_experiment = MacroF1_minority_experiment';
    MacroF1_majority_experiment = MacroF1_majority_experiment';
    exAcc_experiment = exAcc_experiment';

    stats_matrix = [MACROF1_experiment, MCC_experiment, MacroF1_minority_experiment, MacroF1_majority_experiment, exAcc_experiment];
    stats_cell{echo, 1} = stats_matrix;
end


stats_avg = mean([
    stats_cell{1,1}(1,:);
    stats_cell{2,1}(1,:);
    stats_cell{3,1}(1,:);
    stats_cell{4,1}(1,:);
    stats_cell{5,1}(1,:)
], 1);
disp("Final MacroF1_each_class = " + stats_avg(1))
disp("Final MCC = " + stats_avg(2))
disp("Final MacroF1_minority = " + stats_avg(3))
disp("Final MacroF1_majority = " + stats_avg(4))
disp("Final ExAcc = " + stats_avg(5))
