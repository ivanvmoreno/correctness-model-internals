import sys
from pathlib import Path
import os

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # adjust this if needed
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from collections import defaultdict
import pandas as pd
from src.classifying import (
    ActivationsHandler,
    combine_activations_handlers,
    get_correctness_direction_classifier,
    get_logistic_regression_classifier,
    get_between_class_variance_and_within_class_variance,
)
from src.visualisations.utils import plot_interactive_lineplot
from src.utils.data import load_activations, load_labels, get_experiment_activations_configs_df_subset

# BASE_PATH = Path(__file__).resolve().parent.parent.parent
BASE_PATH = "/runpod-volume/anton/correctness-model-internals/data_for_classification"
print(f"BASE_PATH: {BASE_PATH}")
# BASE_PATH = "../../"
PCA_COMPONENTS = None

MODEL_ID = "llama3.1_8b_chat"
DATASET_ID = None
PROMPT_ID = None
SUBSET_ID = None
INPUT_TYPE = None
layers_to_keep = [14]
n_folds = 5

activation_exp_configs_df = get_experiment_activations_configs_df_subset(
    base_path=BASE_PATH,
    model_id=MODEL_ID,
    dataset_id=DATASET_ID,
    prompt_id=PROMPT_ID,
    subset_id=SUBSET_ID,
    input_type=INPUT_TYPE,
)

print(activation_exp_configs_df)

# Drop things that are not needed
activation_exp_configs_df = activation_exp_configs_df[activation_exp_configs_df["dataset_id"].isin(
    ['trivia_qa_2_60k', 'cities_10k', 'birth_years_4k', 'medals_9k', 'math_operations_6k', 'gsm8k']
    # ['trivia_qa_2_60k']
)]
activation_exp_configs_df = activation_exp_configs_df[activation_exp_configs_df["prompt_id"] != "cot_3_shot"]


res_dict = defaultdict(list)

# for input_type in ["prompt_only", "prompt_answer"]:
for input_type in ["prompt_only"]:
    activation_exp_configs_df_ = activation_exp_configs_df[activation_exp_configs_df["input_type"] == input_type]
    for (model_id_train, dataset_id_train, prompt_id_train, subset_id_train, input_type_train), _ in activation_exp_configs_df_.groupby(["model_id", "dataset_id", "prompt_id", "subset_id", "input_type"]):
        for (model_id_test, dataset_id_test, prompt_id_test, subset_id_test, input_type_test), config_df in activation_exp_configs_df_.groupby(["model_id", "dataset_id", "prompt_id", "subset_id", "input_type"]):
            print(f"\n{model_id_train=}, {dataset_id_train=}, {prompt_id_train=}, {subset_id_train=}, {input_type_train=}")
            print(f"\n{model_id_test=}, {dataset_id_test=}, {prompt_id_test=}, {subset_id_test=}, {input_type_test=}")
            labels_df_train = load_labels(
                base_path=BASE_PATH,
                model_id=model_id_train,
                dataset_id=dataset_id_train,
                prompt_id=prompt_id_train,
                subset_id=subset_id_train,
            )

            labels_df_test = load_labels(
                base_path=BASE_PATH,
                model_id=model_id_test,
                dataset_id=dataset_id_test,
                prompt_id=prompt_id_test,
                subset_id=subset_id_test,
            )
            if dataset_id_train == "trivia_qa_2_60k":
                labels_df_train = labels_df_train[10000:].reset_index(drop=True)
            if dataset_id_test == "trivia_qa_2_60k":
                labels_df_test = labels_df_test[10000:].reset_index(drop=True)

            check_indices_train, check_indices_test = None, None
            for layer in config_df["layer"].astype(int).sort_values():
                if len(layers_to_keep) and layer not in layers_to_keep:
                    continue
                print(f"{layer=}", end=", ")
                activations_train, indices_train = load_activations(
                    base_path=BASE_PATH,
                    model_id=model_id_train,
                    dataset_id=dataset_id_train,
                    prompt_id=prompt_id_train,
                    subset_id=subset_id_train,
                    input_type=input_type_train,
                    layer=layer,
                )
                activations_test, indices_test = load_activations(
                    base_path=BASE_PATH,
                    model_id=model_id_test,
                    dataset_id=dataset_id_test,
                    prompt_id=prompt_id_test,
                    subset_id=subset_id_test,
                    input_type=input_type_test,
                    layer=layer,
                )

                # # remove the first 10000 samples
                if dataset_id_train == "trivia_qa_2_60k":
                    activations_train = activations_train[10000:]
                    indices_train = indices_train[10000:].reset_index(drop=True)
                    indices_train = indices_train - 10000
                if dataset_id_test == "trivia_qa_2_60k":
                    activations_test = activations_test[10000:]
                    indices_test = indices_test[10000:].reset_index(drop=True)
                    indices_test = indices_test - 10000

                # print(f"{activations_train.shape=}, {activations_test.shape=}")
                # print(f"{indices_train.shape=}, {indices_test.shape=}")
                # print(f"{labels_df_train.shape=}, {labels_df_test.shape=}")

                if check_indices_train is None:
                    check_indices_train = indices_train.sample(frac=1, replace=False, random_state=42)
                
                if set(indices_train) != set(check_indices_train):
                    raise RuntimeError(f"indices across layers are not the same")

                labels_df_subset_train = labels_df_train.iloc[check_indices_train]
                activations_train = activations_train[check_indices_train]
                
                activations_handler_train = ActivationsHandler(
                    activations=activations_train,
                    labels=labels_df_subset_train["correct"].astype(bool),
                )

                activations_handler_folds_train = list(
                    activations_handler_train.split_dataset(split_sizes=[1/n_folds] * n_folds)
                )

                if check_indices_test is None:
                    check_indices_test = indices_test.sample(frac=1, replace=False, random_state=42)
                
                if set(indices_test) != set(check_indices_test):
                    raise RuntimeError(f"indices across layers are not the same")

                labels_df_subset_test = labels_df_test.iloc[check_indices_test]
                activations_test = activations_test[check_indices_test]

                activations_handler_test = ActivationsHandler(
                    activations=activations_test,
                    labels=labels_df_subset_test["correct"].astype(bool),
                )

                activations_handler_folds_test = list(
                    activations_handler_test.split_dataset(split_sizes=[1/n_folds] * n_folds)
                )

                fold_stats = {}
                for fold_i in range(n_folds):
                    print(f"{fold_i=}", end=", ")
                    activations_handler_test = activations_handler_folds_test[fold_i]
                # for fold_i, activations_handler_test in enumerate(activations_handler_folds_train):
                    activations_handler_train = combine_activations_handlers(
                        [ah for j, ah in enumerate(activations_handler_folds_train) if j != fold_i]
                    )
                    
                    activations_handler_train, pca_info = activations_handler_train.reduce_dims(pca_components=PCA_COMPONENTS)
                    
                    activations_handler_train = activations_handler_train.sample_equally_across_groups(
                        group_labels=[False, True]
                    )
                    activations_handler_test, _ = activations_handler_test.reduce_dims(pca_components=PCA_COMPONENTS, pca_info=pca_info)
                    activations_handler_test = activations_handler_test.sample_equally_across_groups(
                        group_labels=[False, True]
                    )

                    res_dict["model_id_train"].append(model_id_train)
                    res_dict["dataset_id_train"].append(dataset_id_train)
                    res_dict["prompt_id_train"].append(prompt_id_train)
                    res_dict["subset_id_train"].append(subset_id_train)
                    res_dict["input_type_train"].append(input_type_train)
                    res_dict["model_id_test"].append(model_id_test)
                    res_dict["dataset_id_test"].append(dataset_id_test)
                    res_dict["prompt_id_test"].append(prompt_id_test)
                    res_dict["subset_id_test"].append(subset_id_test)
                    res_dict["input_type_test"].append(input_type_test)
                    res_dict["layer"].append(layer)
                    res_dict["fold"].append(fold_i)

                    for center_from_origin in [False, True]:
                        for classifier_cut_name,classifier_cut in [("optimal_train_set_cut", None), ("zero", 0.0)]:
                            direction_classifier, direction_calculator = get_correctness_direction_classifier(
                                activations_handler_train=activations_handler_train,
                                activations_handler_test=activations_handler_test,
                                center_from_origin=center_from_origin,
                                binary_classifier_kwargs={"classification_cut": classifier_cut},
                            )
                            for key, value in direction_classifier.classification_metrics.items():
                                res_dict[
                                    f"direction__center_from_origin_{center_from_origin}__classifier_cut_{classifier_cut_name}__{key}"
                                ].append(value)
                    
                    # can use any of the direction calculators as they should be the same
                    # res_dict["direction__train_set_classifying_direction"].append(direction_calculator.classifying_direction.tolist())

                    # for key, value in get_logistic_regression_classifier(
                    #         activations_handler_train=activations_handler_train,
                    #         activations_handler_test=activations_handler_test,
                    #     )[0].classification_metrics.items():
                    #     res_dict[f"logistic__regression_{key}"].append(value)



res_df = pd.DataFrame(res_dict)
res_df.to_csv(os.path.join("/runpod-volume/correctness-model-internals/", "notebooks", "best_layer_finding", MODEL_ID, "classification_data", "final_data.csv"), index=False)

# PLOTTING
for metric in [
    "direction__center_from_origin_False__classifier_cut_optimal_train_set_cut__test_roc_auc",
    "direction__center_from_origin_False__classifier_cut_optimal_train_set_cut__test_accuracy_score",
    "direction__center_from_origin_False__classifier_cut_zero__test_accuracy_score",
    "direction__center_from_origin_True__classifier_cut_optimal_train_set_cut__test_accuracy_score",
    "direction__center_from_origin_True__classifier_cut_zero__test_accuracy_score",
]:
    print(f"\n\n{metric=}")
    plot_dict = {}
    for conf, res_df_ in res_df.groupby([
        "model_id_train", 
        "dataset_id_train", 
        "prompt_id_train", 
        "subset_id_train", 
        "input_type_train",
        "model_id_test",
        "dataset_id_test",
        "prompt_id_test",
        "subset_id_test",
        "input_type_test"
    ]):
        # if conf[4] != "prompt_only":
        #     continue
        
        print(f"{conf=}")
        res_df_pivot = pd.pivot(
            res_df_.drop(columns=[
                "model_id_train",
                "dataset_id_train",
                "prompt_id_train",
                "subset_id_train",
                "input_type_train",
                "model_id_test",
                "dataset_id_test",
                "prompt_id_test",
                "subset_id_test",
                "input_type_test"
            ]),
            index='layer',
            columns='fold',
            # values=['direction_f1_score', 'logistic_regression_f1_score']  # add all metrics you want to keep
        )
        # for classifier in ["direction", "logistic_regression"]:
        #     for metric in ["f1_score", "accuracy_score", "precision_score", "recall_score"]:
        plot_dict[str(conf)] = res_df_pivot[[metric]]
    save_path = os.path.join("/runpod-volume/correctness-model-internals", "notebooks", "best_layer_finding", MODEL_ID, "classification_data", "figures", f"final_train_test_different_datasets_direction_{metric}.html")
    plot_interactive_lineplot(
        plot_dict,
        x_label="Layer",
        y_label=f"{metric}".replace("_", " ").title(),
        save_path=save_path
    )