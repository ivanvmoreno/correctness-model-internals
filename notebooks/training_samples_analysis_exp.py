import sys
from pathlib import Path
import os
import random

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # adjust this if needed
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
BASE_PATH = "/home/arnau/Desktop/MARS/correctness-model-internals/classification_data"
print(f"BASE_PATH: {BASE_PATH}")
# BASE_PATH = "../../"
PCA_COMPONENTS = None

MODEL_ID = "ministral_8b_instruct"
DATASET_ID = None
PROMPT_ID = None
SUBSET_ID = None
INPUT_TYPE = None
layers_to_keep = [18]
n_folds = 5

folds = 10
batch_size = 20
models_and_layers = {}
models_and_layers['ministral_8b_instruct'] = [18]
# models_and_layers['mistral_7b_instruct'] = [16]
# models_and_layers['qwen_2.5_7b_instruct'] = [22]
# models_and_layers['llama3.1_8b_chat'] = [14]
# models_and_layers['llama3.3_70b'] = [76]
# models_and_layers['deepseek_qwen_32b'] = [44]

models_and_savedsize = {}
models_and_savedsize['ministral_8b_instruct'] = 20
models_and_savedsize['mistral_7b_instruct'] = 20
models_and_savedsize['qwen_2.5_7b_instruct'] = 20
models_and_savedsize['llama3.1_8b_chat'] = 20
models_and_savedsize['llama3.3_70b'] = 10
models_and_savedsize['deepseek_qwen_32b'] = 5

for MODEL_ID, layers_to_keep in models_and_layers.items():
    print(f"Processing model: {MODEL_ID}")
    activation_exp_configs_df = get_experiment_activations_configs_df_subset(
        base_path=BASE_PATH,
        model_id=MODEL_ID,
        dataset_id=DATASET_ID,
        prompt_id=PROMPT_ID,
        subset_id=SUBSET_ID,
        input_type=INPUT_TYPE,
    )
    # Drop things that are not needed
    activation_exp_configs_df = activation_exp_configs_df[activation_exp_configs_df["dataset_id"].isin(
        ['trivia_qa_2_60k', 'cities_10k', 'birth_years_4k', 'medals_9k', 'math_operations_6k', 'gsm8k', 'notable_people']
        # ['trivia_qa_2_60k']
    )]
    activation_exp_configs_df = activation_exp_configs_df[activation_exp_configs_df["prompt_id"] != "cot_3_shot"]
    res_dict = defaultdict(list)

    # generate 10 initial positions
    initial_positions = [10000, 14020, 18000, 22000, 26000, 30000, 34040, 38000, 42000, 46000]


    for size in [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 48540]:
        print(f"Dataset size: {size}")
        for input_type in ["prompt_only"]:
            activation_exp_configs_df_ = activation_exp_configs_df[activation_exp_configs_df["input_type"] == input_type]
            for (model_id_train, dataset_id_train, prompt_id_train, subset_id_train, input_type_train), _ in activation_exp_configs_df_.groupby(["model_id", "dataset_id", "prompt_id", "subset_id", "input_type"]):
                    for (model_id_test, dataset_id_test, prompt_id_test, subset_id_test, input_type_test), config_df in activation_exp_configs_df_.groupby(["model_id", "dataset_id", "prompt_id", "subset_id", "input_type"]):
                        for fold in range(10):
                            if dataset_id_train != "trivia_qa_2_60k":
                                continue
                            # randomize initial position, needs to be a multiple of models_and_savedsize[MODEL_ID]
                            # initial_position = random.randint(10000, 45000-size) // models_and_savedsize[MODEL_ID] * models_and_savedsize[MODEL_ID]
                            # indices_train_selected = list(range(10000 + fold*size, 10000 + (fold + 1)*size))
                            # indices_train_selected = list(range(initial_positions[fold], initial_positions[fold] + size))
                            indices_train_selected = [10000 + (i + initial_positions[fold]) % 48540 for i in range(size)]
                            indices_train_selected.sort()

                            labels_df_train = load_labels(
                                base_path=BASE_PATH,
                                model_id=model_id_train,
                                dataset_id=dataset_id_train,
                                prompt_id=prompt_id_train,
                                subset_id=subset_id_train,
                                indices=indices_train_selected
                            )
                            labels_df_test = load_labels(
                                base_path=BASE_PATH,
                                model_id=model_id_test,
                                dataset_id=dataset_id_test,
                                prompt_id=prompt_id_test,
                                subset_id=subset_id_test,
                            )
                            for layer in config_df["layer"].astype(int).sort_values():
                                if len(layers_to_keep) and layer not in layers_to_keep:
                                    continue
                                activations_train, indices_train = load_activations(
                                    base_path=BASE_PATH,
                                    model_id=model_id_train,
                                    dataset_id=dataset_id_train,
                                    prompt_id=prompt_id_train,
                                    subset_id=subset_id_train,
                                    input_type=input_type_train,
                                    layer=layer,
                                    # batch_ids=[i for i in range(10000 + fold*size, 10000 + (fold + 1)*size, models_and_savedsize[MODEL_ID])],
                                    # batch_ids = [i for i in range(initial_position, initial_position + size, models_and_savedsize[MODEL_ID])],
                                    batch_ids = [10000 + (i + initial_positions[fold]) % 48540 for i in range(0, size, models_and_savedsize[MODEL_ID])],
                                )
                                
                                # print(indices_train_selected)
                                # print([10000 + (i + initial_positions[fold]) % 48540 for i in range(0, size, models_and_savedsize[MODEL_ID])])
                                # print(indices_train)

                                activations_test, indices_test = load_activations(
                                    base_path=BASE_PATH,
                                    model_id=model_id_test,
                                    dataset_id=dataset_id_test,
                                    prompt_id=prompt_id_test,
                                    subset_id=subset_id_test,
                                    input_type=input_type_test,
                                    layer=layer,
                                )
                                
                                # labels_df_subset_train = labels_df_train.iloc[indices_train_selected]
                                activations_handler_train = ActivationsHandler(
                                    activations=activations_train,
                                    labels=labels_df_train["correct"].astype(bool),
                                )
                                # labels_df_subset_test = labels_df_test.iloc[indices_test]
                                activations_handler_test = ActivationsHandler(
                                    activations=activations_test,
                                    labels=labels_df_test["correct"].astype(bool),
                                )

                                direction_classifier, direction_calculator = get_correctness_direction_classifier(
                                    activations_handler_train=activations_handler_train,
                                    activations_handler_test=activations_handler_test,
                                    center_from_origin=False,
                                    binary_classifier_kwargs={"classification_cut": 0.0},
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
                                res_dict["n_samples"].append(size)
                                res_dict["fold"].append(fold)
                                for key, value in direction_classifier.classification_metrics.items():
                                    res_dict[
                                        f"{key}"
                                    ].append(value)
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv(os.path.join("/home/arnau/Desktop/MARS/correctness-model-internals", "notebooks", "best_layer_finding", MODEL_ID, "classification_data", "final_data_samples_10folds_exp.csv"), index=False)

