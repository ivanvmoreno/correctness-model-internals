from datetime import datetime
from pathlib import Path

import pandas as pd
import torch as pt

from src.classifying import (
    ActivationsHandler,
    combine_activations_handlers,
    get_correctness_direction_classifier,
    get_logistic_regression_classifier,
)


def load_activations(
    model_id,
    dataset_id,
    prompt_id,
    subset_id,
    input_type,
    layer,
    batch_ids=None,
):
    if batch_ids:
        batch_ids = [int(batch_id) for batch_id in batch_ids]

    paths = sorted(
        list(
            Path(
                f"./activations/{model_id}/{dataset_id}/{prompt_id}/{subset_id}/{input_type}"
            ).iterdir()
        ),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    activations_list, indices = [], []
    batch_size = None
    for batch_file in paths:
        batch_id = int(batch_file.stem.split("_")[-1])
        if batch_ids and batch_id not in batch_ids:
            continue

        activations = pt.load(batch_file, map_location=pt.device("cpu"))[layer]
        activations_list.append(activations)

        batch_size = activations.shape[0]

        if batch_size is None:
            batch_size = activations.shape[0]
        else:
            assert batch_size == activations.shape[0]

        indices.append(
            pd.Series(range(batch_size), name="index") + batch_id * batch_size
        )
    return (
        pt.cat(activations_list, dim=0),
        pd.concat(indices).reset_index(drop=True),
    )


def load_labels(model_id, dataset_id, prompt_id, subset_id, indices=None):
    paths = list(
        Path(f"./evaluations/{model_id}/{dataset_id}/{prompt_id}/").iterdir()
    )
    for path in paths:
        filename = path.stem
        if subset_id != filename.split("_generations_evaluated")[0]:
            continue
        df = pd.read_csv(path)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)
        return df
    raise ValueError(
        f"No labels found for {model_id} {dataset_id} {prompt_id} {subset_id}"
    )


def classifier_experiment_run(
    run_configs,
    check_correctness_direction_classifier: bool = True,
    check_logistic_regression_classifier: bool = True,
    sample_equally=True,
    notes: str = "",
) -> None:

    overall_stats_dict = {
        "check_correctness_classifier": check_correctness_direction_classifier,
        "check_logistic_regression_classifier": check_logistic_regression_classifier,
        "timestamp": str(datetime.now()),
        "notes": notes,
        "results": [],
    }

    for run_config in run_configs:
        activations, indices = load_activations(
            model_id=run_config["model_id"],
            dataset_id=run_config["dataset_id"],
            prompt_id=run_config["prompt_id"],
            subset_id=run_config["subset_id"],
            input_type=run_config["input_type"],
            layer=run_config["layer"],
        )
        labels_df = load_labels(
            model_id=run_config["model_id"],
            dataset_id=run_config["dataset_id"],
            prompt_id=run_config["prompt_id"],
            subset_id=run_config["subset_id"],
            indices=indices,
        )

        activations = pt.cat([activations] * 100)  # TODO DELETEME!!!!!
        labels_df = pd.concat([labels_df] * 100)  # TODO DELETEME!!!!!
        print(f"\n\n\n{activations.shape}\n\n\n")

        if (
            not check_correctness_direction_classifier
            and not check_logistic_regression_classifier
        ):
            raise ValueError(
                "At least one of check_correctness_classifier or check_logistic_regression_classifier must be True"
            )

        activation_handler = ActivationsHandler(
            activations=activations, labels=labels_df["correct"].astype(bool)
        )
        if sample_equally:
            activation_handler = (
                activation_handler.sample_equally_across_groups(
                    group_labels=[False, True]
                )
            )

        print(f"\n\n\n{activation_handler.labels.value_counts()}\n\n\n")
        print(f"\n\n\n{activation_handler.activations.shape}\n\n\n")

        activations_handler_folds = list(
            activation_handler.split_dataset(split_sizes=[0.2] * 5)
        )

        fold_stats = {}
        for i, activations_handler_test in enumerate(activations_handler_folds):
            activations_handler_train = combine_activations_handlers(
                [ah for j, ah in enumerate(activations_handler_folds) if j != i]
            )
            stats_dict = {
                "n_train": activations_handler_train.activations.shape[0],
                "n_test": activations_handler_test.activations.shape[0],
            }

            if check_correctness_direction_classifier:
                direction_classifier, direction_calculator = (
                    get_correctness_direction_classifier(
                        activations_handler_train=activations_handler_train,
                        activations_handler_test=activations_handler_test,
                    )
                )
                stats_dict["correctness_direction_classifier"] = (
                    direction_classifier.classification_metrics
                )
                stats_dict["activation_space_directions"] = {
                    name: getattr(direction_calculator, name).tolist()
                    for name in [
                        "classifying_direction",
                        "mean_activations",
                        "centroid_from",
                        "centroid_to",
                        "max_activations_from",
                        "min_activations_from",
                        "max_activations_to",
                        "min_activations_to",
                    ]
                }

            if check_logistic_regression_classifier:
                stats_dict["logistic_regression_classifier"] = (
                    get_logistic_regression_classifier(
                        activations_handler_train=activations_handler_train,
                        activations_handler_test=activations_handler_test,
                    )[0].classification_metrics
                )
            fold_stats[f"fold_{i}"] = stats_dict

        overall_stats_dict["results"].append(
            {
                **run_config,
                **fold_stats,
            }
        )
    return overall_stats_dict


if __name__ == "__main__":
    print("Run this from the command line at the base level of the repo")

    # args_parser = argparse.ArgumentParser()
    # # args_parser.add_argument("--config", dest="config", required=True) # is this needed?
    # args_parser.add_argument("--model", dest="model", required=True)
    # args_parser.add_argument("--layer", dest="layer", required=True)
    # args_parser.add_argument(
    #     "--activations-dir", dest="activations_dir", required=True
    # )
    # args_parser.add_argument(
    #     "--question-answer-file", dest="question_answer_file", required=True
    # )
    # args_parser.add_argument("--first-batch", dest="first_batch", default=None)
    # args_parser.add_argument("--n-batches", dest="n_batches", default=None)
    # args_parser.add_argument("--notes", dest="notes", default=None)
    # args_parser.add_argument("--random-seed", dest="random_seed", default=42)
    # args_parser.add_argument("--train-frac", dest="train_frac", default=0.8)
    # args = args_parser.parse_args()

    # np.random.seed(args.random_seed)

    # classifier_experiment_run(
    #     model=args.model,
    #     layer=args.layer,
    #     activations_dir=args.activations_dir,
    #     question_answer_file=args.question_answer_file,
    #     first_batch=int(args.first_batch) if args.first_batch is not None else None,
    #     n_batches=int(args.n_batches) if args.first_batch is not None else None,
    #     train_frac=args.train_frac,
    #     notes=args.notes,
    # )

    res = classifier_experiment_run(
        run_configs=[
            {
                "model_id": "llama3_3b_chat",
                "dataset_id": "gsm8k",
                "prompt_id": "base_3_shot",
                "subset_id": "main",
                "input_type": "prompt_answer",
                "layer": 1,
            },
        ],
        check_correctness_direction_classifier=True,
        check_logistic_regression_classifier=True,
        sample_equally=True,
    )
    print(res)
    # with Path("./classification_outputs.json").open("w") as f:
    print("\n\n\nFinished.")
