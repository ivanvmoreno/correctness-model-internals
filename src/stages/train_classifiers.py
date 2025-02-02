from src.classifying import (
    ActivationsHandler,
    get_correctness_direction_classifier,
    get_logistic_regression_classifier,
)


def classifier_experiment_run(
    run_config: list[tuple[str, str, str]],
    check_correctness_direction_classifier: bool = True,
    check_logistic_regression_classifier: bool = True,
    train_frac=0.8,
    sample_equally=True,
) -> None:
    train_frac = float(train_frac)
    for model_name, dataset_name, layer in run_config:
        labels_df, activations = model_data_loaders[
            (model_name, dataset_name)
        ]()
        if (
            not check_correctness_direction_classifier
            and not check_logistic_regression_classifier
        ):
            raise ValueError(
                "At least one of check_correctness_classifier or check_logistic_regression_classifier must be True"
            )

        activation_handler = ActivationsHandler(
            activations=activations, labels=labels_df["correct"]
        )
        if sample_equally:
            activation_handler = (
                activation_handler.sample_equally_across_groups(
                    group_labels=[False, True]
                )
            )

        activations_handler_train, activations_handler_test = (
            activation_handler.split_dataset(splits=[0.8, 0.2])
        )

        stats_dict = {
            "n_train": activations_handler_train.activations.shape[0],
            "n_test": activations_handler_test.activations.shape[0],
        }

        if check_correctness_direction_classifier:
            stats_dict["correctness_direction_classifier"] = (
                get_correctness_direction_classifier(
                    activations_handler_train=activations_handler_train,
                    activations_handler_test=activations_handler_test,
                )[0].classification_metrics
            )

        if check_logistic_regression_classifier:
            stats_dict["logistic_regression_classifier"] = (
                get_logistic_regression_classifier(
                    activations_handler_train=activations_handler_train,
                    activations_handler_test=activations_handler_test,
                )[0].classification_metrics
            )

        overall_stats_dict[(model_name, dataset_name, f"layer_{layer}")] = (
            stats_dict
        )
    return stats_dict


# todo update this so it loads from config
# if __name__ == "__main__":
#     print("Run this from the command line at the base level of the repo")

#     args_parser = argparse.ArgumentParser()
#     # args_parser.add_argument("--config", dest="config", required=True) # is this needed?
#     args_parser.add_argument("--model", dest="model", required=True)
#     args_parser.add_argument("--layer", dest="layer", required=True)
#     args_parser.add_argument("--activations-dir", dest="activations_dir", required=True)
#     args_parser.add_argument(
#         "--question-answer-file", dest="question_answer_file", required=True
#     )
#     args_parser.add_argument("--first-batch", dest="first_batch", default=None)
#     args_parser.add_argument("--n-batches", dest="n_batches", default=None)
#     args_parser.add_argument("--notes", dest="notes", default=None)
#     args_parser.add_argument("--random-seed", dest="random_seed", default=42)
#     args_parser.add_argument("--train-frac", dest="train_frac", default=0.8)
#     args = args_parser.parse_args()

#     np.random.seed(args.random_seed)

#     classifier_experiment_run(
#         model=args.model,
#         layer=args.layer,
#         activations_dir=args.activations_dir,
#         question_answer_file=args.question_answer_file,
#         first_batch=int(args.first_batch) if args.first_batch is not None else None,
#         n_batches=int(args.n_batches) if args.first_batch is not None else None,
#         train_frac=args.train_frac,
#         notes=args.notes,
#     )
#     print("\n\n\nFinished.")
