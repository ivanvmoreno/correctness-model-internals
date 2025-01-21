import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler

from classifying.activations_handler import ActivationsHandler
from classifying.direction_calculator import DirectionCalculator


class BinaryClassifier:
    def __init__(
        self,
        train_labels,
        train_classification_score,
        test_labels,
        test_classification_score,
        classification_metric_funcs=(accuracy_score, f1_score),
        classification_cut=None,  # set this if you know the cut e.g. for LR want 0.5 don't want to get this from train set
    ):
        """
        Build a classifier by slicing along a continuous variable on the train set
        """
        if not all(
            isinstance(train_label, bool) for train_label in train_labels
        ) or not all(isinstance(test_label, bool) for test_label in test_labels):
            raise TypeError("Labels must be boolean")

        self.train_labels = train_labels
        self.train_classification_score = train_classification_score
        self.test_labels = test_labels
        self.test_classification_score = test_classification_score

        (
            self.train_false_positive_rate,
            self.train_true_positive_rate,
            self.train_thresholds,
        ) = roc_curve(self.train_labels, self.train_classification_score)
        self.test_false_positive_rate, self.test_true_positive_rate, _ = roc_curve(
            self.test_labels, self.test_classification_score
        )
        self.roc_auc = float(
            auc(self.test_false_positive_rate, self.test_true_positive_rate)
        )
        self.classification_metric_funcs = classification_metric_funcs

        self.optimal_cut = (
            classification_cut
            if classification_cut is not None
            else self.optimal_train_set_cut
        )
        self.test_pred_class = self.test_classification_score >= (self.optimal_cut)

        self.classification_metrics = {
            "optimal_cut": self.optimal_cut,
            "optimal_train_set_cut": self.optimal_train_set_cut,
            "roc_auc": float(self.roc_auc),
        }
        for classification_metric in self.classification_metric_funcs:
            self.classification_metrics[classification_metric.__name__] = float(
                classification_metric(self.test_labels, self.test_pred_class)
            )

    @property
    def optimal_train_set_cut(self):
        youden_index = self.train_true_positive_rate - self.train_false_positive_rate
        optimal_idx = np.argmax(youden_index[1:]) + 1
        return float(self.train_thresholds[optimal_idx])


def get_correctness_direction_classifier(
    activations_handler_train: ActivationsHandler,
    activations_handler_test: ActivationsHandler,
):
    direction_calculator = DirectionCalculator(
        activations_handler=activations_handler_train,
        from_group=False,
        to_group=True,
    )
    direction_classifier = BinaryClassifier(
        train_labels=activations_handler_train.labels,
        train_classification_score=direction_calculator.get_distance_along_classifying_direction(
            activations_handler_train.activations
        ),
        test_labels=activations_handler_test.labels,
        test_classification_score=direction_calculator.get_distance_along_classifying_direction(
            activations_handler_test.activations
        ),
    )
    return direction_classifier, direction_calculator


def get_logistic_regression_classifier(
    activations_handler_train: ActivationsHandler,
    activations_handler_test: ActivationsHandler,
    classification_cut=0.5,
):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(activations_handler_train.activations)
    X_test = scaler.transform(activations_handler_test.activations)

    model = LogisticRegression(
        random_state=42, solver="lbfgs", max_iter=1000, class_weight="balanced"
    )
    model.fit(X_train, activations_handler_train.labels)

    logistic_regression_classifier = BinaryClassifier(
        train_labels=activations_handler_train.labels,
        train_classification_score=model.predict_proba(X_train)[:, 1],
        test_labels=activations_handler_test.labels,
        test_classification_score=model.predict_proba(X_test)[:, 1],
        classification_cut=classification_cut,
    )
    return logistic_regression_classifier, model
