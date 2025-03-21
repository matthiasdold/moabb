from __future__ import annotations

from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump
from typing import Sequence

import numpy as np
from numpy import argmax
from sklearn.pipeline import Pipeline


try:
    from optuna.distributions import CategoricalDistribution

    optuna_available = True
except ImportError:
    optuna_available = False


def _check_if_is_pytorch_model(model):
    """Check if the model is a skorch model.

    Parameters
    ----------
    model: object
        Model to check
    Returns
    -------
    is_pytorch_model: bool
        True if the model is a Skorch model
    """
    try:
        from skorch import NeuralNetClassifier

        is_pytorch_model = isinstance(model, NeuralNetClassifier)
        return is_pytorch_model
    except ImportError:
        return False


def _check_if_is_pytorch_steps(model):
    skorch_valid = False
    try:
        skorch_valid = any(
            _check_if_is_pytorch_model(j) for j in model.named_steps.values()
        )
        return skorch_valid
    except Exception:
        return skorch_valid


def save_model_cv(model: object, save_path: str | Path, cv_index: str | int):
    """Save a model fitted to a given fold from cross-validation.

    Parameters
    ----------
    model: object
        Model (pipeline) fitted
    save_path: str
        Path to save the model, will create if it does not exist
        based on the parameter hdf5_path from the evaluation object.
    cv_index: str
        Index of the cross-validation fold used to fit the model
        or 'best' if the model is the best fitted

    Returns
    -------
    """
    if save_path is None:
        raise IOError("No path to save the model")
    else:
        Path(save_path).mkdir(parents=True, exist_ok=True)

    if _check_if_is_pytorch_steps(model):
        for step_name in model.named_steps:
            step = model.named_steps[step_name]
            file_step = f"{step_name}_fitted_{cv_index}"

            if _check_if_is_pytorch_model(step):
                step.save_params(
                    f_params=Path(save_path) / f"{file_step}_model.pkl",
                    f_optimizer=Path(save_path) / f"{file_step}_optim.pkl",
                    f_history=Path(save_path) / f"{file_step}_history.json",
                    f_criterion=Path(save_path) / f"{file_step}_criterion.pkl",
                )
            else:
                with open((Path(save_path) / f"{file_step}.pkl"), "wb") as file:
                    dump(step, file, protocol=HIGHEST_PROTOCOL)
    else:
        with open((Path(save_path) / f"fitted_model_{cv_index}.pkl"), "wb") as file:
            dump(model, file, protocol=HIGHEST_PROTOCOL)


def save_model_list(model_list: list | Pipeline, score_list: Sequence, save_path: str):
    """Save a list of models fitted to a folder.

    Parameters
    ----------
    model_list: list | Pipeline
        List of models or model (pipelines) fitted
    score_list: Sequence
        List of scores for each model in model_list
    save_path: str
        Path to save the models, will create if it does not exist
        based on the parameter hdf5_path from the evaluation object.
    Returns
    -------
    """
    if model_list is None:
        return

    Path(save_path).mkdir(parents=True, exist_ok=True)

    if not isinstance(model_list, list):
        model_list = [model_list]

    for cv_index, model in enumerate(model_list):
        save_model_cv(model, save_path, str(cv_index))

    best_model = model_list[argmax(score_list)]

    save_model_cv(best_model, save_path, "best")


def create_save_path(
    hdf5_path,
    code: str,
    subject: int | str,
    session: str,
    name: str,
    grid=False,
    eval_type="WithinSession",
):
    """Create a save path based on evaluation parameters.

    Parameters
    ----------
    hdf5_path : str
       The base path where the models will be saved.
    code : str
       The code for the evaluation.
    subject : int
       The subject ID for the evaluation.
    session : str
       The session ID for the evaluation.
    name : str
       The name for the evaluation.
    grid : bool, optional
       Whether the evaluation is a grid search or not. Defaults to False.
    eval_type : str, optional
       The type of evaluation, either 'WithinSession', 'CrossSession' or 'CrossSubject'.
       Defaults to WithinSession.
    Returns
    -------
    path_save: str
       The created save path.
    """
    if hdf5_path is not None:
        if eval_type != "WithinSession":
            session = ""

        if grid:
            path_save = (
                Path(hdf5_path)
                / f"GridSearch_{eval_type}"
                / code
                / f"{str(subject)}"
                / str(session)
                / str(name)
            )
        else:
            path_save = (
                Path(hdf5_path)
                / f"Models_{eval_type}"
                / code
                / f"{str(subject)}"
                / str(session)
                / str(name)
            )

        return str(path_save)
    else:
        print("No hdf5_path provided, models will not be saved.")


def _convert_sklearn_params_to_optuna(param_grid: dict) -> dict:
    """
    Function to convert the parameter in Optuna format. This function will
    create a categorical distribution of values from the list of values
    provided in the parameter grid.

    Parameters
    ----------
    param_grid:
        Dictionary with the parameters to be converted.

    Returns
    -------
    optuna_params: dict
        Dictionary with the parameters converted to Optuna format.
    """
    if not optuna_available:
        raise ImportError(
            "Optuna is not available. Please install it optuna " "and optuna-integration."
        )
    else:
        optuna_params = {}
        for key, value in param_grid.items():
            try:
                if isinstance(value, list):
                    optuna_params[key] = CategoricalDistribution(value)
                else:
                    optuna_params[key] = value
            except Exception as e:
                raise ValueError(f"Conversion failed for parameter {key}: {e}")
        return optuna_params


class ChronoGroupsSplit:
    """Leave out blocks of chronologically sorted pairs.

    This class is designed to split data into training and testing sets while
    preserving the temporal order of the data. It ensures that the training set
    includes a full group for each unique label, and the test set includes the
    remaining groups. This is particularly useful for time-series data or
    experiments with multiple blocks of trials.

    Attributes
    ----------
    None

    Methods
    -------
    split(X, y, groups)
        Return the indices of all splits of the data in X and y sorted according
        to a grouping variable.
    """

    def __init__(self, **kwargs):
        for k in kwargs:
            print(f"Received kwargs for {k=} which will not be used")

    def split(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Return the indices of all splits of the data in X and y sorted according
        to a grouping variable.

        This method is used to preserve the temporal order of the data when
        splitting into train and test sets. It ensures that the training set
        includes a full group for each unique label, and the test set includes
        the remaining groups. This is particularly useful for time-series data
        or experiments with multiple blocks of trials.

        Parameters
        ----------
        X : np.ndarray (nsamples, nfeatures)
            The data array with the sample dimension first.
        y : np.ndarray (nsamples, )
            The labels vector.
        groups : np.ndarray (nsamples, )
            A vector associating a group with each label. Splits will sort the
            available group elements per unique label and then ensure that the
            training set includes a full group for each unique label. By
            modifying the group labels, one can control how many samples are
            included in the according train/test sets.

        Returns
        -------
        splits : list[tuple[np.ndarray, np.ndarray]]
            A list of splits of (ix_train, ix_test) tuples to loop over.

        Examples
        --------
        Consider data from an experiment with 4 blocks and 2 trials for each
        block. The split keeps adjacent blocks (encoded in the `groups` variable).

        >>> import numpy as np
        >>> from moabb.evaluations.utils import ChronoGroupsSplit
        >>> X = np.random.randn(8, 2)
        >>> y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        >>> groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        >>> splitter = ChronoGroupsSplit()
        >>> splits = splitter.split(X, y, groups)
        >>> for train_index, test_index in splits:
        ...     print("TRAIN:", train_index, "TEST:", test_index)

            TRAIN: [4 5 6 7] TEST: [0 1 2 3]   # test idxs are: group 1 and 2, according to labels 0, 1
            TRAIN: [0 1 2 3] TEST: [4 5 6 7]   # test idxs are: group 3 and 4, according to labels 0, 1
        """
        y = np.asarray(y)
        groups = np.asarray(groups)

        # get groups per label
        gm = {k: list(set(groups[y == k])) for k in set(y)}

        # sort the group labels to match
        for v in gm.values():
            v.sort()

        # ensure that groups are distinct in labels
        assert all(
            [set(s).intersection(groups[y != k]) == set() for k, s in gm.items()]
        ), " Groups are not unique in labels. Please ensure uniqueness."

        # ensure lengths match
        set_lens = [len(v) for v in gm.values()]
        if not all([e == set_lens[0] for e in set_lens]):
            print(
                "Groups per label do not align along all unique label values (y)"
                "- will zip and thus drop all groups longer than the smallest "
                "set."
            )

        Xidcs = np.arange(X.shape[0])

        splits = [
            (
                np.hstack(
                    [
                        Xidcs[groups == list(gv)[j]]
                        for gv in gm.values()
                        for j in range(min(set_lens))
                        if j != i
                    ]
                ),  # the non selected --> train set
                np.hstack(
                    [Xidcs[groups == list(gv)[i]] for gv in gm.values()]
                ),  # the selected per grp --> test set
            )
            for i in range(min(set_lens))
        ]

        return splits
