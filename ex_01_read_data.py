import numpy as np
import pandas as pd
from pathlib import Path

from numpy.lib._stride_tricks_impl import sliding_window_view


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load and preprocess data from a CSV file. Remove rows with unlabeled data.

    Args:
        data_path (Path): Path to the CSV data file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with unlabeled data removed.
    Raises:
        FileNotFoundError: If the specified data file does not exist.
        ValueError: If the data is empty after removing unlabeled data and dropping NaN values.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file {data_path} not found.")

    data = pd.read_csv(data_path)
    data = remove_unlabeled_data(data)
    data = data.dropna()

    if data.empty:
        raise ValueError("Data is empty after removing unlabeled data and NaN values.")

    return data


def remove_unlabeled_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with unlabeled data (where labels == -1).

    Args:
        data (pd.DataFrame): Input DataFrame containing a 'labels' column.

    Returns:
        pd.DataFrame: DataFrame with unlabeled data removed.
    """
    if "labels" not in data.columns:
        raise KeyError("DataFrame must contain a 'labels' column.")

    return data[data["labels"] != -1]


def convert_to_np(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert DataFrame to numpy arrays, separating labels, experiment IDs, and features.

    Args:
        data (pd.DataFrame): Input DataFrame containing 'labels', 'exp_ids', and feature columns.

    Returns:
        tuple: A tuple containing:
            - labels (np.ndarray): Array of labels
            - exp_ids (np.ndarray): Array of experiment IDs
            - data (np.ndarray): Combined array of current and voltage features
    """
    if "labels" not in data.columns or "exp_ids" not in data.columns:
        raise ValueError("DataFrame must contain 'labels' and 'exp_ids' columns.")

    labels = data["labels"].to_numpy()
    exp_ids = data["exp_ids"].to_numpy()

    current = data.filter(like="I").to_numpy()
    voltage = data.filter(like="V").to_numpy()
    features = np.stack((current, voltage), axis=-1)

    if not np.issubdtype(features.dtype, np.number):
        raise ValueError("Current and voltage data must be numeric.")

    return labels, exp_ids, features


def create_sliding_windows_first_dim(
    data: np.ndarray, sequence_length: int
) -> np.ndarray:
    """
    Create sliding windows over the first dimension of a 3D array.

    Args:
        data (np.ndarray): Input array of shape (n_samples, timesteps, features)
        sequence_length (int): Length of each window

    Returns:
        np.ndarray: Windowed data of shape (n_windows, sequence_length*timesteps, features)
    """
    n_samples, timesteps, features = data.shape

    if sequence_length > n_samples:
        raise ValueError("Sequence length can't be greater than the number of samples.")

    windows = sliding_window_view(
        data, (sequence_length, timesteps, features), axis=(0, 1, 2)
    )
    windows = windows.reshape(windows.shape[0], sequence_length * timesteps, features)

    return windows


def get_welding_data(
    path: Path,
    n_samples: int | None = None,
    return_sequences: bool = False,
    sequence_length: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load welding data from CSV or cached numpy files.

    If numpy cache files don't exist, loads from CSV and creates cache files.
    If cache files exist, loads directly from them.

    Args:
        path (Path): Path to the CSV data file.
        n_samples (int | None): Number of samples to sample from the data. If None, all data is returned.
        return_sequences (bool): If True, return sequences of length sequence_length.
        sequence_length (int): Length of sequences to return.
    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of welding data features
            - np.ndarray: Array of labels
            - np.ndarray: Array of experiment IDs
    """
    cache_path = path.with_suffix(".npy")

    if not cache_path.exists():
        data = load_data(path)
        labels, exp_ids, features = convert_to_np(data)
        np.savez(cache_path, labels=labels, exp_ids=exp_ids, features=features)
    else:
        with np.load(cache_path) as data:
            labels = data["labels"]
            exp_ids = data["exp_ids"]
            features = data["features"]

    if n_samples is not None:
        indices = np.random.choice(len(labels), n_samples, replace=False)
        labels = labels[indices]
        exp_ids = exp_ids[indices]
        features = features[indices]

    if return_sequences:
        features = create_sliding_windows_first_dim(features, sequence_length)
        labels = np.array(
            [
                labels[i : i + sequence_length]
                for i in range(len(labels) - sequence_length + 1)
            ]
        )
        exp_ids = np.array(
            [
                exp_ids[i : i + sequence_length]
                for i in range(len(exp_ids) - sequence_length + 1)
            ]
        )

    return features, labels, exp_ids
