from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from evaluate import load
from pandas import read_csv, Series
from pynvml import *


def get_device():
    """
    Get the current torch device.

    :return: The current torch device.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device


def print_gpu_utilization():
    nvmlInit()

    handle = nvmlDeviceGetHandleByIndex(0)

    info = nvmlDeviceGetMemoryInfo(handle)

    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")

    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")

    print_gpu_utilization()


def set_seed_in_everything(seed_int: int = 42):
    torch.manual_seed(seed_int)
    if get_device() == 'cuda':
        torch.cuda.manual_seed(seed_int)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed_int)


class DataImporter:

    def __init__(self, hf_name: str | None) -> None:
        """
        Initialize an instance of AbstractRawDataImporter.

        Args:
             hf_name (str | None): Name of the HuggingFace dataset
        """
        self._hf_name = hf_name
        self._raw_data = None

    @property
    def raw_data(self) -> pd.DataFrame | None:
        """
        Property for original dataset in a table format.

        Returns:
            pandas.DataFrame | None: A dataset in a table format
        """
        return self._raw_data

    def obtain(self) -> None:
        """
        Import dataset.
        """
        dataset = load_dataset(self._hf_name,
                               'wikitext-2-raw-v1',
                               split='train').to_pandas()

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError()
        self._raw_data = dataset


class DataPreprocessor:

    def __init__(self, raw_data: pd.DataFrame) -> None:
        """
        Initialize an instance of AbstractRawDataPreprocessor.

        Args:
            raw_data (pandas.DataFrame): Original dataset in a table format
        """
        self._raw_data = raw_data
        self._data = None

    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = self._raw_data
        self._data.dropna(inplace=True, subset=['text'])
        self._data.reset_index(inplace=True)
        self._data = self.data.drop(self.data[self.data['text'] == ''].index)
        self._data = self._data.head(1000)
        self._data['start_text'] = self._data['text'].str[:-10]
        self._data['target'] = self._data['text'].str[-10:]
        self._data = self._data[['start_text', 'target']]

    @property
    def data(self) -> pd.DataFrame | None:
        """
        Property for preprocessed dataset.

        Returns:
            pandas.DataFrame | None: Preprocessed dataset in a table format
        """
        return self._data


class TrainDataset(Dataset):
    """
    Custom dataset for loading and preprocessing translation data.
    """

    """
    Dataset with translation data.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): original data.
        """
        self._data = data

    @property
    def data(self) -> pd.DataFrame:
        """
        Property with access to preprocessed DataFrame.
        """
        return self._data

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self._data)

    def __getitem__(self, index: int) -> str:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """
        row = self._data.iloc[index]
        return (
            str(row['start_text'])
        )


class Evaluator:
    """
    A class that compares prediction quality using the specified metric.
    """

    def __init__(self, data_path: Path) -> None:
        """
        Initialize an instance of Evaluator.

        Args:
            data_path (Path): Path to predictions
        """
        self._data_path = data_path
        self._rouge_metric = load('rouge', seed=42)
        self._bleu_metric = load('bleu', seed=42)

    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        preds = read_csv(self._data_path)
        preds.reset_index(inplace=True)

        kwargs: dict[str, Series | str] = {
            'predictions': preds['prediction'],
            'references': preds['target']
        }

        rouge_res = self._rouge_metric.compute(**kwargs).get('rougeL')
        bleu_res = self._bleu_metric.compute(**kwargs).get('bleu')
        result = {'bleu': bleu_res, 'rouge': rouge_res}
        return result
