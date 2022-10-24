from typing import List, Dict
import os
import glob

import pandas as pd
from sklearn.model_selection import train_test_split


def merge_datasets(dataset_dirs: List[str]) -> pd.DataFrame:
    dataset = pd.DataFrame({'image_path': [], 'label': [], 'source': []})
    for dataset_dir in dataset_dirs:
        labels = glob.glob(os.path.join(dataset_dir, '*'))
        for label in labels:
            images = list(glob.glob(os.path.join(label, '*')))
            dataset = pd.concat(
                [
                    dataset,
                    pd.DataFrame({
                        'image_path': images,
                        'label': [os.path.basename(label)] * len(images),
                        'source': [dataset_dir] * len(images)
                    })
                ],
                ignore_index=True
            )
    return dataset


def smart_train_test_split(dataset: pd.DataFrame, *args, **kwargs) -> Dict[str, pd.DataFrame]:
    # TODO
    return {
        'train': pd.DataFrame(),
        'test': pd.DataFrame()
    }


def default_train_test_split(dataset: pd.DataFrame, *args, **kwargs) -> Dict[str, pd.DataFrame]:
    train, test = train_test_split(dataset, *args, **kwargs)
    return {
        'train': train.reset_index(),
        'test': test.reset_index()
    }
