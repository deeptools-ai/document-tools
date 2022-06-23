# coding=utf-8
#
# Copyright The deeptools.ai team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""tokenize.py allows to automatically tokenize any dataset to prepare it for the training of a target model."""
import copy
import logging
from typing import Any, Dict, Optional, Union

from datasets import ClassLabel, Dataset, DatasetDict

from .encoders import TARGET_MODELS
from .utils import _get_label_list

logger = logging.getLogger(__name__)


def tokenize_dataset(
    dataset: Union[Dataset, DatasetDict],
    target_model: str = None,
    image_column: Optional[str] = "image",
    label_column: Optional[str] = "label",
    batched: Optional[bool] = True,
    batch_size: Optional[int] = 2,
    cache_file_names: Optional[Dict[str, str]] = None,
    keep_in_memory: Optional[bool] = False,
    num_proc: Optional[int] = None,
    processor_config: Optional[Dict[str, Any]] = None,
    save_to_disk: Optional[bool] = False,
    save_path: Optional[str] = None,
) -> DatasetDict:
    """
    Tokenize a dataset using a target model and return a new dataset with the encoded features and labels.

    Parameters
    ----------
    dataset : Dataset or DatasetDict, required
        Dataset to be tokenized.
    target_model : str, optional (default=None)
        Target model to use for tokenization.
    image_column : str, optional (default="image")
        Name of the column containing the image.
    label_column : str, optional (default="label")
        Name of the column containing the label.
    batched : bool, optional (default=True)
        Whether to use batched encoding.
    batch_size : int, optional (default=2)
        Batch size for batched encoding.
    cache_file_names : Dict[str, str], optional (default=None)
        Dictionary containing the cache file names for each target model.
    keep_in_memory : bool, optional (default=False)
        Whether to keep the dataset in memory.
    num_proc : int, optional (default=None)
        Number of processes to use for batched encoding.
    processor_config : Dict[str, Any], optional (default=None)
        Configuration for the processor of the target model.
    save_to_disk : bool, optional (default=False)
        Whether to save the dataset to disk or not.
    save_path : str, optional (default=None)
        Path to save the dataset to disk if `save_to_disk` is True.

    Returns
    -------
    DatasetDict
        Dataset with the encoded features and labels.

    Raises
    ------
    ValueError
        If there is no target model for the dataset. Or if saving to disk is requested but the save path is not
        provided.
    KeyError
        If the target model is not supported.
    TypeError
        If the dataset is not a Dataset or DatasetDict.
    """
    if not target_model:
        raise ValueError("""You need to specify the target architecture you want to use to tokenize your dataset.""")
    else:
        try:
            TARGET_MODELS[target_model]
        except KeyError:
            raise KeyError(
                f"""
                You specified a `target_model` that is not supported. Available models: {list(TARGET_MODELS.keys())}
                If you think that new model should be available, please feel free to open a new issue on the project
                repository: https://github.com/deeptools-ai/document-tools/issues
            """
            )

    if save_to_disk and save_path is None:
        raise ValueError(
            """
            You need to specify a path to save the dataset, because you chose to save it to disk. You can disable saving
            to disk by setting `save_to_disk=False`.
        """
        )
    elif not save_to_disk and save_path is not None:
        logger.warning(
            """
            You have indicated a path to save the dataset, but have chosen not to save it to disk. You need to add
            `save_to_disk=True` to the call to `tokenize_dataset` to save the dataset to disk.
        """
        )
    else:
        logger.info(
            """
        The dataset will not be saved to disk. If you want to save it to disk, add `save_to_disk=True` to the call to
        `tokenize_dataset`.
        """
        )

    dataset = copy.deepcopy(dataset)
    if isinstance(dataset, DatasetDict):
        tmp_dataset = dataset
        dataset_first_key = list(tmp_dataset.keys())[0]
    elif isinstance(dataset, Dataset):
        tmp_dataset = DatasetDict()
        dataset_first_key = "train"
        tmp_dataset[dataset_first_key] = dataset
    else:
        raise TypeError(f"The dataset has to be either a `Dataset` or a `DatasetDict`. You provided: {type(dataset)}")

    if isinstance(tmp_dataset[dataset_first_key].features[label_column].feature, ClassLabel):
        labels = tmp_dataset[dataset_first_key].features[label_column].feature.names
    else:
        labels = _get_label_list(tmp_dataset[dataset_first_key][label_column])

    encoder = TARGET_MODELS[target_model](config=processor_config, labels=labels)
    features = encoder.features

    encoded_dataset = tmp_dataset.map(
        encoder,
        features=features,
        remove_columns=[image_column, label_column],
        batched=batched,
        batch_size=batch_size,
        cache_file_names=cache_file_names,
        keep_in_memory=keep_in_memory,
        num_proc=num_proc,
    )

    if save_to_disk:
        try:
            encoded_dataset.save(save_path)
        except Exception as e:
            logger.error(e)

    return encoded_dataset
