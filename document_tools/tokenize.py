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
import logging
from typing import Optional, Union

from datasets import Array2D, Array3D, ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from transformers import LayoutLMTokenizer, LayoutLMv2Processor, LayoutLMv3Processor, LayoutXLMProcessor

from .encode_functions import encode_layoutlm, encode_layoutlmv2, encode_layoutlmv3, encode_layoutxlm

logger = logging.getLogger(__name__)

TARGET_MODELS = {
    "layoutlm": {
        "preprocessor": LayoutLMTokenizer,
        "default_model": "microsoft/layoutlm-base-uncased",
        "encode_function": encode_layoutlm(),
    },
    "layoutlmv2": {
        "preprocessor": LayoutLMv2Processor,
        "default_model": "microsoft/layoutlmv2-base-uncased",
        "encode_function": encode_layoutlmv2(),
        "features": Features(
            {
                'image': Array3D(dtype="int64", shape=(3, 224, 224)),
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'attention_mask': Sequence(Value(dtype='int64')),
                'token_type_ids': Sequence(Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
            }
        ),
    },
    "layoutlmv3": {
        "preprocessor": LayoutLMv3Processor,
        "default_model": "microsoft/layoutlmv3-base-uncased",
        "encode_function": encode_layoutlmv3(),
    },
    "layoutxlm": {
        "preprocessor": LayoutXLMProcessor,
        "default_model": "microsoft/layoutxlm-base-uncased",
        "encode_function": encode_layoutxlm(),
    },
}


def tokenize_dataset(
    dataset: Union[Dataset, DatasetDict],
    target_model: str = None,
    image_column: Optional[str] = "image",
    label_colum: Optional[str] = "label",
    batched: Optional[bool] = True,
    batch_size: Optional[int] = 2,
    cache_file_name: Optional[str] = None,
    keep_in_memory: Optional[bool] = False,
    num_proc: Optional[int] = None,
    preprocessor_config: Optional[Union[str, dict]] = None,
    save_to_disk: Optional[bool] = False,
    save_path: Optional[str] = None,
) -> Union[Dataset, DatasetDict]:
    """"""
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

    if isinstance(dataset, Dataset):
        dataset_is_dict = False
        labels = dataset.features[label_colum].names
    elif isinstance(dataset, DatasetDict):
        dataset_is_dict = True
        dict_keys = list(dataset.keys())
        labels = dataset[dict_keys[0]].features[label_colum].names

    target_model_config = TARGET_MODELS[target_model]  # type: ignore

    features = target_model_config["features"]
    features["labels"] = ClassLabel(num_classes=len(labels), names=labels)

    encode_fct = target_model_config["encode_function"]

    if dataset_is_dict:
        encoded_dataset = DatasetDict()
        for key in dict_keys:
            encoded_dataset[key] = dataset.map(
                encode_fct,
                features=features,
                remove_columns=[image_column, label_colum],
                batched=batched,
                batch_size=batch_size,
                cache_file_name=cache_file_name,
                keep_in_memory=keep_in_memory,
                num_proc=num_proc,
            )
    else:
        encoded_dataset = dataset.map(
            encode_fct,
            features=features,
            remove_columns=[image_column, label_colum],
            batched=batched,
            batch_size=batch_size,
            cache_file_name=cache_file_name,
            keep_in_memory=keep_in_memory,
            num_proc=num_proc,
        )

    if save_to_disk:
        encoded_dataset.save(save_path)

    return encoded_dataset
