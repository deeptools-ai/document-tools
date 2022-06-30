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
"""encode_functions.py defines all encoding functions used to tokenize a dataset."""
import logging
from typing import Any, Dict, List, Optional

from datasets import Array2D, Array3D, ClassLabel, Features, Sequence, Value
from transformers import LayoutLMv2Processor, LayoutLMv3Processor, LayoutXLMProcessor

logger = logging.getLogger(__name__)


class BaseEncoder:
    """BaseEncoder is the base class for all encoders."""

    def __init__(self, labels: List[Any], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the encoder.

        Parameters
        ----------
        config : Dict[str, Any], optional (default=None)
            Configuration for the encoder.
        labels : List[Any]
            List of the labels in the dataset.
        """
        self.config = config if config else {"padding": "max_length", "truncation": True}
        self.labels = labels
        self.features = None

    def __call__(self, batch: Dict[str, List]):
        """
        Call the encoder.

        Parameters
        ----------
        batch : Dict[str, List]
            Batch of data to be encoded.
        """
        raise NotImplementedError()


class LayoutLMv2Encoder(BaseEncoder):
    """LayoutLMv2Encoder is the encoder for datasets using LayoutLMv2."""

    def __init__(self, **kwargs):
        """
        Initialize the LayoutLMv2Encoder.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Check the documentation of the LayoutLMv2Processor for the available parameters :
            https://huggingface.co/docs/transformers/model_doc/layoutlmv2#transformers.LayoutLMv2Processor
        """
        super().__init__(**kwargs)
        self.default_model = self.config.get("default_model", "microsoft/layoutlmv2-base-uncased")
        self.processor = LayoutLMv2Processor.from_pretrained(self.default_model, **self.config)
        self.features = Features(
            {
                "image": Array3D(dtype="int64", shape=(3, 224, 224)),
                "input_ids": Sequence(feature=Value(dtype="int64")),
                "attention_mask": Sequence(Value(dtype="int64")),
                "token_type_ids": Sequence(Value(dtype="int64")),
                "bbox": Array2D(dtype="int64", shape=(512, 4)),
                "labels": Sequence(ClassLabel(num_classes=len(self.labels), names=self.labels)),
            }
        )

    def __call__(self, batch: Dict[str, List]):
        """Call the LayoutLMv2Encoder."""
        images = [image.convert("RGB") for image in batch["image"]]
        encoded_inputs = self.processor(images)
        encoded_inputs["labels"] = [label for label in batch["label"]]
        return encoded_inputs


class LayoutLMv3Encoder(BaseEncoder):
    """LayoutLMv3Encoder is the encoder for datasets using LayoutLMv3."""

    def __init__(self, **kwargs):
        """
        Initialize the LayoutLMv3Encoder.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Check the documentation of the LayoutLMv3Processor for the available parameters :
            https://huggingface.co/docs/transformers/model_doc/layoutlmv3#transformers.LayoutLMv3Processor
        """
        super().__init__(**kwargs)
        self.default_model = self.config.get("default_model", "microsoft/layoutlmv3-base")
        self.processor = LayoutLMv3Processor.from_pretrained(self.default_model, **self.config)
        self.features = Features(
            {
                "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
                "input_ids": Sequence(feature=Value(dtype="int64")),
                "attention_mask": Sequence(Value(dtype="int64")),
                "bbox": Array2D(dtype="int64", shape=(512, 4)),
                "labels": Sequence(feature=Value(dtype='int64')),
            }
        )

    def __call__(self, batch: Dict[str, List]):
        """Call the LayoutLMv3Encoder."""
        images = [image.convert("RGB") for image in batch["image"]]
        encoded_inputs = self.processor(images)
        encoded_inputs["labels"] = [label for label in batch["label"]]
        return encoded_inputs


class LayoutXLMEncoder(BaseEncoder):
    """LayoutXLMEncoder is the encoder for datasets using LayoutXLM."""

    def __init__(self, **kwargs):
        """
        Initialize the LayoutXLMEncoder.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Check the documentation of the LayoutXLMProcessor for the available parameters :
            https://huggingface.co/docs/transformers/model_doc/layoutxlm#transformers.LayoutXLMProcessor
        """
        super().__init__(**kwargs)
        self.default_model = self.config.get("default_model", "microsoft/layoutxlm-base")
        self.config["return_token_type_ids"] = True
        self.processor = LayoutXLMProcessor.from_pretrained(self.default_model, **self.config)
        self.features = Features(
            {
                "image": Array3D(dtype="int64", shape=(3, 224, 224)),
                "input_ids": Sequence(feature=Value(dtype="int64")),
                "attention_mask": Sequence(Value(dtype="int64")),
                # "token_type_ids": Sequence(Value(dtype="int64")),
                "bbox": Array2D(dtype="int64", shape=(512, 4)),
                "labels": Sequence(ClassLabel(num_classes=len(self.labels), names=self.labels)),
            }
        )

    def __call__(self, batch: Dict[str, List]):
        """Call the LayoutXLMEncoder."""
        images = [image.convert("RGB") for image in batch["image"]]
        encoded_inputs = self.processor(images)
        encoded_inputs["labels"] = [label for label in batch["label"]]
        return encoded_inputs
