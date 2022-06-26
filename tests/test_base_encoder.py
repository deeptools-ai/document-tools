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
from typing import List

import numpy as np
import pytest

from document_tools.encoders.encoders import BaseEncoder


@pytest.fixture()
def get_labels():
    return [1, 2, 3, 4, 5]


def test_base_encoder(get_labels: List[int]):
    encoder = BaseEncoder(labels=get_labels)
    assert encoder.labels == get_labels
    assert encoder.config == {"padding": "max_length", "truncation": True}
    assert encoder.features is None


def test_base_encoder_call(get_labels: List[int]):
    encoder = BaseEncoder(labels=get_labels)
    with pytest.raises(NotImplementedError):
        encoder({"image": [np.zeros((1, 1, 1, 1))]})
