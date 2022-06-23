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
"""Export the classes and functions in this module to the package."""
from .encoders import LayoutLMv2Encoder, LayoutLMv3Encoder, LayoutXLMEncoder

TARGET_MODELS = {"layoutlmv2": LayoutLMv2Encoder, "layoutlmv3": LayoutLMv3Encoder, "layoutxlm": LayoutXLMEncoder}


__all__ = [
    "LayoutLMv2Encoder",
    "LayoutLMv3Encoder",
    "LayoutXLMEncoder",
    "TARGET_MODELS",
]