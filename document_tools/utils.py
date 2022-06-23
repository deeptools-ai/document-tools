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
"""utils.py group all utils functions in one file."""
from typing import List, Set


def _get_label_list(labels: List[str]) -> List[str]:
    """
    Get the list of all dataset labels.

    Parameters
    ----------
    labels : List[str]
        List of labels.

    Returns
    -------
    List[str]
        List of labels without duplicates and sorted.
    """
    unique_labels: Set[str] = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list
