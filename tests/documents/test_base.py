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
import pytest

from document_tools.documents.base import Document


@pytest.fixture
def file_name():
    return "test.txt"


def test_document_base_class(file_name):
    """
    Test the base class for all documents.
    """
    document = Document(file_name)
    assert document.file == file_name
    assert document.extension == file_name.split(".")[-1]


def test_document_repr(file_name):
    """
    Test the representation of a document.
    """
    document = Document(file_name)
    assert repr(document) == f"Document(file='{file_name}')"


def test_document_eq(file_name):
    """
    Test the equality of two documents.
    """
    document1 = Document(file_name)
    document2 = Document(file_name)
    assert document1 == document2
