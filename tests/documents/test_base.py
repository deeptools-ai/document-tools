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
def file_names():
    return ["file1.txt", "file2.txt", "my_image.png", "my_favorite_file.pdf", "my.other.file.pdf"]


def test_document_base_class(file_names):
    """
    Test the base class for all documents.
    """
    for file_name in file_names:
        document = Document(file_name)
        assert document.file == file_name
        assert document.extension == file_name.split(".")[-1]


def test_document_repr(file_names):
    """
    Test the representation of a document.
    """
    for file_name in file_names:
        document = Document(file_name)
        assert repr(document) == f"Document(file='{file_name}')"


def test_document_eq(file_names):
    """
    Test the equality of two documents.
    """
    for file_name in file_names:
        document1 = Document(file_name)
        document2 = Document(file_name)
        assert document1 == document2
