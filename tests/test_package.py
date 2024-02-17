from __future__ import annotations

import importlib.metadata

import mammoth as m


def test_version():
    assert importlib.metadata.version("mammoth") == m.__version__
