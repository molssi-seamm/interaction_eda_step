#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `interaction_eda_step` package."""

import pytest  # noqa: F401
import interaction_eda_step  # noqa: F401


def test_construction():
    """Just create an object and test its type."""
    result = interaction_eda_step.InteractionEda()
    assert (
        str(type(result))
        == "<class 'interaction_eda_step.interaction_eda.InteractionEda'>"
    )
