"""Tests for Streamlit UI async bridge helpers."""

from __future__ import annotations

import pytest

from KairoScale.ui.app import _run_coro


def test_runCoroOutsideEventLoop():
    async def _sample() -> int:
        return 7

    assert _run_coro(_sample()) == 7


@pytest.mark.asyncio
async def test_runCoroInsideActiveEventLoop():
    async def _sample() -> str:
        return "ok"

    assert _run_coro(_sample()) == "ok"
