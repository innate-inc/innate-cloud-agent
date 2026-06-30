# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

import pytest_asyncio

from tests.websocket_cleanup import close_tracked_websockets


@pytest_asyncio.fixture(autouse=True)
async def cleanup_websocket_tests():
    try:
        yield
    finally:
        await close_tracked_websockets()
