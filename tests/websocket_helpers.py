# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

import json


async def wait_for_ready_after_chat(websocket):
    """
    A chat_in without image_b64 produces an immediate protocol response before
    the client should send the next image. The fast agent may emit chat_out
    first, but the turn is complete once ready_for_image arrives.
    """
    while True:
        msg = json.loads(await websocket.recv())
        if msg["type"] == "ready_for_image":
            return msg

        assert msg["type"] == "chat_out", (
            "Expected chat_out or ready_for_image after chat_in, "
            f"got {msg['type']}"
        )
