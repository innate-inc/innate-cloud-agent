# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc
"""Tests for image-loop liveness on never-activated terminals.

ready_for_image is withheld when a task is sent and normally re-armed by the
primitive_activated handler. Since innate-os#533/#557 the robot deliberately
answers never-started tasks with a terminal "failed" and NO activation — a
terminal handler that clears the state without re-arming the loop leaves no
image request outstanding and the vision agent silently stalls (INN-711).

Pins: a terminal resolving a never-activated primitive re-arms the loop; a
terminal after a normal activation does not double-grant; activation sets the
confirmation flag; a stale-id terminal touches nothing.
"""

import pytest

from src.agents.types import PrimitiveDefinition
from src.brain import Brain
from src.message_types import MessageIn, MessageInType


def make_brain():
    sent = []

    async def send_callback(message):
        sent.append(message)

    brain = Brain("test_connection", send_callback)
    return brain, sent


def arm_primitive(brain, *, confirmed):
    task = PrimitiveDefinition(name="turn_and_move", inputs={"x": 0.5}, primitive_id="prim-42")
    brain.state.primitive_ids_map["prim-42"] = task
    brain.state.primitive_in_execution = task
    brain.state.primitive_activation_seen = confirmed
    return task


def ready_for_image_count(sent):
    return sum(1 for m in sent if m.type == "ready_for_image")


@pytest.mark.asyncio
async def test_unactivated_terminal_rearms_image_loop():
    # The robot refused/never started the task and answered with a terminal
    # "failed" (see innate-os report_start_failure): activation never came,
    # so the terminal handler must re-arm the image loop itself.
    brain, sent = make_brain()
    arm_primitive(brain, confirmed=False)

    await brain.handle_primitive_failed(
        MessageIn(
            type=MessageInType.PRIMITIVE_FAILED,
            payload={"primitive_name": "turn_and_move", "primitive_id": "prim-42", "reason": "dropped"},
        )
    )

    assert brain.state.primitive_in_execution is None
    assert ready_for_image_count(sent) == 1


@pytest.mark.asyncio
async def test_confirmed_terminal_does_not_double_grant():
    # Normal life: activated re-armed the loop already; the terminal must not
    # inject an extra grant.
    brain, sent = make_brain()
    arm_primitive(brain, confirmed=True)

    await brain.handle_primitive_completed(
        MessageIn(
            type=MessageInType.PRIMITIVE_COMPLETED,
            payload={"primitive_name": "turn_and_move", "primitive_id": "prim-42", "output": "done"},
        )
    )

    assert brain.state.primitive_in_execution is None
    assert brain.state.primitive_activation_seen is False  # cleared for the next task
    assert ready_for_image_count(sent) == 0


@pytest.mark.asyncio
async def test_activation_sets_confirmation_flag():
    brain, _sent = make_brain()
    arm_primitive(brain, confirmed=False)

    await brain.handle_primitive_activated(
        MessageIn(
            type=MessageInType.PRIMITIVE_ACTIVATED,
            payload={"primitive_name": "turn_and_move", "primitive_id": "prim-42"},
        )
    )

    assert brain.state.primitive_activation_seen is True


@pytest.mark.asyncio
async def test_stale_terminal_leaves_state_and_loop_alone():
    # A terminal for a superseded primitive id is ignored by the handler; it
    # must neither clear the current primitive nor grant an image.
    brain, sent = make_brain()
    task = arm_primitive(brain, confirmed=False)

    await brain.handle_primitive_failed(
        MessageIn(
            type=MessageInType.PRIMITIVE_FAILED,
            payload={"primitive_name": "old", "primitive_id": "prim-OLD", "reason": "stale"},
        )
    )

    assert brain.state.primitive_in_execution is task
    assert ready_for_image_count(sent) == 0
