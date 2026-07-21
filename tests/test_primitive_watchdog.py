# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc
"""Tests for the primitive confirmation watchdog and image-loop liveness.

primitive_in_execution is set optimistically when a task is SENT and only a
terminal lifecycle message with the matching id clears it. Before these fixes
a single lost message (task dropped by the robot mid-registration, reply lost
in transit) pinned the agent forever: it kept narrating "the tool has already
been called and has to finish" while validate_vision_output silently
discarded every new task, and no image request was outstanding (INN-711).

Pins:
- the watchdog clears a sent-but-never-confirmed primitive and re-arms the
  image loop; a confirmed (activated) primitive is never touched;
- a terminal that resolves a never-activated primitive re-arms the image
  loop (activation, the usual re-arm source, never happened);
- a terminal after a normal activation does not double-grant.
"""

import time

import pytest

from src.agents.types import PrimitiveDefinition
from src.brain import Brain
from src.history.history import HistoryEntryType
from src.message_types import MessageIn, MessageInType


def make_brain():
    sent = []

    async def send_callback(message):
        sent.append(message)

    brain = Brain("test_connection", send_callback)
    return brain, sent


def arm_primitive(brain, *, confirmed, age_s=0.0):
    task = PrimitiveDefinition(name="turn_and_move", inputs={"x": 0.5}, primitive_id="prim-42")
    brain.state.primitive_ids_map["prim-42"] = task
    brain.state.primitive_in_execution = task
    brain.state.primitive_sent_at = time.monotonic() - age_s
    brain.state.primitive_activation_seen = confirmed
    return task


def ready_for_image_count(sent):
    return sum(1 for m in sent if m.type == "ready_for_image")


@pytest.mark.asyncio
async def test_watchdog_clears_unconfirmed_primitive():
    brain, sent = make_brain()
    arm_primitive(brain, confirmed=False, age_s=Brain.PRIMITIVE_CONFIRMATION_TIMEOUT_S + 1)

    await brain._check_primitive_watchdog()

    assert brain.state.primitive_in_execution is None
    assert brain.state.primitive_sent_at is None
    assert ready_for_image_count(sent) == 1
    cancelled = [e for e in brain.history.entries if e.type == HistoryEntryType.PRIMITIVE_CANCELLED]
    assert len(cancelled) == 1
    assert "never confirmed" in cancelled[0].description


@pytest.mark.asyncio
async def test_watchdog_spares_confirmed_primitive():
    # A confirmed primitive may legitimately run far longer than the timeout.
    brain, sent = make_brain()
    task = arm_primitive(brain, confirmed=True, age_s=Brain.PRIMITIVE_CONFIRMATION_TIMEOUT_S * 10)

    await brain._check_primitive_watchdog()

    assert brain.state.primitive_in_execution is task
    assert ready_for_image_count(sent) == 0


@pytest.mark.asyncio
async def test_watchdog_spares_recent_send():
    brain, sent = make_brain()
    task = arm_primitive(brain, confirmed=False, age_s=0.0)

    await brain._check_primitive_watchdog()

    assert brain.state.primitive_in_execution is task
    assert ready_for_image_count(sent) == 0


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
    assert ready_for_image_count(sent) == 0


@pytest.mark.asyncio
async def test_activation_confirms_and_disarms_watchdog():
    brain, _sent = make_brain()
    arm_primitive(brain, confirmed=False, age_s=Brain.PRIMITIVE_CONFIRMATION_TIMEOUT_S + 1)

    await brain.handle_primitive_activated(
        MessageIn(
            type=MessageInType.PRIMITIVE_ACTIVATED,
            payload={"primitive_name": "turn_and_move", "primitive_id": "prim-42"},
        )
    )
    assert brain.state.primitive_activation_seen is True

    await brain._check_primitive_watchdog()  # would have fired without the confirmation
    assert brain.state.primitive_in_execution is not None


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
