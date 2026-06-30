# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

"""
Constants for the brain module.
Centralizes magic strings and values to improve maintainability.
"""


class PrimitiveNames:
    """Primitive name constants to avoid magic strings throughout the codebase."""

    NAVIGATE_IN_SIGHT = "navigate_in_sight"
    NAV_INSIGHT_CONTINUOUS = "nav_insight_continuous"
    NAVIGATE_THROUGH_MEMORY = "navigate_through_memory"
    NAVIGATE_TO_POSITION = "navigate_to_position"
    TURN_AND_MOVE = "turn_and_move"
    CHECK_DISTANCE_AND_ORIENTATION = "check_distance_and_orientation"


# Navigation primitives that require special handling in the brain
NAVIGATION_PRIMITIVES = frozenset(
    [
        PrimitiveNames.NAVIGATE_IN_SIGHT,
        PrimitiveNames.NAV_INSIGHT_CONTINUOUS,
        PrimitiveNames.NAVIGATE_THROUGH_MEMORY,
        PrimitiveNames.TURN_AND_MOVE,
        PrimitiveNames.CHECK_DISTANCE_AND_ORIENTATION,
    ]
)

# Local primitives whose handlers emit a navigate_to_position task. They can only
# be exposed to the VLM if the client has registered navigate_to_position.
PRIMITIVES_REQUIRING_NAVIGATE_TO_POSITION = frozenset(
    [
        PrimitiveNames.NAVIGATE_IN_SIGHT,
        PrimitiveNames.NAV_INSIGHT_CONTINUOUS,
        PrimitiveNames.NAVIGATE_THROUGH_MEMORY,
        PrimitiveNames.TURN_AND_MOVE,
    ]
)


def filter_locals_requiring_navigate_to_position(local_primitives, input_primitives):
    """Drop local primitives that need navigate_to_position when the client hasn't registered it."""
    has_navigate_to_position = any(
        p.name == PrimitiveNames.NAVIGATE_TO_POSITION for p in input_primitives
    )
    if has_navigate_to_position:
        return list(local_primitives)
    return [
        p
        for p in local_primitives
        if p.name not in PRIMITIVES_REQUIRING_NAVIGATE_TO_POSITION
    ]
