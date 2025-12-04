"""
Constants for the brain module.
Centralizes magic strings and values to improve maintainability.
"""


class PrimitiveNames:
    """Primitive name constants to avoid magic strings throughout the codebase."""

    NAVIGATE_IN_SIGHT = "navigate_in_sight"
    NAVIGATE_THROUGH_MEMORY = "navigate_through_memory"
    NAVIGATE_TO_POSITION = "navigate_to_position"
    TURN_AND_MOVE = "turn_and_move"
    CHECK_DISTANCE_AND_ORIENTATION = "check_distance_and_orientation"


# Navigation primitives that require special handling in the brain
NAVIGATION_PRIMITIVES = frozenset(
    [
        PrimitiveNames.NAVIGATE_IN_SIGHT,
        PrimitiveNames.NAVIGATE_THROUGH_MEMORY,
        PrimitiveNames.TURN_AND_MOVE,
        PrimitiveNames.CHECK_DISTANCE_AND_ORIENTATION,
    ]
)

