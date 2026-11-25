"""Explicit model-visible boundary for the strict information ablation.

Apply BEFORE archiving, mounting or hashing observations, not just display.
Unknown source fields fail closed rather than silently reaching Python tools.
"""
from copy import deepcopy

SOURCE_FIELDS = frozenset('source_observation_index instruction state calibration home_check '
    'control_steps_used control_steps_sent unacknowledged_actions control_step_limit '
    'control_steps_remaining control_time_remaining_s control_step_limit_source '
    'sim_time_s control_dt_s execution_limits sim_time_source'.split())
ARM_FIELDS = frozenset('grasp_point tool_axis jaw_axis gripper_pose_points '
    'gripper_pose_finger_links commanded_gripper flange_point fingertip_center '
    'distal_pad_bbox_center commanded_gap_mm collision_gap_range_mm '
    'held_width_estimate_mm collision_geometry_source gripper_state_semantics '
    'home_position_error_m home_orientation_error_deg near_home '
    'home_grasp_point home_tool_axis home_jaw_axis'.split())
DROP_ARM = frozenset(('near_home', 'home_position_error_m', 'home_orientation_error_deg'))
FORBIDDEN_KEYS = frozenset(('home_check', 'near_home', 'home_position_error_m',
    'home_orientation_error_deg', 'reward', 'score', 'success', 'object_pose',
    'object_position', 'target_pose', 'target_position', 'segmentation', 'depth'))


def validate_published(observation):
    """Check the exact raw dict exposed to Python, including execution feedback."""
    def walk(value):
        if isinstance(value, dict):
            forbidden = set(value) & FORBIDDEN_KEYS
            if forbidden:
                raise ValueError('Restricted strict observation keys: '+repr(sorted(forbidden)))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(observation)


def strict_observation(observation):
    unknown = set(observation) - SOURCE_FIELDS
    if unknown:
        raise ValueError('Unaudited strict observation fields: '+repr(sorted(unknown)))
    value = deepcopy(observation)
    value.pop('home_check', None)
    for arm, state in value['state'].items():
        if arm not in ('left', 'right') or set(state) - ARM_FIELDS:
            raise ValueError('Unaudited strict robot state: '+arm)
        for key in DROP_ARM:
            state.pop(key, None)
    return value
