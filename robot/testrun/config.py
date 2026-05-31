import numpy as np


# This script is POSITION-ONLY.
# No orientation commands.
# No joint commands.

DEFAULT_JSON_PATH = "strokes_canvas_plane.json"

# Test-run canvas displacement.
# Positive x shifts every loaded canvas stroke point forward by this amount.
# Tune this to test generated stroke paths without contacting the real canvas.
CANVAS_X_OFFSET_M = 0.03
CANVAS_OFFSET_VEC = np.array([CANVAS_X_OFFSET_M, 0.0, 0.0], dtype=float)

config_file_for_this_example = "basket.xml"
controller_to_use = "cartesian_controller"

# Main loop timing.
DT = 0.01  # 100 Hz

# Position tolerance for advancing between waypoints.
POS_TOL_M = 1.0e-2  # 10 mm

# Segment interpolation resolution.
# Smaller = more intermediate waypoints = slower/smoother.
TRAVEL_STEP_M = 0.006   # free-space travel / lift moves
DRAW_STEP_M = 0.003     # drawing motion on paper

# Retraction direction.
# User requested:
#   lift off by retracting in -x by -0.1
#   then go forward in +x by +0.1
#
# Units are meters.
# -0.1 = -10 cm in world x.
RETRACT_X_M = -0.08
RETRACT_VEC = np.array([RETRACT_X_M, 0.0, 0.0], dtype=float)

# INIT position from calibration.
# Position only. No orientation.
INIT_POS = np.array([0.54671, 0.11226, 0.33151], dtype=float)

# Optional dwell times.
DWELL_AT_INIT_S = 0.5
DWELL_BEFORE_CONTACT_S = 0.10
DWELL_AFTER_RETRACT_S = 0.05

# Damping profile.
# This makes the end of each segment land gently.
DAMP_START = 0.70
DAMP_POWER = 3.0

# Safety validation.
MIN_POINT_DIM = 3
