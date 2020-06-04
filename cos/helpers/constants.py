"""
A collection of constants that probably should not be changed
"""

import numpy as np

# Universal Constants
SPEED_OF_SOUND = 343.0  # m/s
FAR_FIELD_RADIUS = 3.0  # meters, assume larger the mic array radius

# Algorithmic Constants
ALL_WINDOW_SIZES = [
    np.pi / 2,  # 90 degrees
    np.pi / 4,  # 45 degrees
    np.pi / 8,  # 22.5 degrees
    np.pi / 16,  # 11.25 degrees
    np.pi / 32,  # 5.625 degrees
]
