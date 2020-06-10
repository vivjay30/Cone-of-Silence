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

def get_mic_diagram():
    """
    A simple hard-coded mic figure for matplotlib
    """
    import matplotlib
    matplotlib.use("Agg")
    mic_verts = np.array([[24.  , 28.  ],
                           [27.31, 28.  ],
                           [29.98, 25.31],
                           [29.98, 22.  ],
                           [30.  , 10.  ],
                           [30.  ,  6.68],
                           [27.32,  4.  ],
                           [24.  ,  4.  ],
                           [20.69,  4.  ],
                           [18.  ,  6.68],
                           [18.  , 10.  ],
                           [18.  , 22.  ],
                           [18.  , 25.31],
                           [20.69, 28.  ],
                           [24.  , 28.  ],
                           [24.  , 28.  ],
                           [34.6 , 22.  ],
                           [34.6 , 28.  ],
                           [29.53, 32.2 ],
                           [24.  , 32.2 ],
                           [18.48, 32.2 ],
                           [13.4 , 28.  ],
                           [13.4 , 22.  ],
                           [10.  , 22.  ],
                           [10.  , 28.83],
                           [15.44, 34.47],
                           [22.  , 35.44],
                           [22.  , 42.  ],
                           [26.  , 42.  ],
                           [26.  , 35.44],
                           [32.56, 34.47],
                           [38.  , 28.83],
                           [38.  , 22.  ],
                           [34.6 , 22.  ],
                           [34.6 , 22.  ]])
    mic_verts[:,1] = (48 - mic_verts[:,1]) - 24
    mic_verts[:,0] -= 24

    mic_verts[:,0] /= 240
    mic_verts[:,1] /= 240

    mic_verts *= 10

    mic_codes = np.array([ 1,  4,  4,  4,  2,  4,  4,  4,  4,  4,  4,  2,  4,  4,  4, 79,  1,
                            4,  4,  4,  4,  4,  4,  2,  4,  4,  4,  2,  2,  2,  4,  4,  4,  2,
                           79], dtype=np.uint8)
    
    mic = matplotlib.path.Path(mic_verts, mic_codes)
    return mic
