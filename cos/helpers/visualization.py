import os

import numpy as np

import cos.helpers.utils as utils
from cos.helpers.constants import get_mic_diagram



def draw_diagram(voice_positions, candidate_angles, angle_window_size, output_file):
    """
    Draws the setup of all the voices in space, and colored triangles for the beams
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge, Polygon
    from matplotlib.collections import PatchCollection
    matplotlib.style.use('ggplot')

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [colors[0], colors[3], colors[2], colors[1]]

    fig, ax = plt.subplots()
    ax.set(xlim=(-5, 5), ylim = (-5, 5))
    ax.set_aspect("equal")
    ax.annotate('X', xy=(-5,0), xytext=(5.2,0),
                arrowprops={'arrowstyle': '<|-', "color": "black", "linewidth":3}, va='center', fontsize=20)
    ax.annotate('Y', xy=(0,-5), xytext=(-0.25, 5.5),
                arrowprops={'arrowstyle': '<|-', "color": "black", "linewidth":3}, va='center', fontsize=20)

    plt.tick_params(axis='both',
        which='both', bottom='off',
        top='off', labelbottom='off', right='off', left='off', labelleft='off'
    )

    for pos in voice_positions:
        if pos[0] != 0.0:
            a_circle = plt.Circle((pos[0], pos[1]), 0.3, color='b', fill=False)
            ax.add_artist(a_circle)

    patches = []
    for idx, target_angle in enumerate(candidate_angles):
        vertices = angle_to_triangle(target_angle, angle_window_size) * 4.96

        ax.fill(vertices[:, 0], vertices[:, 1],
                edgecolor='black', linewidth=2, alpha=0.6)

    mic = get_mic_diagram()
    patch = matplotlib.patches.PathPatch(mic, fill=True, facecolor='black')
    ax.add_patch(patch)
    ax.tick_params(axis='both', which='both', labelcolor="white", colors="white")
    plt.savefig(output_file)


def angle_to_triangle(target_angle, angle_window_size):
    """
    Takes a target angle and window size and returns a
    triangle corresponding to that pie slice
    """
    first_point = [0,0]  # Always start at the origiin
    second_point = angle_to_point(utils.convert_angular_range(target_angle - angle_window_size/2))
    third_point = angle_to_point(utils.convert_angular_range(target_angle + angle_window_size/2))

    return(np.array([first_point, second_point, third_point]))


def angle_to_point(angle):
    """Angle must be -pi to pi"""
    if -np.pi <= angle < -3*np.pi/4:
        return[-1, -np.tan(angle + np.pi)]

    elif -3*np.pi/4 <= angle < -np.pi/2:
        return[-np.tan(-np.pi/2 - angle), -1]

    elif -np.pi/2 <= angle < -np.pi/4:
        return[np.tan(angle + np.pi/2), -1]

    elif -np.pi/4 <= angle < 0:
        return[1, -np.tan(-angle)]

    elif 0 <= angle < np.pi / 4:
        return [1, np.tan(angle)]

    elif np.pi/4 <= angle < np.pi/2:
        return [np.tan(np.pi/2 - angle), 1]

    elif np.pi/2 <= angle <= 3*np.pi/4:
        return [-np.tan(angle - np.pi/2), 1]

    elif 3*np.pi/4 < angle <= np.pi:
        return [-1, np.tan(np.pi - angle)]
