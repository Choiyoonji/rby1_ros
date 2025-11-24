import numpy as np
import cv2

def draw_3d_point(ax, point, color='r', label=None):
    ax.scatter(point[0], point[1], point[2], color=color)
    if label is not None:
        ax.text(point[0], point[1], point[2], label, color=color)

def draw_3d_line(ax, start_point, end_point, color='b'):
    ax.plot([start_point[0], end_point[0]], 
            [start_point[1], end_point[1]], 
            [start_point[2], end_point[2]], color=color)

def create_3d_plot():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    return fig, ax

def convert_to_3d_coordinates(pos):
    return np.array([pos[0], pos[1], pos[2]])  # Assuming pos is (x, y, z)

def update_3d_visualization(ax, current_positions, desired_positions):
    ax.cla()  # Clear the current axes
    for key, pos in current_positions.items():
        if pos is not None:
            draw_3d_point(ax, convert_to_3d_coordinates(pos), color='g', label=f'{key}_cur')
    for key, pos in desired_positions.items():
        if pos is not None:
            draw_3d_point(ax, convert_to_3d_coordinates(pos), color='r', label=f'{key}_des')
    ax.legend()