import os
import numpy as np

from src.Map.Shadowing import load_or_create_shadowing
from src.Map.Map import load_map


class SimpleSquareCameraParams:
    def __init__(self):
        self.half_length = 2
        self.map_path = "res/downtown.png"


class SimpleSquareCamera:

    def __init__(self, params: SimpleSquareCameraParams):
        self.params = params
        total_map = load_map(self.params.map_path)
        self.obstacles = total_map.obstacles
        self.size = self.obstacles.shape[:2]
        self.obstruction_map = load_or_create_shadowing(self.params.map_path)

    def computeView(self, position, attitude):
        view = np.zeros(self.size, dtype=bool)
        camera_width = self.params.half_length
        x_pos, y_pos = position[0], position[1]
        x_size, y_size = self.size[0], self.size[1]
        view[max(0, y_pos - camera_width):min(y_size, y_pos + camera_width + 1),
             max(0, x_pos - camera_width):min(x_size, x_pos + camera_width + 1)] = True

        view &= ~self.obstacles
        view &= ~self.obstruction_map[y_pos, x_pos]
        return view
