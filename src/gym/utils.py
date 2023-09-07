import math
import os
import pickle
import time

import numpy as np
import pygame
from skimage import io


class Map:
    def __init__(self, map_data, path, skip_model=False):
        self.name = path.split('/')[-1].split('.')[0]

        self.map_data = map_data.astype(bool).transpose((1, 0, 2))[..., :3]
        self.start_land_zone = self.map_data[:, :, 2]
        self.nfz = self.map_data[:, :, 0]
        self.obstacles = self.map_data[:, :, 1]

        self.original_shape = map_data.shape[:2]
        self.shadowing_map = None
        self.landing_map = None
        self.distance_map = None
        self.model = None

        if skip_model:
            return

        self.init_model(path)

    def init_model(self, path):
        self.model = self.load_or_create_model(path)
        self.distance_map = self.model["distances"]
        self.landing_map = self.model["landing"]
        self.shadowing_map = self.model["shadowing"]

    def pad_to_size(self, size):
        old = self.get_size()
        self.start_land_zone = np.pad(self.start_land_zone, ((0, size[0] - old[0]), (0, size[1] - old[1])),
                                      constant_values=0)
        self.nfz = np.pad(self.nfz, ((0, size[0] - old[0]), (0, size[1] - old[1])), constant_values=1)
        self.obstacles = np.pad(self.obstacles, ((0, size[0] - old[0]), (0, size[1] - old[1])), constant_values=1)

    def get_starting_vector(self):
        similar = np.where(self.start_land_zone)
        return list(zip(similar[1], similar[0]))

    def get_free_space_vector(self):
        free_space = np.logical_not(
            np.logical_or(self.obstacles, self.start_land_zone))
        free_idcs = np.where(free_space)
        return list(zip(free_idcs[1], free_idcs[0]))

    def get_size(self):
        return self.start_land_zone.shape[:2]

    @staticmethod
    def load_map(path):
        if type(path) is not str:
            raise TypeError('path needs to be a string')
        data = io.imread(path, as_gray=False)
        return Map(data, path)

    def load_or_create_model(self, map_path):
        filename = os.path.splitext(map_path)[0] + "_model.pickle"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                model = pickle.load(f)
                if "map_data" in model and model["map_data"].shape == self.map_data.shape and np.equal(
                        model["map_data"], self.map_data).all():
                    return model
                else:
                    print(f"Map {self.name} appears to have changed. Recomputing model.")
        model = self.calculate_model()
        with open(filename, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        return model

    def calculate_model(self):
        print(f"Calculating model for map {self.name}")
        model = {"distances": self.calculate_shortest_distance(),
                 "landing": self.calculate_landing(),
                 "shadowing": self.calculate_shadowing(),
                 "map_data": self.map_data}
        return model

    def calculate_shortest_distance(self):
        print("Calculating shortest distance map")
        nfz = np.pad(self.nfz, pad_width=((1, 1), (1, 1)), constant_values=1)
        x, y = nfz.shape
        nfz = np.expand_dims(np.expand_dims(nfz, axis=0), axis=0)  # [1, 1, x, y]
        nfz = np.logical_or(np.transpose(nfz, (2, 3, 0, 1)), nfz)

        min_distance = np.ones((x, y, x, y)) * np.inf
        x_index = np.repeat(np.arange(x), y, axis=0)
        y_index = np.tile(np.arange(y), x)

        min_distance[x_index, y_index, x_index, y_index] = 0
        previous = np.zeros_like(min_distance)

        while np.not_equal(previous, min_distance).any():
            previous = min_distance
            temp = min_distance + 1
            t1 = np.roll(temp, shift=1, axis=2)
            t2 = np.roll(temp, shift=-1, axis=2)
            t3 = np.roll(temp, shift=1, axis=3)
            t4 = np.roll(temp, shift=-1, axis=3)
            t = np.stack((min_distance, t1, t2, t3, t4), axis=-1)
            min_distance = np.where(nfz, np.inf, np.min(t, axis=-1))

        min_distance = min_distance[1:-1, 1:-1, 1:-1, 1:-1]
        min_distance = np.where(min_distance == np.inf, -1, min_distance).astype(int)

        return min_distance

    def calculate_landing(self):
        print("Calculating landing map")
        nfz = np.pad(self.nfz, pad_width=((1, 1), (1, 1)), constant_values=1)
        slz = np.pad(self.start_land_zone, pad_width=((1, 1), (1, 1)), constant_values=0)

        min_distance = np.where(slz, 1, np.inf)
        previous = np.zeros_like(min_distance)
        while np.not_equal(previous, min_distance).any():
            previous = min_distance
            temp = min_distance + 1
            t1 = np.roll(temp, shift=1, axis=0)
            t2 = np.roll(temp, shift=-1, axis=0)
            t3 = np.roll(temp, shift=1, axis=1)
            t4 = np.roll(temp, shift=-1, axis=1)
            t = np.stack((min_distance, t1, t2, t3, t4), axis=-1)
            min_distance = np.where(nfz, np.inf, np.min(t, axis=-1))

        min_distance = min_distance[1:-1, 1:-1]
        min_distance = np.where(min_distance == np.inf, -1, min_distance).astype(int)
        return min_distance

    def calculate_shadowing(self):
        print("Calculating shadowing maps")
        obstacles = self.obstacles
        size = self.obstacles.shape[0]
        total = size * size

        total_shadow_map = np.ones((size, size, size, size), dtype=bool)
        for i, j in np.ndindex(self.obstacles.shape):
            if self.obstacles[i, j]:
                continue
            shadow_map = np.ones((size, size), dtype=bool)

            for x in range(size):
                bresenham(i, j, x, 0, obstacles, shadow_map)
                bresenham(i, j, x, size - 1, obstacles, shadow_map)
                bresenham(i, j, 0, x, obstacles, shadow_map)
                bresenham(i, j, size - 1, x, obstacles, shadow_map)

            total_shadow_map[i, j] = shadow_map

        return total_shadow_map


def load_image(path):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=True)
    return np.array(data, dtype=bool)


def save_image(path, image):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    if image.dtype == bool:
        io.imsave(path, image * np.uint8(255))
    else:
        io.imsave(path, image)


def load_target(path, obstacles=None):
    if type(path) is not str:
        raise TypeError('path needs to be a string')

    data = np.array(io.imread(path, as_gray=True), dtype=bool)
    if obstacles is not None:
        data = data & ~obstacles
    return data


def bresenham(x0, y0, x1, y1, obstacles, shadow_map):
    x_dist = abs(x0 - x1)
    y_dist = -abs(y0 - y1)
    x_step = 1 if x1 > x0 else -1
    y_step = 1 if y1 > y0 else -1

    error = x_dist + y_dist

    # shadowed = False
    shadow_map[x0, y0] = False

    while x0 != x1 or y0 != y1:
        if 2 * error - y_dist > x_dist - 2 * error:
            # horizontal step
            error += y_dist
            x0 += x_step
        else:
            # vertical step
            error += x_dist
            y0 += y_step

        if obstacles[x0, y0]:
            # shadowed = True
            return

        # if shadowed:
        shadow_map[x0, y0] = False


def get_arrow_polygon(origin, destination, head_width=8, head_length=8, shaft_width=2):
    # Calculate the unit vector in the direction of the arrow
    diff = [destination[0] - origin[0], destination[1] - origin[1]]
    arrow_length = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
    unit_vec = [diff[0] / arrow_length, diff[1] / arrow_length]

    # Calculate the points on the arrowhead
    tip = destination
    base1 = [destination[0] - head_length * unit_vec[0] - head_width / 2 * unit_vec[1],
             destination[1] - head_length * unit_vec[1] + head_width / 2 * unit_vec[0]]
    base2 = [destination[0] - head_length * unit_vec[0] + head_width / 2 * unit_vec[1],
             destination[1] - head_length * unit_vec[1] - head_width / 2 * unit_vec[0]]

    # Calculate the points on the shaft
    shaft_base1 = [origin[0] + shaft_width / 2 * unit_vec[1], origin[1] - shaft_width / 2 * unit_vec[0]]
    shaft_base2 = [origin[0] - shaft_width / 2 * unit_vec[1], origin[1] + shaft_width / 2 * unit_vec[0]]
    shaft_tip1 = [origin[0] + (arrow_length - head_length) * unit_vec[0] + shaft_width / 2 * unit_vec[1],
                  origin[1] + (arrow_length - head_length) * unit_vec[1] - shaft_width / 2 * unit_vec[0]]
    shaft_tip2 = [origin[0] + (arrow_length - head_length) * unit_vec[0] - shaft_width / 2 * unit_vec[1],
                  origin[1] + (arrow_length - head_length) * unit_vec[1] + shaft_width / 2 * unit_vec[0]]

    # Return the points as a list of coordinates
    return [tip, base1, shaft_tip1, shaft_base1, shaft_base2, shaft_tip2, base2]


def get_visibility_map(half_length, shadowing_map):
    sm = np.logical_not(shadowing_map)
    n, m = sm.shape[:2]
    for i in range(n):
        if i - half_length > 0:
            sm[i, :, :i - half_length, :] = False
        if i + half_length + 1 < n:
            sm[i, :, i + half_length + 1:, :] = False

    for i in range(m):
        if i - half_length > 0:
            sm[:, i, :, :i - half_length] = False
        if i + half_length + 1 < m:
            sm[:, i, :, i + half_length + 1:] = False

    return sm


def draw_text(canvas, font, text, position, stroke_width=1, fill=(0, 0, 0),
              stroke=None, align="center"):
    text_surface = font.render(text, True, fill)  # .convert_alpha()
    shape = np.array(text_surface.get_size())
    position = np.array(position)
    if "top" in align:
        pass
    elif "bottom" in align:
        position[1] -= shape[1]
    else:
        position[1] -= shape[1] / 2

    if "left" in align:
        pass
    elif "right" in align:
        position[0] -= shape[0]
    else:
        position[0] -= shape[0] / 2
    size = shape

    if stroke is not None:
        text_surface_back = font.render(text, True, stroke)  # .convert_alpha()
        up = np.array((0, stroke_width))
        ri = np.array((stroke_width, 0))
        canvas.blit(text_surface_back, position + ri)
        canvas.blit(text_surface_back, position + ri + up)
        canvas.blit(text_surface_back, position + up)
        canvas.blit(text_surface_back, position - ri + up)
        canvas.blit(text_surface_back, position - ri)
        canvas.blit(text_surface_back, position - ri - up)
        canvas.blit(text_surface_back, position - up)
        canvas.blit(text_surface_back, position + ri - up)
        size += [2 * stroke_width, 2 * stroke_width]

    canvas.blit(text_surface, position)

    return size


def draw_shape_matrix(mat, canvas_shape, pix_size, fill=None, stroke=None, stroke_width=1):
    canvas = pygame.Surface(canvas_shape, pygame.SRCALPHA, 32)
    padded = np.pad(mat, ((1, 1), (1, 1)))
    half_stroke = (stroke_width + 1) // 2
    size = pix_size if np.ceil(pix_size) == np.floor(pix_size) else pix_size + 1
    for pos in np.transpose(np.nonzero(padded)):

        pos_image = (pos - [1, 1]) * pix_size

        if fill is not None:
            patch = pygame.Rect(pos_image, (size, size))
            pygame.draw.rect(canvas, fill, patch)

        if stroke is not None:
            x, y = pos
            if not padded[x - 1, y]:
                pygame.draw.line(canvas, stroke, pos_image + [half_stroke - 1, 0],
                                 pos_image + [half_stroke - 1, pix_size - 1],
                                 width=stroke_width)
            if not padded[x + 1, y]:
                pygame.draw.line(canvas, stroke, pos_image + [pix_size - half_stroke - 1, 0],
                                 pos_image + [pix_size - half_stroke - 1, pix_size - 1], width=stroke_width)
            if not padded[x, y - 1]:
                pygame.draw.line(canvas, stroke, pos_image + [0, half_stroke - 1],
                                 pos_image + [pix_size - 1, half_stroke - 1],
                                 width=stroke_width)
            if not padded[x, y + 1]:
                pygame.draw.line(canvas, stroke, pos_image + [0, pix_size - half_stroke - 1],
                                 pos_image + [pix_size - 1, pix_size - half_stroke - 1], width=stroke_width)

            if not padded[x - 1, y - 1]:
                pygame.draw.rect(canvas, stroke, pygame.Rect(pos_image, [stroke_width, stroke_width]))
            if not padded[x - 1, y + 1]:
                pygame.draw.rect(canvas, stroke, pygame.Rect(pos_image + [0, pix_size - stroke_width],
                                                             [stroke_width, stroke_width]))
            if not padded[x + 1, y - 1]:
                pygame.draw.rect(canvas, stroke, pygame.Rect(pos_image + [pix_size - stroke_width, 0],
                                                             [stroke_width, stroke_width]))
            if not padded[x + 1, y + 1]:
                pygame.draw.rect(canvas, stroke, pygame.Rect(pos_image + pix_size - stroke_width,
                                                             [stroke_width, stroke_width]))
    return canvas
