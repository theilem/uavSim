import math
import os
import pickle
from queue import Empty, Full
from time import sleep

import numpy as np
import pygame
from scipy.ndimage import convolve
from skimage import io


class Map:
    def __init__(self, map_data, name, map_path=None):
        self.name = name
        self.map_data = map_data
        self.original_shape = map_data.shape[:2]
        self.model = {"name": name, "map_data": map_data}
        self.map_path = map_path

        if self.map_path is not None:
            self.load_or_create_model(self.map_path)

    def pad_to_size(self, size, padding_values=(0, 1, 1)):
        x, y = self.map_data.shape[:2]
        padded = np.ones((*size, 3), dtype=bool) & np.array(padding_values).astype(bool)[np.newaxis, np.newaxis, :]
        padded[:x, :y] = self.map_data
        self.map_data = padded

    def pad_model(self, max_size):
        distances = self.model["distances"]
        landing = self.model["landing"]
        shadowing = self.model["shadowing"]

        xo, yo = self.original_shape
        x, y = max_size

        pad_x = x - xo
        pad_y = y - yo

        self.model["distances"] = np.pad(distances, ((0, pad_x), (0, pad_y), (0, pad_x), (0, pad_y)),
                                         constant_values=-1)
        self.model["landing"] = np.pad(landing, ((0, pad_x), (0, pad_y)), constant_values=-1)
        self.model["shadowing"] = np.pad(shadowing, ((0, pad_x), (0, pad_y), (0, pad_x), (0, pad_y)),
                                         constant_values=True)

        self.original_shape = max_size

    def get_starting_vector(self):
        similar = np.where(self.slz)
        return list(zip(similar[1], similar[0]))

    def get_free_space_vector(self):
        free_space = np.logical_not(
            np.logical_or(self.obst, self.slz))
        free_idcs = np.where(free_space)
        return list(zip(free_idcs[1], free_idcs[0]))

    def get_size(self):
        return self.map_data.shape[:2]

    @staticmethod
    def load_map(path):
        if type(path) is not str:
            raise TypeError('path needs to be a string')
        data = io.imread(path, as_gray=False)
        data = data.astype(bool).transpose((1, 0, 2))[..., :3]
        name = os.path.splitext(os.path.split(path)[1])[0]
        return Map(data, name, map_path=path)

    def load_or_create_model(self, map_path):
        filename = os.path.splitext(map_path)[0] + ".pickle"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                model = pickle.load(f)
                if "map_data" in model and model["map_data"].shape == self.map_data.shape and np.equal(
                        model["map_data"], self.map_data).all():
                    self.model = model
                else:
                    print(f"Map {self.name} appears to have changed. Deleting old model.")
        else:
            self.save_model(map_path)

    def save_model(self, path=None):
        if path is None:
            path = self.map_path
        filename = os.path.splitext(path)[0] + ".pickle"
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def all_distances(self):
        if "distances" in self.model:
            return self.model["distances"]
        self.model["distances"] = calculate_all_distances(self.nfz)
        if self.map_path is not None:
            self.save_model(self.map_path)
        return self.model["distances"]

    def landing_distances(self):
        if "landing" in self.model:
            return self.model["landing"]
        self.model["landing"] = calculate_landing_distances(self.nfz, self.slz)
        if self.map_path is not None:
            self.save_model(self.map_path)
        return self.model["landing"]

    def shadowing(self):
        if "shadowing" in self.model:
            return self.model["shadowing"]
        self.model["shadowing"] = calculate_shadowing(self.obst)
        if self.map_path is not None:
            self.save_model(self.map_path)
        return self.model["shadowing"]

    @property
    def slz(self):
        return self.map_data[:self.original_shape[0], :self.original_shape[1], 2]

    @property
    def obst(self):
        return self.map_data[:self.original_shape[0], :self.original_shape[1], 1]

    @property
    def nfz(self):
        return self.map_data[:self.original_shape[0], :self.original_shape[1], 0]

    def rotate_90(self):
        self.name = self.name + "_rot"
        self.map_data = np.rot90(self.map_data, k=1, axes=(0, 1))
        self.model["map_data"] = self.map_data
        if "distances" in self.model:
            self.model["distances"] = np.rot90(self.model["distances"], k=1, axes=(0, 1))
            self.model["distances"] = np.rot90(self.model["distances"], k=1, axes=(2, 3))
        if "landing" in self.model:
            self.model["landing"] = np.rot90(self.model["landing"], k=1, axes=(0, 1))
        if "shadowing" in self.model:
            self.model["shadowing"] = np.rot90(self.model["shadowing"], k=1, axes=(0, 1))
            self.model["shadowing"] = np.rot90(self.model["shadowing"], k=1, axes=(2, 3))

        return self


def calculate_all_distances(nfz):
    print("Computing all distances. This may take a while.")
    nfz = np.pad(nfz, pad_width=((1, 1), (1, 1)), constant_values=1)
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

    print("Done.")
    return min_distance


def calculate_landing_distances(nfz, slz, init=None):
    nfz = np.pad(nfz, pad_width=((1, 1), (1, 1)), constant_values=1)
    slz = np.pad(slz, pad_width=((1, 1), (1, 1)), constant_values=0)
    if init is not None:
        init = np.where(init == -1, np.inf, init.astype(float))
        init = np.pad(init, pad_width=((1, 1), (1, 1)), constant_values=np.inf)
    else:
        init = np.inf

    min_distance = np.where(slz, 1, init)
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


def calculate_shadowing(obst):
    size = obst.shape[0]

    total_shadow_map = np.ones((size, size, size, size), dtype=bool)
    for i, j in np.ndindex(obst.shape):
        if obst[i, j]:
            continue
        shadow_map = np.ones((size, size), dtype=bool)

        for x in range(size):
            bresenham(i, j, x, 0, obst, shadow_map)
            bresenham(i, j, x, size - 1, obst, shadow_map)
            bresenham(i, j, 0, x, obst, shadow_map)
            bresenham(i, j, size - 1, x, obst, shadow_map)

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


def is_solvable(landing_map, obstacles, visibility_map, max_budget):
    max_steps = max_budget // 2 - 1

    n, m = landing_map.shape

    lm = landing_map
    vm = visibility_map
    obs = obstacles[:n, :m]

    lm = np.where(lm < 0, max_steps + 1, lm)

    reach = lm <= max_steps

    non_reach = np.logical_not(np.logical_or(reach, obs))
    if not np.any(non_reach):
        return True

    nr = np.transpose(np.stack(np.nonzero(non_reach), axis=0))
    for x, y in nr:
        if not np.any(np.logical_and(reach, vm[x, y])):
            return False

    return True


def generate_map(map_generation_params, camera_half_length, max_budget):
    r = map_generation_params.size_range
    size = np.random.randint(r[0], r[1] + 1)

    m = np.zeros((size, size, 3), dtype=bool)
    filled = np.zeros((size, size), dtype=bool)

    adj_free_filled = np.zeros_like(filled, dtype=int)
    adj_low_filled = np.zeros_like(filled, dtype=int)
    adj_filled = np.zeros_like(filled, dtype=int)

    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    values = ([0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1])

    initial = np.random.randint(size, size=2)
    filled[initial[0], initial[1]] = True

    x, y = adj_indices(initial, size)
    adj_filled[x, y] += 1
    adj_free_filled[x, y] += 1

    while not np.all(filled):

        prob = (~filled).astype(int) * (
                adj_filled.astype(int) + map_generation_params.prio_free_adj * adj_free_filled.astype(int))
        prob_norm = prob / np.sum(prob)

        prob_flat = np.reshape(prob_norm, (-1,))
        idx = np.random.choice(range(len(prob_flat)), p=prob_flat)
        x, y = np.unravel_index(idx, shape=prob_norm.shape)

        if adj_low_filled[x, y]:
            p = map_generation_params.p_adj_low
        elif adj_free_filled[x, y]:
            p = map_generation_params.p_adj_free
        else:
            p = (0, 0, 0, 1.0)

        v_idx = np.random.choice(range(len(values)), p=p)
        filled[x, y] = True
        m[x, y] = values[v_idx]

        x, y = adj_indices((x, y), size)
        adj_filled[x, y] += 1
        if v_idx == 0:
            adj_free_filled[x, y] += 1
        elif v_idx == 2:
            adj_low_filled[x, y] += 1

    # prune single cells
    occupied = np.any(m, axis=-1)
    alone = convolve(occupied.astype(int), kernel, mode='reflect') == 1
    remove = np.random.uniform(size=(size, size))
    m = np.where((alone & (remove < map_generation_params.single_cell_prune))[:, :, None], False, m)

    # Place landing zones
    free = ~np.any(m, axis=-1)
    free_idx = np.array(np.nonzero(free))

    i = np.random.choice(range(free_idx.shape[1]))
    landing_cell = free_idx[:, i]
    m[landing_cell[0], landing_cell[1], 0] = True

    map_data = m[:, :, [1, 2, 0]]

    map_ = Map(map_data=map_data, name="gen")

    visibility_map = get_visibility_map(camera_half_length, shadowing_map=map_.shadowing())
    while not is_solvable(map_.landing_distances(), map_.obst, visibility_map, max_budget):
        distances = map_.landing_distances()
        candidates = (distances <= max_budget - 2) & free
        candidates = np.array(np.nonzero(candidates))
        i = np.random.choice(range(candidates.shape[1]))
        landing_cell = candidates[:, i]
        map_.map_data[landing_cell[0], landing_cell[1], 2] = True
        map_.model["landing"] = calculate_landing_distances(map_.nfz, map_.slz, map_.model["landing"])

    map_.pad_to_size((map_generation_params.size_range[1], map_generation_params.size_range[1]))

    return map_


def adj_indices(cell, size):
    x, y = cell
    indices = []
    if x < size - 1:
        indices.append([x + 1, y])
    if x > 0:
        indices.append([x - 1, y])
    if y < size - 1:
        indices.append([x, y + 1])
    if y > 0:
        indices.append([x, y - 1])
    return zip(*indices)


def queue_maps(map_generation_params, camera_half_length, max_budget, map_queue, stop_event):
    while not stop_event.is_set():
        try:
            map_ = generate_map(map_generation_params, camera_half_length, max_budget)
            if not stop_event.is_set():
                map_queue.put(map_, timeout=1)
            else:
                break
        except Full:
            sleep(1)
