import io
import tensorflow as tf
from tqdm import tqdm

from src.Map.Map import Map
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb
from matplotlib import patches
import cv2

from PIL import Image


class BaseDisplay:
    def __init__(self):
        self.arrow_scale = 14
        self.marker_size = 15

    def create_grid_image(self, ax, env_map: Map, value_map, green=None):
        area_y_max, area_x_max = env_map.get_size()

        if green is None:
            green = np.zeros((area_y_max, area_x_max))

        nfz = np.expand_dims(env_map.nfz, -1)
        lz = np.expand_dims(env_map.start_land_zone, -1)
        green = np.expand_dims(green, -1)

        neither = np.logical_not(np.logical_or(np.logical_or(nfz, lz), green))

        base = np.zeros((area_y_max, area_x_max, 3))

        nfz_color = base.copy()
        nfz_color[..., 0] = 0.8

        lz_color = base.copy()
        lz_color[..., 2] = 0.8

        green_color = base.copy()
        green_color[..., 1] = 0.8

        neither_color = np.ones((area_y_max, area_x_max, 3), dtype=np.float)
        grid_image = green_color * green + nfz_color * nfz + lz_color * lz + neither_color * neither

        hsv_image = rgb2hsv(grid_image)
        hsv_image[..., 2] *= value_map.astype('float32')

        grid_image = hsv2rgb(hsv_image)

        if (area_x_max, area_y_max) == (64, 64):
            tick_labels_x = np.arange(0, area_x_max, 4)
            tick_labels_y = np.arange(0, area_y_max, 4)
            self.arrow_scale = 14
            self.marker_size = 6
        elif (area_x_max, area_y_max) == (32, 32):
            tick_labels_x = np.arange(0, area_x_max, 2)
            tick_labels_y = np.arange(0, area_y_max, 2)
            self.arrow_scale = 8
            self.marker_size = 15
        elif (area_x_max, area_y_max) == (50, 50):
            tick_labels_x = np.arange(0, area_x_max, 4)
            tick_labels_y = np.arange(0, area_y_max, 4)
            self.arrow_scale = 12
            self.marker_size = 8
        else:
            tick_labels_x = np.arange(0, area_x_max, 1)
            tick_labels_y = np.arange(0, area_y_max, 1)
            self.arrow_scale = 5
            self.marker_size = 15

        plt.sca(ax)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks(tick_labels_x)
        plt.yticks(tick_labels_y)
        plt.axis([0, area_x_max, area_y_max, 0])
        ax.imshow(grid_image.astype(float), extent=[0, area_x_max, area_y_max, 0])
        # plt.axis('off')

        obst = env_map.obstacles
        for i in range(area_x_max):
            for j in range(area_y_max):
                if obst[j, i]:
                    rect = patches.Rectangle((i, j), 1, 1, fill=None, hatch='////', edgecolor="Black")
                    ax.add_patch(rect)

        # offset to shift tick labels
        locs, labels = plt.xticks()
        locs_new = [x + 0.5 for x in locs]
        plt.xticks(locs_new, tick_labels_x)

        locs, labels = plt.yticks()
        locs_new = [x + 0.5 for x in locs]
        plt.yticks(locs_new, tick_labels_y)

    def draw_start_and_end(self, trajectory):
        first_state = trajectory[0][0]
        final_state = trajectory[-1][3]

        plt.scatter(first_state.position[0] + 0.5, first_state.position[1] + 0.5, s=self.marker_size, marker="D",
                    color="w")

        if final_state.landed:
            plt.scatter(final_state.position[0] + 0.5, final_state.position[1] + 0.5,
                        s=self.marker_size, marker="D", color="green")
        else:
            plt.scatter(final_state.position[0] + 0.5, final_state.position[1] + 0.5,
                        s=self.marker_size, marker="D", color="r")

    def draw_movement(self, from_position, to_position, color):
        y = from_position[1]
        x = from_position[0]
        dir_y = to_position[1] - y
        dir_x = to_position[0] - x
        if dir_x == 0 and dir_y == 0:
            plt.scatter(x + 0.5, y + 0.5, marker="X", color=color)
        else:
            if abs(dir_x) >= 1 or abs(dir_y) >= 1:
                plt.quiver(x + 0.5, y + 0.5, dir_x, -dir_y, color=color,
                           scale=self.arrow_scale, scale_units='inches')
            else:
                plt.quiver(x + 0.5, y + 0.5, dir_x, -dir_y, color=color,
                           scale=self.arrow_scale, scale_units='inches')

    def create_tf_image(self):
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=180, bbox_inches='tight')
        buf.seek(0)
        plt.close('all')
        combined_image = tf.image.decode_png(buf.getvalue(), channels=3)
        return tf.expand_dims(combined_image, 0)

    def create_video(self, map_image, trajectory, save_path, frame_rate, draw_path=True):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = None
        print("Creating video, if it freezes tap ctrl...")
        for k in tqdm(range(len(trajectory))):
            if draw_path:
                frame = np.squeeze(self.display_episode(map_image, trajectory[:k + 1]).numpy())
            else:
                frame = np.squeeze(self.display_state(map_image, trajectory[0][0], trajectory[k][0]).numpy())
            r = frame[..., 0]
            g = frame[..., 1]
            b = frame[..., 2]
            img = np.stack([b, g, r], axis=2)
            if not video:
                height, width, channels = img.shape
                print("Creating video writer for {}, with the shape {}".format(save_path, (height, width)))
                video = cv2.VideoWriter(save_path, fourcc, frame_rate, (width, height))
            video.write(img)

        if not draw_path:
            frame = np.squeeze(self.display_episode(map_image, trajectory).numpy())
            r = frame[..., 0]
            g = frame[..., 1]
            b = frame[..., 2]
            img = np.stack([b, g, r], axis=2)
            video.write(img)

        video.release()

    def display_episode(self, map_image, trajectory, plot=False, save_path=None) -> tf.Tensor:
        pass

    def display_state(self, env_map, initial_state, state, plot=False) -> tf.Tensor:
        pass
