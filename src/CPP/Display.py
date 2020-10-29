import numpy as np
import matplotlib.pyplot as plt
from src.Map.Map import Map
from src.base.BaseDisplay import BaseDisplay


class CPPDisplay(BaseDisplay):

    def __init__(self):
        super().__init__()

    def display_episode(self, env_map: Map, trajectory, plot=False, save_path=None):

        first_state = trajectory[0][0]
        final_state = trajectory[-1][3]

        fig_size = 5.5
        fig, ax = plt.subplots(1, 1, figsize=[fig_size, fig_size])
        value_map = final_state.coverage * 1.0 + (~final_state.coverage) * 0.75

        self.create_grid_image(ax=ax, env_map=env_map, value_map=value_map, green=first_state.target)

        self.draw_start_and_end(trajectory)

        # plot trajectory arrows
        for exp in trajectory:
            self.draw_movement(exp[0].position, exp[3].position, color="black")

        # save image and return
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight',
                        format='png', dpi=300)
        if plot:
            plt.show()

        return self.create_tf_image()

    def display_state(self, env_map, initial_state, state, plot=False):
        fig_size = 5.5
        fig, ax = plt.subplots(1, 1, figsize=[fig_size, fig_size])
        value_map = state.coverage * 1.0 + (~state.coverage) * 0.75

        self.create_grid_image(ax=ax, env_map=env_map, value_map=value_map, green=initial_state.target)

        color = "green" if state.landed else "r"
        plt.scatter(state.position[0] + 0.5, state.position[1] + 0.5,
                    s=self.marker_size, marker="D", color=color)

        if plot:
            plt.show()

        return self.create_tf_image()

    def draw_map(self, map_in):
        rgb = map_in[0, :, :, :3]
        rgb = np.stack([rgb[:, :, 0], rgb[:, :, 2], rgb[:, :, 1]], axis=2)
        plt.imshow(rgb.astype(float))
        plt.show()

    def draw_maps(self, total_map, global_map, local_map):
        fig, ax = plt.subplots(1, 3)
        rgb = total_map[0, :, :, :3]
        rgb = np.stack([rgb[:, :, 0], rgb[:, :, 2], rgb[:, :, 1]], axis=2)
        ax[0].imshow(rgb.astype(float))

        rgb = global_map[0, :, :, :3]
        rgb = np.stack([rgb[:, :, 0], rgb[:, :, 2], rgb[:, :, 1]], axis=2)
        ax[1].imshow(rgb.astype(float))

        rgb = local_map[0, :, :, :3]
        rgb = np.stack([rgb[:, :, 0], rgb[:, :, 2], rgb[:, :, 1]], axis=2)
        ax[2].imshow(rgb.astype(float))
        plt.show()
