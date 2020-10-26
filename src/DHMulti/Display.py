import numpy as np
import matplotlib.pyplot as plt
import io
import tensorflow as tf
import matplotlib.patches as patches

from src.DH.Display import DHDisplay
from src.Map.Map import Map
from src.base.BaseDisplay import BaseDisplay


class DHMultiDisplay(DHDisplay):

    def __init__(self):
        super().__init__()

    def draw_start_and_end(self, trajectory):
        for exp in trajectory:
            state, action, reward, next_state = exp

            # Identify first moves
            if state.movement_budget == state.initial_movement_budget:
                plt.scatter(state.position[0] + 0.5, state.position[1] + 0.5, s=self.marker_size, marker="D",
                            color="w")

            # Identify last moves
            if next_state.terminal:
                if next_state.landed:
                    plt.scatter(next_state.position[0] + 0.5, next_state.position[1] + 0.5,
                                s=self.marker_size, marker="D", color="green")
                else:
                    plt.scatter(next_state.position[0] + 0.5, next_state.position[1] + 0.5,
                                s=self.marker_size, marker="D", color="r")

    def display_episode(self, env_map: Map, trajectory, plot=False, save_path=None):

        first_state = trajectory[0][0]
        final_state = trajectory[-1][3]

        fig_size = 5.5
        fig, ax = plt.subplots(1, 2, figsize=[2 * fig_size, fig_size])
        ax_traj = ax[0]
        ax_bar = ax[1]

        value_step = 0.4 / first_state.device_list.num_devices
        # Start with value of 200
        value_map = np.ones(env_map.get_size(), dtype=float)
        for device in first_state.device_list.get_devices():
            value_map -= value_step * self.channel.total_shadow_map[device.position[1], device.position[0]]

        self.create_grid_image(ax=ax_traj, env_map=env_map, value_map=value_map)

        for device in first_state.device_list.get_devices():
            ax_traj.add_patch(
                patches.Circle(np.array(device.position) + np.array((0.5, 0.5)), 0.4, facecolor=device.color,
                               edgecolor="black"))

        self.draw_start_and_end(trajectory)

        for exp in trajectory:
            idx = exp[3].device_coms[exp[0].active_agent]
            if idx == -1:
                color = "black"
            else:
                color = exp[0].device_list.devices[idx].color

            self.draw_movement(exp[0].position, exp[3].position, color=color)

        # Add bar plots
        self.draw_bar_plots(final_state, ax_bar)

        # save image and return
        if save_path is not None:
            # save just the trajectory subplot 0
            extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            extent.x0 -= 0.3
            extent.y0 -= 0.1
            fig.savefig(save_path, bbox_inches=extent,
                        format='png', dpi=300, pad_inches=1)
        if plot:
            plt.show()

        return self.create_tf_image()
