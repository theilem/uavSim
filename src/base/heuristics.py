from dataclasses import dataclass

import numpy as np


class GreedyHeuristic:
    @dataclass
    class Params:
        pass

    def __init__(self, gym):
        self.gym = gym
        self.max_budget = self.gym.max_budget

    def get_action(self, state=None):
        if state is None:
            state = self.gym.state
        x, y = self.gym.map_image(state).original_shape

        position = state.position
        # Charge fully
        if state.landed:
            if state.budget >= self.max_budget:
                return np.array((5,))  # Take off action
            return np.array((6,))  # Charging action
        # Find the closest target cell from which the landing zone can be reached in time
        budget = state.budget
        landing = self.gym.map_image(state).landing_distances()
        distance_map = self.gym.map_image(state).all_distances()
        distances = distance_map[position[0], position[1]]
        targets = state.map[:x, :y, 3]
        # Calculate all the ways a target under nfzs can be seen
        reachable = np.logical_and(landing + distances < budget, distances > 0)
        target_idx = np.array(np.where(np.logical_and(targets, np.logical_not(reachable)))).transpose()
        for idx in target_idx:
            targets = np.logical_or(targets, self.gym.camera(state).visibility_map[:, :, idx[0], idx[1]])
        # Get all relevant cells, reachable and target
        relevant = np.logical_and(reachable, targets)
        if not np.any(relevant):
            if not np.any(targets):
                if state.map[position[0], position[1], 0]:
                    return np.array((4,))
                return self.shortest_path_action(landing, position)

            # Seems like there is no directly reachable target. Find landing zone that is closer.
            reachable_landings = np.logical_and(np.logical_and(distances < budget, distances >= 0),
                                                self.gym.map_image(state).slz)

            landing_idx = np.array(np.nonzero(reachable_landings))
            target_idx = np.array(np.nonzero(targets))

            landing_idx_rep = np.repeat(landing_idx, target_idx.shape[1], axis=1)
            target_idx_rep = np.reshape(np.repeat(np.expand_dims(target_idx, 1), landing_idx.shape[1], axis=1), (2, -1))

            dist_to_target = distance_map[landing_idx_rep[0], landing_idx_rep[1], target_idx_rep[0], target_idx_rep[1]]
            dist_to_target = np.where(dist_to_target == -1, 100000, dist_to_target)
            argmin = np.argmin(dist_to_target)
            goal = landing_idx_rep[:, argmin]
            if np.all(position == goal):
                return np.array((4,))

            goal_ = distance_map[:, :, goal[0], goal[1]]
            return self.shortest_path_action(goal_, position)

        distances = np.where(relevant, distances, np.inf)
        distances = np.where(distances == -1, np.inf, distances)
        goal = np.unravel_index(distances.argmin(), distances.shape)
        goal_ = distance_map[:, :, goal[0], goal[1]]
        return self.shortest_path_action(goal_, position)

    @staticmethod
    def shortest_path_action(distance_map, position):
        shortest_distance = np.pad(distance_map,
                                   ((1, 1), (1, 1)), constant_values=1000)
        shortest_distance = np.where(shortest_distance == -1, 1000, shortest_distance)
        query = position + 1  # Padding compensation
        dist_up = shortest_distance[query[0], query[1] + 1]
        dist_down = shortest_distance[query[0], query[1] - 1]
        dist_left = shortest_distance[query[0] - 1, query[1]]
        dist_right = shortest_distance[query[0] + 1, query[1]]

        distances = [dist_right, dist_up, dist_left, dist_down]
        actions = np.where(distances == np.min(distances))[0]
        return np.array((actions[0],))
