import numpy as np
from skimage.draw import random_shapes
import logging


class RandomTargetGeneratorParams:
    def __init__(self):
        self.coverage_range = (0.2, 0.8)
        self.shape_range = (1, 5)


class RandomTargetGenerator:

    def __init__(self, params: RandomTargetGeneratorParams, shape):
        self.params = params
        self.shape = shape

    def generate_target(self, obstacles):

        area = np.product(self.shape)

        target = self.__generate_random_shapes_area(
            self.params.shape_range[0],
            self.params.shape_range[1],
            area * self.params.coverage_range[0],
            area * self.params.coverage_range[1]
        )

        return target & ~obstacles

    def __generate_random_shapes(self, min_shapes, max_shapes):
        img, _ = random_shapes(self.shape, max_shapes, min_shapes=min_shapes, multichannel=False,
                               allow_overlap=True, random_seed=np.random.randint(2**32 - 1))
        # Numpy random usage for random seed unifies random seed which can be set for repeatability
        attempt = np.array(img != 255, dtype=bool)
        return attempt, np.sum(attempt)

    def __generate_random_shapes_area(self, min_shapes, max_shapes, min_area, max_area, retry=100):
        for attemptno in range(retry):
            attempt, area = self.__generate_random_shapes(min_shapes, max_shapes)
            if min_area is not None and min_area > area:
                continue
            if max_area is not None and max_area < area:
                continue
            return attempt
        print("Here")
        logging.warning("Was not able to generate shapes with given area constraint in allowed number of tries."
                        " Randomly returning next attempt.")
        attempt, area = self.__generate_random_shapes(min_shapes, max_shapes)
        logging.warning("Size is: ", area)
        return attempt

    def __generate_exclusive_shapes(self, exclusion, min_shapes, max_shapes):
        attempt, area = self.__generate_random_shapes(min_shapes, max_shapes)
        attempt = attempt & (~exclusion)
        area = np.sum(attempt)
        return attempt, area

    # Create target image and then subtract exclusion area
    def __generate_exclusive_shapes_area(self, exclusion, min_shapes, max_shapes, min_area, max_area, retry=100):
        for attemptno in range(retry):
            attempt, area = self.__generate_exclusive_shapes(exclusion, min_shapes, max_shapes)
            if min_area is not None and min_area > area:
                continue
            if max_area is not None and max_area < area:
                continue
            return attempt

        logging.warning("Was not able to generate shapes with given area constraint in allowed number of tries."
                        " Randomly returning next attempt.")
        attempt, area = self.__generate_exclusive_shapes(exclusion, min_shapes, max_shapes)
        logging.warning("Size is: ", area)
        return attempt
