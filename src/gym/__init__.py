from typing import Tuple

from src.gym.cpp import CPPGym
from src.gym.grid import GridGym
from utils import Factory


class PathPlanningGymFactory(Factory):
    @classmethod
    def registry(cls):
        return {
            "grid": GridGym,
            "cpp": CPPGym,
        }

    @classmethod
    def defaults(cls) -> Tuple[str, type]:
        return "cpp", CPPGym
