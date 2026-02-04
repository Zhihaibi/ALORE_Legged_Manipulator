# from omni.isaac.lab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
import env.terrain as terrain
# from omni.isaac.lab.utils import configclass
from dataclasses import MISSING

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.utils import configclass

@configclass
class HfUniformDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a discrete obstacles height field terrain."""

    function = terrain.uniform_discrete_obstacles_terrain
    seed: float = 0
    """Env seed."""
    obstacle_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the obstacles (in m)."""
    obstacle_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the obstacles (in m)."""
    num_obstacles: int = MISSING
    """The number of obstacles to generate."""
    obstacles_distance: float = MISSING
    """The minimum distance between obstacles (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    avoid_positions: list[list[float, float]] = []