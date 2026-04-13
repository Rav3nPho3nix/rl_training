# Saut long
# Crée par Giuliano CAPITANO
# Basé sur rl_training de DEEP Robotics

from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg

@configclass
class DeeproboticsLite3FlatEnvCfg(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        # Appel du post init du parent
        super().__post_init__()

        # Terrain plat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        """
        Pour un long saut :
            - Hauteur
            - Distance
            - Stabilité
            - Sur le bon côté
        """

        if self.__class__.__name__ == "DeeproboticsLite3LongJumpEnvCfg":
            self.disable_zero_weight_rewards()