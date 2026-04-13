# Saut long
# Crée par Giuliano CAPITANO
# Basé sur rl_training de DEEP Robotics

from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg

@configclass
class CustomLite3LongJumpEnvCfg(DeeproboticsLite3RoughEnvCfg):
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
            - Droit
        """

        self.rewards.track_lin_vel_xy_exp.weight = 1.5  # réduit vs 3.0, secondaire par rapport au saut
        # On ne souhaite pas de rotation du l'axe Z
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        # Hauteur
        self.rewards.base_height_l2.weight = 10.0
        self.rewards.base_height_l2.params["target_height"] = 0.55
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Impulsion du saut
        self.rewards.lin_vel_z_l2.weight = 2.0

        # Stabilité
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        
        # Contacts avec le sol
        self.rewards.undesired_contacts.weight = -0.5
        self.rewards.feet_slide.weight = -0.05

        # Rester droit sur le saut
        self.rewards.joint_deviation_l1.weight = -0.2

        # A desactiver pour le saut
        # Rotation sur Z
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        # Pas de déplacement au trot (pas de sens)
        self.rewards.feet_gait.weight = 0.0 
        # Pas de cycle sur le mouvement (on ne saute qu'une fois)
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        # On ne souhaite pas que le robot reste figé
        self.rewards.stand_still.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_height.weight = 0.0

        # Vitesse vers l'avant pour avancer pendnt le saut
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 2.0)
        # MAis on ne veux pas tourner sur le côté
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # On reste tout droit
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg":
            self.disable_zero_weight_rewards()