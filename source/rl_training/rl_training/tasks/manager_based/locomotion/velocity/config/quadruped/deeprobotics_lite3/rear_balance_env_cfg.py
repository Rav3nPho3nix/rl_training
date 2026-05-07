# Equilibre sur les pattes arrieres
# Crée par Giuliano CAPITANO
# Basé sur rl_training de DEEP Robotics

from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg

from isaaclab.managers import SceneEntityCfg, RewardTermCfg
from isaaclab.envs import ManagerBasedRLEnv, mdp

from rl_training.tasks.manager_based.locomotion.velocity import mdp

# V0 : reste sur place
@configclass
class CustomLite3RearBalanceEnvCfg(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Commandes : pas de déplacement, juste équilibre
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # --- Ajustement des récompenses EXISTANTES ---
        # 1. Hauteur du centre de masse (déjà définie, on ajuste poids + params)
        self.rewards.base_height_l2.weight = 10.0
        self.rewards.base_height_l2.params["target_height"] = 0.55
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # 2. Pénalise la vitesse verticale négative (chute) (déjà définie)
        self.rewards.lin_vel_z_l2.weight = -8.0

        # 3. Pénalise l'orientation non plate (on la désactive, car on veut une inclinaison)
        self.rewards.flat_orientation_l2 = None

        # 4. Pénalise la vitesse angulaire XY (déjà définie)
        self.rewards.ang_vel_xy_l2.weight = -5.0

        # --- NOUVELLES récompenses pour l'équilibre arrière ---
        # 5. Récompense l'inclinaison vers l'arrière (pitch négatif)
        self.rewards.target_orientation = RewardTermCfg(
            func=mdp.target_orientation_reward,
            weight=15.0,
            params={
                "target_pitch": -0.3,  # ~-17° (penché vers l'arrière)
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # 6. Pénalise fortement les contacts des pattes avant
        self.rewards.penalize_front_contacts = RewardTermCfg(
            func=mdp.penalize_front_contacts,
            weight=-50.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
                "front_leg_indices": [0, 1],  # Patte avant gauche/droite
            },
        )

        # 7. Récompense si les pattes arrière sont en contact
        self.rewards.rear_legs_contact = RewardTermCfg(
            func=mdp.rear_legs_contact_reward,
            weight=10.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
                "rear_leg_indices": [2, 3],  # Patte arrière gauche/droite
            },
        )

        # 8. Pénalise les mouvements latéraux ou vers l'avant
        self.rewards.penalize_lateral_and_forward = RewardTermCfg(
            func=mdp.penalize_lateral_and_forward,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # 9. Récompense la stabilité globale
        self.rewards.stability = RewardTermCfg(
            func=mdp.stability_reward,
            weight=5.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Désactive les récompenses inutiles
        self.rewards.forward_distance = None
        self.rewards.all_feet_airborne = None
        self.rewards.lin_vel_z_positive = None
        self.rewards.penalize_folded_legs = None
        self.rewards.penalize_backward_and_lateral = None
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.feet_air_time_variance = None
        self.rewards.stand_still = None
        self.rewards.feet_height = None
        self.rewards.feet_height_body = None
        self.rewards.wheel_vel_penalty = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.feet_contact = None
        self.rewards.flat_orientation_l2 = None

        if self.__class__.__name__ == "CustomLite3RearBalanceEnvCfg":
            self.disable_zero_weight_rewards()

# v1 : se redresse mais recule un tout petit peu a chaque fois et replie les pattes avants

@configclass
class CustomLite3RearBalanceEnvCfg_1(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.episode_length_s = 10.0

        # ── Signal principal : pattes arrières au sol, avant en l'air ──
        self.rewards.rear_legs_only_contact = RewardTermCfg(
            func=mdp.rear_legs_contact_reward,
            weight=20.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
                ),
                "rear_leg_indices": [2,3],
            },
        )

        # ── Corps dressé verticalement — signal clé ─────────────────────
        # Remplace upright_orientation_reward qui ne distinguait pas
        # "dressé" de "à plat sur le dos"
        self.rewards.body_upright_reward = RewardTermCfg(
            func=mdp.body_upright_reward,
            weight=25.0,  # dominant
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ── Pénaliser le contact des pattes avant ───────────────────────
        self.rewards.penalize_front_contact = RewardTermCfg(
            func=mdp.penalize_front_contacts,
            weight=-15.0,  # augmenté
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
                ),
                "front_leg_indices": [0, 1],
            },
        )

        # ── Pénaliser les genoux au sol ─────────────────────────────────
        # Le robot s'agenouille → pénaliser les contacts non-pieds
        self.rewards.undesired_contacts.weight = -5.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            "FL_SHANK", "FR_SHANK", "HL_SHANK", "HR_SHANK",
            "FL_THIGH", "FR_THIGH", "HL_THIGH", "HR_THIGH",
            "TORSO"
        ]

        # ── Désactiver les récompenses incompatibles ────────────────────
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        # Désactivé — on veut l'inverse de plat
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = -0.01

        # ── Incompatibles Lite3 ─────────────────────────────────────────
        self.rewards.wheel_vel_penalty = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.feet_contact = None

        if self.__class__.__name__ == "CustomLite3RearBalanceEnvCfg_1":
            self.disable_zero_weight_rewards()


# V3

@configclass
class CustomLite3RearBalanceEnvCfg_2(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.episode_length_s = 10.0

        # ── Signal principal : pattes arrières au sol, avant en l'air ──
        self.rewards.rear_legs_only_contact = RewardTermCfg(
            func=mdp.rear_legs_contact_reward,
            weight=20.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
                ),
                "rear_leg_indices": [2,3],
            },
        )

        # ── Corps dressé verticalement — signal clé ─────────────────────
        # Remplace upright_orientation_reward qui ne distinguait pas
        # "dressé" de "à plat sur le dos"
        self.rewards.body_upright_reward = RewardTermCfg(
            func=mdp.body_upright_reward,
            weight=30.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ── Pénaliser le contact des pattes avant ───────────────────────
        self.rewards.penalize_front_contact = RewardTermCfg(
            func=mdp.penalize_front_contacts,
            weight=-15.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
                ),
                "front_leg_indices": [0, 1],
            },
        )

        # ── Pénaliser les genoux au sol ─────────────────────────────────
        # Le robot s'agenouille → pénaliser les contacts non-pieds
        self.rewards.undesired_contacts.weight = -5.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            "FL_SHANK", "FR_SHANK", "HL_SHANK", "HR_SHANK",
            "FL_THIGH", "FR_THIGH", "HL_THIGH", "HR_THIGH",
            "TORSO"
        ]

        self.rewards.penalize_backward_motion = RewardTermCfg(
            func=mdp.penalize_backward_motion,
            weight=-20.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.rewards.front_leg_fold_penalty = RewardTermCfg(
            func=mdp.front_leg_fold_penalty,
            weight=-15.0,
            params={
                "front_knee_joint_ids": [2, 5],  # adapte selon ton robot
                "threshold": 1.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.rewards.stability = RewardTermCfg(
            func=mdp.stability_reward,
            weight=5.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ── Désactiver les récompenses incompatibles ────────────────────
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        # Désactivé — on veut l'inverse de plat
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = -0.01

        # ── Incompatibles Lite3 ─────────────────────────────────────────
        self.rewards.wheel_vel_penalty = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.feet_contact = None

        if self.__class__.__name__ == "CustomLite3RearBalanceEnvCfg_2":
            self.disable_zero_weight_rewards()