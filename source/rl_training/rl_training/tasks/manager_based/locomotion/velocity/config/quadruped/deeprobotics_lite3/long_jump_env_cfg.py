# Saut long
# Crée par Giuliano CAPITANO
# Basé sur rl_training de DEEP Robotics

from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg

from isaaclab.managers import SceneEntityCfg, RewardTermCfg
from isaaclab.envs import ManagerBasedRLEnv, mdp

from rl_training.tasks.manager_based.locomotion.velocity import mdp

import torch

# V0

@configclass
class CustomLite3LongJumpEnvCfg(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        # Appel du post init du parent
        super().__post_init__()

        # Terrain plat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.command_levels = None
        
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


# V1

@configclass
class CustomLite3LongJumpEnvCfg_1(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ── Terrain plat ───────────────────────────────────────────────
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.command_levels = None
        self.curriculum.terrain_levels = None

        # ── Commandes : saut vers l'avant uniquement ───────────────────
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 2.5)  # parent: (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)  # parent: (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)  # parent: (-0.8, 0.8)

        # ── Récompenses positives ──────────────────────────────────────

        # Signal dominant : vitesse horizontale vers l'avant
        # parent: 3.0 — on monte à 5.0 pour dominer toutes les autres
        self.rewards.track_lin_vel_xy_exp.weight = 5.0

        # Temps en l'air : signal de décollage
        # parent: 5.0 — on garde mais on ajuste le threshold pour un vrai saut
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["threshold"] = 0.3  # parent: 0.5

        # ── Récompenses à désactiver ───────────────────────────────────

        # Hauteur cible : c'est lui qui causait le saut sur place
        # parent: -10.0 — on coupe complètement
        self.rewards.base_height_l2.weight = 0.0

        # Vitesse verticale : parent pénalise toute vz, nuit au décollage
        self.rewards.lin_vel_z_l2.weight = 0.0

        # Pas de rotation sur Z (déjà bien géré par la commande ang_vel_z=0)
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        # Pas de gait trot (incompatible avec un saut)
        self.rewards.feet_gait.weight = 0.0

        # Pas de variance de air_time (on veut un saut groupé)
        self.rewards.feet_air_time_variance.weight = 0.0

        # Pas de stand_still (on veut que le robot bouge)
        self.rewards.stand_still.weight = 0.0

        # ── Pénalités assouplies pour tolérer le vol ───────────────────

        # parent: -5.0 — trop sévère en l'air, le robot tourne naturellement
        self.rewards.flat_orientation_l2.weight = -1.0

        # parent: -0.05 — légèrement réduit pour tolérer les mouvements en vol
        self.rewards.ang_vel_xy_l2.weight = -0.02

        # parent: -0.5 — relâché, la symétrie parfaite est impossible en saut
        self.rewards.joint_deviation_l1.weight = -0.1

        if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_1":
            self.disable_zero_weight_rewards()


# V2

def jump_distance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Récompense la distance maximale parcourue depuis l'origine.
    Signal principal pour le saut en longueur.
    """
    asset = env.scene[asset_cfg.name]
    distance = torch.norm(
        asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], 
        dim=1
    )
    return distance


def landing_stability_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Récompense la stabilité après atterrissage :
    - toutes les pattes au sol
    - vitesse faible
    - orientation plate
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_cfg.name]

    # Vérifie si toutes les pattes touchent le sol
    feet_contact = contact_sensor.data.net_forces_w_history[:, :, :, 2]  # force Z
    all_feet_grounded = (feet_contact[:, 0, :] > 1.0).all(dim=-1)  # toutes en contact

    # Vitesse linéaire faible = stabilisé
    lin_vel = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    is_slow = lin_vel < 0.3

    # Orientation plate
    flat = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1) < 0.1

    return (all_feet_grounded & is_slow & flat).float()


def flight_phase_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Récompense le fait d'être en l'air (toutes les pattes décollées).
    Encourage le vrai décollage plutôt que la course.
    """
    contact_sensor = env.scene[sensor_cfg.name]
    feet_contact = contact_sensor.data.net_forces_w_history[:, 0, :, 2]
    all_feet_in_air = (feet_contact < 0.5).all(dim=-1)
    return all_feet_in_air.float()

@configclass
class CustomLite3LongJumpEnvCfg_2(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ── Terrain plat ───────────────────────────────────────────────
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.command_levels = None
        self.curriculum.terrain_levels = None

        # ── Episode court : un seul saut ──────────────────────────────
        # ~3 secondes à 50Hz = 150 steps : élan + vol + atterrissage
        self.episode_length_s = 3.0

        # ── Commande : élan vers l'avant ──────────────────────────────
        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.rewards.body_lin_acc_l2.weight = -1e-4
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = ["TORSO"]

        # ══════════════════════════════════════════════════════════════
        # RÉCOMPENSES CUSTOM (à enregistrer dans RewardsCfg du parent)
        # ══════════════════════════════════════════════════════════════

        # [1] Distance parcourue — signal principal du saut en longueur
        self.rewards.jump_distance_reward = RewardTermCfg(
            func=jump_distance_reward,
            weight=10.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # [2] Phase de vol — toutes les pattes en l'air simultanément
        self.rewards.flight_phase_reward = RewardTermCfg(
            func=flight_phase_reward,
            weight=5.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"])},
        )

        # [3] Stabilisation après atterrissage
        self.rewards.landing_stability_reward = RewardTermCfg(
            func=landing_stability_reward,
            weight=3.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ── Synchronisation bound (avant/arrière) ──────────────────────
        self.rewards.feet_gait.weight = 2.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_FOOT", "FR_FOOT"],
            ["HL_FOOT", "HR_FOOT"],
        ]

        # ── Désactivés ─────────────────────────────────────────────────
        self.rewards.track_lin_vel_xy_exp.weight = 0.0  # remplacé par jump_distance
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0         # remplacé par flight_phase
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0

        # ── Pénalités allégées pour tolérer le vol ─────────────────────
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.ang_vel_xy_l2.weight = -0.02
        self.rewards.joint_deviation_l1.weight = -0.1
        self.rewards.feet_slide.weight = -0.3


        if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_2":
            self.disable_zero_weight_rewards()


# V3

def jump_forward_reward_3(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Récompense structurée en 3 phases :
    - Phase sol    : récompense la vitesse horizontale (élan)
    - Phase vol    : récompense la hauteur + distance
    - Phase atterr : récompense la stabilité
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_cfg.name]

    # Détection des contacts
    feet_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]
    n_feet_contact = (feet_forces > 1.0).sum(dim=-1).float()  # nb pattes au sol
    all_grounded = n_feet_contact == 4
    all_airborne = n_feet_contact == 0

    # Vitesse et hauteur
    lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    height = asset.data.root_pos_w[:, 2]
    base_height = 0.35  # hauteur normale debout

    # Distance depuis origine
    distance = torch.norm(
        asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1
    )

    # Orientation
    gravity_proj = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    lin_vel_norm = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

    # Phase 1 — élan : toutes pattes au sol, récompense vitesse vers l'avant
    phase_elan = all_grounded.float() * torch.clamp(lin_vel_x, min=0.0)

    # Phase 2 — vol : toutes pattes en l'air
    # récompense hauteur AU DESSUS de la normale + distance
    phase_vol = all_airborne.float() * (
        torch.clamp(height - base_height, min=0.0) * 5.0  # bonus hauteur
        + distance * 2.0                                    # bonus distance
    )

    # Phase 3 — atterrissage : toutes pattes au sol après avoir bougé
    is_stable = (gravity_proj < 0.15) & (lin_vel_norm < 0.5)
    phase_atterrissage = all_grounded.float() * is_stable.float() * distance

    return phase_elan + phase_vol + phase_atterrissage


def penalize_no_flight_3(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Pénalise le fait qu'aucune patte ne soit en l'air après
    que le robot a commencé à bouger — force le décollage.
    """
    contact_sensor = env.scene[sensor_cfg.name]
    feet_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]
    all_grounded = (feet_forces > 1.0).all(dim=-1)

    # Ne pénalise que si le robot a déjà bougé (évite de pénaliser le début)
    return all_grounded.float()


@configclass
class CustomLite3LongJumpEnvCfg_3(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ── Terrain plat ───────────────────────────────────────────────
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ── Episode court : 4 secondes = élan + vol + atterrissage ────
        self.episode_length_s = 4.0

        # ── Commandes ─────────────────────────────────────────────────
        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # ── Récompense principale : les 3 phases en une fonction ───────
        self.rewards.jump_forward_reward = RewardTermCfg(
            func=jump_forward_reward_3,
            weight=10.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[".*_FOOT"]
                ),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ── Pénalité si le robot reste au sol après l'élan ────────────
        self.rewards.penalize_no_flight = RewardTermCfg(
            func=penalize_no_flight_3,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[".*_FOOT"]
                ),
            },
        )

        # ── Synchronisation bound (avant/arrière) ─────────────────────
        self.rewards.feet_gait.weight = 2.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_FOOT", "FR_FOOT"],
            ["HL_FOOT", "HR_FOOT"],
        ]

        # ── Garder les anti-vibrations hérités ────────────────────────
        # action_rate_l2:   -0.02   :white_check_mark:
        # joint_torques_l2: -2.5e-5 :white_check_mark:
        # joint_acc_l2:     -1e-8   :white_check_mark:

        # ── Désactiver tout ce qui entre en conflit ────────────────────
        self.rewards.track_lin_vel_xy_exp.weight = 1.0  # faible, juste pour l'élan
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -0.3
        self.rewards.ang_vel_xy_l2.weight = -0.02
        self.rewards.joint_deviation_l1.weight = -0.1
        self.rewards.feet_slide.weight = -0.3

        # ── Termes incompatibles avec le Lite3 ────────────────────────
        self.rewards.wheel_vel_penalty = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.feet_contact = None

        if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_3":
            self.disable_zero_weight_rewards()

@configclass
class CustomLite3LongJumpEnvCfg_2(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ── Terrain plat ───────────────────────────────────────────────
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ── Episode court ──────────────────────────────────────────────
        self.episode_length_s = 4.0

        # ── Commandes ─────────────────────────────────────────────────
        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # ── Récompense principale ──────────────────────────────────────
        self.rewards.jump_forward_reward = RewardTermCfg(
            func=mdp.jump_forward_reward,
            weight=10.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ── Bound (avant-avant / arrière-arrière) ──────────────────────
        self.rewards.feet_gait.weight = 2.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_FOOT", "FR_FOOT"],
            ["HL_FOOT", "HR_FOOT"],
        ]

        # ── Faible signal d'élan directionnel ─────────────────────────
        self.rewards.track_lin_vel_xy_exp.weight = 1.0

        # ── Désactivés ─────────────────────────────────────────────────
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0

        # ── Pénalités allégées pour tolérer le vol ─────────────────────
        self.rewards.flat_orientation_l2.weight = -0.3
        self.rewards.ang_vel_xy_l2.weight = -0.02
        self.rewards.joint_deviation_l1.weight = -0.1
        self.rewards.feet_slide.weight = -0.3

        # ── Hérités sans changement ────────────────────────────────────
        # action_rate_l2:   -0.02   :white_check_mark:
        # joint_torques_l2: -2.5e-5 :white_check_mark:
        # joint_acc_l2:     -1e-8   :white_check_mark:
        # undesired_contacts: -0.5  :white_check_mark:

        # ── Incompatibles Lite3 ────────────────────────────────────────
        self.rewards.wheel_vel_penalty = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.feet_contact = None

        if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_2":
            self.disable_zero_weight_rewards()
            
# V3

@configclass
class CustomLite3LongJumpEnvCfg_3(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ── Terrain plat ───────────────────────────────────────────────
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ── Episode court ──────────────────────────────────────────────
        self.episode_length_s = 4.0

        # ── Commandes ─────────────────────────────────────────────────
        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # ── Récompense principale ──────────────────────────────────────
        self.rewards.jump_forward_reward = RewardTermCfg(
            func=mdp.jump_forward_reward,
            weight=10.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ── Bound (avant-avant / arrière-arrière) ──────────────────────
        self.rewards.feet_gait.weight = 2.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_FOOT", "FR_FOOT"],
            ["HL_FOOT", "HR_FOOT"],
        ]

        # ── Faible signal d'élan directionnel ─────────────────────────
        self.rewards.track_lin_vel_xy_exp.weight = 1.0

        # ── Désactivés ─────────────────────────────────────────────────
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0

        # ── Pénalités allégées pour tolérer le vol ─────────────────────
        self.rewards.flat_orientation_l2.weight = -0.3
        self.rewards.ang_vel_xy_l2.weight = -0.02
        self.rewards.joint_deviation_l1.weight = -0.1
        self.rewards.feet_slide.weight = -0.3

        # ── Hérités sans changement ────────────────────────────────────
        # action_rate_l2:   -0.02   :white_check_mark:
        # joint_torques_l2: -2.5e-5 :white_check_mark:
        # joint_acc_l2:     -1e-8   :white_check_mark:
        # undesired_contacts: -0.5  :white_check_mark:

        # ── Incompatibles Lite3 ────────────────────────────────────────
        self.rewards.wheel_vel_penalty = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.feet_contact = None

        if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_3":
            self.disable_zero_weight_rewards()

# V4

@configclass
class CustomLite3LongJumpEnvCfg_4(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.episode_length_s = 4.0

        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Signal principal : avancer sur X
        self.rewards.forward_distance_reward = RewardTermCfg(
            func=mdp.forward_distance_reward,
            weight=10.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Synchronisation bound avant/arrière
        self.rewards.feet_gait.weight = 3.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_FOOT", "FR_FOOT"],
            ["HL_FOOT", "HR_FOOT"],
        ]

        # Forcer le décollage
        self.rewards.feet_air_time.weight = 5.0
        self.rewards.feet_air_time.params["threshold"] = 0.2

        # Désactivés
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0

        # Pénalités allégées
        self.rewards.flat_orientation_l2.weight = -0.3
        self.rewards.ang_vel_xy_l2.weight = -0.02
        self.rewards.joint_deviation_l1.weight = -0.1
        self.rewards.feet_slide.weight = -0.3

        # Incompatibles Lite3
        self.rewards.wheel_vel_penalty = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.feet_contact = None

        if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_4":
            self.disable_zero_weight_rewards()

# V5
# Aller en avant avec minimum de synchronisation

@configclass
class CustomLite3LongJumpEnvCfg_5(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Seul vrai changement : bound au lieu du trot
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_FOOT", "FR_FOOT"],
            ["HL_FOOT", "HR_FOOT"],
        ]

        # Incompatibles Lite3
        self.rewards.wheel_vel_penalty = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.feet_contact = None

        if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_5":
            self.disable_zero_weight_rewards()

# V6
# Aller en avant avec jambes liees + decollage court

@configclass
class CustomLite3LongJumpEnvCfg_6(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # V1 & V3
        self.rewards.feet_gait.weight = 2.0
        
        # V2
        # self.rewards.feet_gait.weight = 8.0

        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_FOOT", "FR_FOOT"],
            ["HL_FOOT", "HR_FOOT"],
        ]

        # V1 & V3
        self.rewards.feet_air_time.weight = 8.0
        
        # V2
        # self.rewards.feet_air_time.weight = 2.0

        self.rewards.feet_air_time.params["threshold"] = 0.3

        # V1 & V2
        # self.rewards.flat_orientation_l2.weight = -0.5

        # V3
        self.rewards.flat_orientation_l2.weight = -5.0

        # Incompatibles Lite3
        self.rewards.wheel_vel_penalty = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.feet_contact = None

        if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_6":
            self.disable_zero_weight_rewards()

# V7
# Se pencher vers l'avant

@configclass
class CustomLite3LongJumpEnvCfg_7(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.rewards.feet_gait.weight = 2.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_FOOT", "FR_FOOT"],
            ["HL_FOOT", "HR_FOOT"],
        ]

        self.rewards.feet_air_time.weight = 10.0
        self.rewards.feet_air_time.params["threshold"] = 0.3

        self.rewards.lin_vel_z_positive = RewardTermCfg(
            func=mdp.lin_vel_z_positive,
            weight=5.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        self.rewards.lin_vel_z_l2.weight = 0.0

        self.rewards.flat_orientation_l2.weight = -5.0

        # Incompatibles Lite3
        self.rewards.wheel_vel_penalty = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.feet_contact = None

        if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_7":
            self.disable_zero_weight_rewards()

# V8
# Stabilitsation et impulsion

@configclass
class CustomLite3LongJumpEnvCfg_8(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Bound — identique v7
        self.rewards.feet_gait.weight = 2.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_FOOT", "FR_FOOT"],
            ["HL_FOOT", "HR_FOOT"],
        ]

        # Air time — augmenté pour forcer le décollage
        self.rewards.feet_air_time.weight = 15.0
        self.rewards.feet_air_time.params["threshold"] = 0.4

        # Orientation relâchée : le robot se penche, on tolère
        self.rewards.flat_orientation_l2.weight = -1.0

        # Récompenser simplement la hauteur du centre de masse au dessus du normal
        #    (même sans décollage complet, encourage à se redresser et pousser)
        self.rewards.base_height_l2.weight = 5.0
        self.rewards.base_height_l2.params["target_height"] = 0.55  # au dessus du normal 0.35
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Garder lin_vel_z_positive pour l'impulsion
        self.rewards.lin_vel_z_positive = RewardTermCfg(
            func=mdp.lin_vel_z_positive,
            weight=10.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.lin_vel_z_l2.weight = 0.0


        # Incompatibles Lite3
        self.rewards.wheel_vel_penalty = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.feet_contact = None

        if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_8":
            self.disable_zero_weight_rewards()

# V9
# Saut

# @configclass
# class CustomLite3LongJumpEnvCfg_9(DeeproboticsLite3RoughEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.scene.terrain.terrain_type = "plane"
#         self.scene.terrain.terrain_generator = None

#         self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
#         self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
#         self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

#         # Bound — identique
#         self.rewards.feet_gait.weight = 2.0
#         self.rewards.feet_gait.params["synced_feet_pair_names"] = [
#             ["FL_FOOT", "FR_FOOT"],
#             ["HL_FOOT", "HR_FOOT"],
#         ]

#         # Pousser vers le haut
#         self.rewards.base_height_l2.weight = 10.0
#         self.rewards.base_height_l2.params["target_height"] = 0.55
#         self.rewards.base_height_l2.params["sensor_cfg"] = None
#         self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

#         # Impulsion verticale augmentée
#         self.rewards.lin_vel_z_positive = RewardTermCfg(
#             func=mdp.lin_vel_z_positive,
#             weight=15.0,
#             params={"asset_cfg": SceneEntityCfg("robot")},
#         )
#         self.rewards.lin_vel_z_l2.weight = 0.0

#         # Air time
#         self.rewards.feet_air_time.weight = 15.0
#         self.rewards.feet_air_time.params["threshold"] = 0.3

#         # Pénalités relâchées pour tolérer le décollage
#         self.rewards.flat_orientation_l2.weight = -0.5
#         self.rewards.ang_vel_xy_l2.weight = -0.01

#         # Incompatibles Lite3
#         self.rewards.wheel_vel_penalty = None
#         self.rewards.body_lin_acc_l2 = None
#         self.rewards.feet_contact = None

#         if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_9":
#             self.disable_zero_weight_rewards()


# Il saute mais pas dans la bonne direction (arriere, gqauche ou droite) et fais du sur place (avance peu dans n'importe auel direction, meme si fausse)

@configclass
class CustomLite3LongJumpEnvCfg_9(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Synchronisation relâchée pour permettre un saut dynamique
        self.rewards.feet_gait.weight = 0.0

        # Récompenser explicitement les 4 pattes en l'air
        self.rewards.all_feet_airborne = RewardTermCfg(
            func=mdp.all_feet_airborne,
            weight=30.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]),
                "asset_cfg": SceneEntityCfg("robot"),
                "height_bonus_scale": 2.0,
                "duration_bonus_scale": 1.0,
            },
        )

        # Impulsion verticale
        self.rewards.lin_vel_z_positive = RewardTermCfg(
            func=mdp.lin_vel_z_positive,
            weight=20.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.lin_vel_z_l2.weight = 0.0

        self.rewards.track_lin_vel_xy_exp.weight = 3.0

        # Stabilité et orientation
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.1

        # # Récompenser la distance vers l'avant
        # self.rewards.forward_distance = RewardTermCfg(
        #     func=mdp.forward_distance_reward,
        #     weight=5.0,
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        # )

        # Incompatibles Lite3
        self.rewards.wheel_vel_penalty = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.feet_contact = None

        if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_9":
            self.disable_zero_weight_rewards()

# V10
# Saut

# @configclass
# class CustomLite3LongJumpEnvCfg_10(DeeproboticsLite3RoughEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.scene.terrain.terrain_type = "plane"
#         self.scene.terrain.terrain_generator = None

#         self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
#         self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
#         self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

#         self.rewards.track_lin_vel_xy_exp.weight = 15.0

#         # Signal dominant : avancer sur X
#         self.rewards.forward_distance_reward = RewardTermCfg(
#             func=mdp.forward_distance_reward,
#             weight=10.0,  
#             params={"asset_cfg": SceneEntityCfg("robot")},
#         )

#         # Remplace all_feet_airborne — récompense hauteur UNIQUEMENT en vol réel
#         self.rewards.base_height_in_flight = RewardTermCfg(
#             func=mdp.base_height_in_flight,
#             weight=20.0,
#             params={
#                 "target_height": 0.35,
#                 "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
#                 "asset_cfg": SceneEntityCfg("robot"),
#             },
#         )

#         # Pénaliser les pattes repliées au sol — anti reward hacking
#         self.rewards.base_height_l2.weight = 8.0
#         self.rewards.base_height_l2.params["target_height"] = 0.35
#         self.rewards.base_height_l2.params["sensor_cfg"] = None
#         self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

#         # Impulsion verticale réduite — ne doit pas dominer
#         self.rewards.lin_vel_z_positive = RewardTermCfg(
#             func=mdp.lin_vel_z_positive,
#             weight=5.0,  # était 20.0
#             params={"asset_cfg": SceneEntityCfg("robot")},
#         )
#         self.rewards.lin_vel_z_l2.weight = 0.0

#         # Désactiver track_lin_vel_xy_exp — remplacé par forward_distance_reward
#         self.rewards.track_lin_vel_xy_exp.weight = 0.0

#         # Bound avant/arrière
#         self.rewards.feet_gait.weight = 2.0
#         self.rewards.feet_gait.params["synced_feet_pair_names"] = [
#             ["FL_FOOT", "FR_FOOT"],
#             ["HL_FOOT", "HR_FOOT"],
#         ]

#         # Pénalités
#         self.rewards.flat_orientation_l2.weight = -1.0
#         self.rewards.ang_vel_xy_l2.weight = -0.05

#         # Incompatibles Lite3
#         self.rewards.wheel_vel_penalty = None
#         self.rewards.body_lin_acc_l2 = None
#         self.rewards.feet_contact = None

#         if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_10":
#             self.disable_zero_weight_rewards()

@configclass
class CustomLite3LongJumpEnvCfg_10(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Signal de vitesse : gradient immédiat
        self.rewards.track_lin_vel_xy_exp.weight = 15.0

        # Distance X
        self.rewards.forward_distance_reward = RewardTermCfg(
            func=mdp.forward_distance_reward,
            weight=10.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Hauteur uniquement en vol réel
        self.rewards.base_height_in_flight = RewardTermCfg(
            func=mdp.base_height_in_flight,
            weight=20.0,
            params={
                "target_height": 0.35,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Anti reward hacking : pénalise les pattes repliées sans vrai saut
        self.rewards.penalize_folded_legs = RewardTermCfg(
            func=mdp.penalize_folded_legs,
            weight=-3.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Impulsion verticale
        self.rewards.lin_vel_z_positive = RewardTermCfg(
            func=mdp.lin_vel_z_positive,
            weight=5.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.lin_vel_z_l2.weight = 0.0

        # Bound avant/arrière
        self.rewards.feet_gait.weight = 2.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_FOOT", "FR_FOOT"],
            ["HL_FOOT", "HR_FOOT"],
        ]

        # Désactivés
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0

        # Pénalités
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05

        # Incompatibles Lite3
        self.rewards.wheel_vel_penalty = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.feet_contact = None

        if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_10":
            self.disable_zero_weight_rewards()

# V11
# Saut

@configclass
class CustomLite3LongJumpEnvCfg_11(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Signal de vitesse : gradient immédiat
        self.rewards.track_lin_vel_xy_exp.weight = 15.0

        # Distance X
        self.rewards.forward_distance_reward = RewardTermCfg(
            func=mdp.forward_distance_reward,
            weight=10.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Hauteur uniquement en vol réel
        self.rewards.base_height_in_flight = RewardTermCfg(
            func=mdp.base_height_in_flight,
            weight=30.0,
            params={
                "target_height": 0.35,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Anti reward hacking : pénalise les pattes repliées sans vrai saut
        self.rewards.penalize_folded_legs = RewardTermCfg(
            func=mdp.penalize_folded_legs,
            weight=-3.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Impulsion verticale
        self.rewards.lin_vel_z_positive = RewardTermCfg(
            func=mdp.lin_vel_z_positive,
            weight=15.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.lin_vel_z_l2.weight = 0.0

        # Bound avant/arrière
        self.rewards.feet_gait.weight = 2.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = [
            ["FL_FOOT", "FR_FOOT"],
            ["HL_FOOT", "HR_FOOT"],
        ]

        # Désactivés
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0

        # Pénalités
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05

        # Incompatibles Lite3
        self.rewards.wheel_vel_penalty = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.feet_contact = None

        if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_11":
            self.disable_zero_weight_rewards()

# V12

# @configclass
# class CustomLite3LongJumpEnvCfg_12(DeeproboticsLite3RoughEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.scene.terrain.terrain_type = "plane"
#         self.scene.terrain.terrain_generator = None

#         self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
#         self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
#         self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

#         # Signal dominant : avancer sur X AVANT et PENDANT le saut
#         self.rewards.track_lin_vel_xy_exp.weight = 25.0  # très fort

#         # Pénalité directe sur le mouvement vers l'arrière
#         self.rewards.penalize_backward_and_lateral = RewardTermCfg(
#             func=mdp.penalize_backward_and_lateral,
#             weight=-10.0,
#             params={"asset_cfg": SceneEntityCfg("robot")},
#         )

#         # Hauteur en vol — maintenu mais réduit pour ne plus dominer
#         self.rewards.base_height_in_flight = RewardTermCfg(
#             func=mdp.base_height_in_flight,
#             weight=10.0,  # était 30.0
#             params={
#                 "target_height": 0.35,
#                 "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
#                 "asset_cfg": SceneEntityCfg("robot"),
#             },
#         )

#         # Impulsion verticale réduite
#         self.rewards.lin_vel_z_positive = RewardTermCfg(
#             func=mdp.lin_vel_z_positive,
#             weight=5.0,  # était 15.0
#             params={"asset_cfg": SceneEntityCfg("robot")},
#         )
#         self.rewards.lin_vel_z_l2.weight = 0.0

#         # Anti triche
#         self.rewards.penalize_folded_legs = RewardTermCfg(
#             func=mdp.penalize_folded_legs,
#             weight=-3.0,
#             params={
#                 "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
#                 "asset_cfg": SceneEntityCfg("robot"),
#             },
#         )

#         # Bound maintenu
#         self.rewards.feet_gait.weight = 2.0
#         self.rewards.feet_gait.params["synced_feet_pair_names"] = [
#             ["FL_FOOT", "FR_FOOT"],
#             ["HL_FOOT", "HR_FOOT"],
#         ]

#         # Désactivés
#         self.rewards.base_height_l2.weight = 0.0
#         self.rewards.track_ang_vel_z_exp.weight = 0.0
#         self.rewards.feet_air_time_variance.weight = 0.0
#         self.rewards.stand_still.weight = 0.0
#         self.rewards.feet_contact_without_cmd.weight = 0.0
#         self.rewards.feet_height.weight = 0.0
#         self.rewards.feet_height_body.weight = 0.0

#         # Pénalités
#         self.rewards.flat_orientation_l2.weight = -1.0
#         self.rewards.ang_vel_xy_l2.weight = -0.05

#         # Incompatibles Lite3
#         self.rewards.wheel_vel_penalty = None
#         self.rewards.body_lin_acc_l2 = None
#         self.rewards.feet_contact = None

#         if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_12":
#             self.disable_zero_weight_rewards()


# @configclass
# class CustomLite3LongJumpEnvCfg_12(DeeproboticsLite3RoughEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         self.scene.terrain.terrain_type = "plane"
#         self.scene.terrain.terrain_generator = None

#         # Commandes : pousser vers l’avant uniquement
#         self.commands.base_velocity.ranges.lin_vel_x = (2.0, 3.0)
#         self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
#         self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

#         self.rewards.forward_distance = RewardTermCfg(
#             func=mdp.forward_distance_reward,
#             weight=8.0,
#             params={
#                 "asset_cfg": SceneEntityCfg("robot"),
#             },
#         )
#         self.rewards.track_lin_vel_xy_exp.weight = 0.0

#         self.rewards.all_feet_airborne = RewardTermCfg(
#             func=mdp.all_feet_airborne,
#             weight=3.0,
#             params={
#                 "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
#                 "asset_cfg": SceneEntityCfg("robot"),
#                 "height_bonus_scale": 2.0,
#                 "duration_bonus_scale": 1.0,
#             },
#         )

#         self.rewards.lin_vel_z_positive = RewardTermCfg(
#             func=mdp.lin_vel_z_positive,
#             weight=2.5,
#             params={
#                 "asset_cfg": SceneEntityCfg("robot"),
#             },
#         )

#         self.rewards.penalize_folded_legs = RewardTermCfg(
#             func=mdp.penalize_folded_legs,
#             weight=-5.0,
#             params={
#                 "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
#                 "asset_cfg": SceneEntityCfg("robot"),
#             },
#         )

#         self.rewards.penalize_backward_and_lateral = RewardTermCfg(
#             func=mdp.penalize_backward_and_lateral,
#             weight=-3.0,
#             params={
#                 "asset_cfg": SceneEntityCfg("robot"),
#             },
#         )

#         self.rewards.flat_orientation_l2.weight = -1.0
#         self.rewards.lin_vel_z_l2.weight = -0.5
#         self.rewards.ang_vel_xy_l2.weight = -0.05

#         self.rewards.track_ang_vel_z_exp.weight = 0.0
#         self.rewards.base_height_l2.weight = 0.0
#         self.rewards.feet_air_time_variance.weight = 0.0
#         self.rewards.stand_still.weight = 0.0
#         self.rewards.feet_contact_without_cmd.weight = 0.0
#         self.rewards.feet_height.weight = 0.0
#         self.rewards.feet_height_body.weight = 0.0

#         self.rewards.wheel_vel_penalty = None
#         self.rewards.body_lin_acc_l2 = None
#         self.rewards.feet_contact = None

#         if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_12":
#             self.disable_zero_weight_rewards()


@configclass
class CustomLite3LongJumpEnvCfg_12(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Commandes
        self.commands.base_velocity.ranges.lin_vel_x = (2.5, 3.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.rewards.forward_distance = RewardTermCfg(
            func=mdp.forward_distance_reward,
            weight=20.0,  # :arrow_up: Augmenté (vs 8.0 dans V12)
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Récompenses de saut
        self.rewards.all_feet_airborne = RewardTermCfg(
            func=mdp.all_feet_airborne,
            weight=1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
                "asset_cfg": SceneEntityCfg("robot"),
                "height_bonus_scale": 1.0,  # :arrow_down: Réduit
                "duration_bonus_scale": 0.5,
            },
        )
        self.rewards.lin_vel_z_positive = RewardTermCfg(
            func=mdp.lin_vel_z_positive,
            weight=1.0,  
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        self.rewards.penalize_folded_legs = RewardTermCfg(
            func=mdp.penalize_folded_legs,
            weight=-10.0, 
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT"]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.penalize_backward_and_lateral = RewardTermCfg(
            func=mdp.penalize_backward_and_lateral,
            weight=-15.0, 
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        self.rewards.flat_orientation_l2.weight = -1.0  
        self.rewards.ang_vel_xy_l2.weight = -0.01      # :a

        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0

        # Incompatibles Lite3
        self.rewards.wheel_vel_penalty = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.feet_contact = None

        if self.__class__.__name__ == "CustomLite3LongJumpEnvCfg_12":
            self.disable_zero_weight_rewards()