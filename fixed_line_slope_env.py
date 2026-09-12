"""Ground-contact fixed-line travel on a 28-degree alpine slope.

The robot coordinates alternating crampon steps with a handled one-way
ascender.  The learned arm command loads the Jumar through the right wrist;
an equal-and-opposite reaction is applied to the deformable rope, so the arm
can contribute real uphill impulse without an unbalanced body force.
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class G1FixedLineSlopeEnv(gym.Env):
    """High-level alternating-step policy for inclined fixed-line travel."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        model_path: str | None = None,
        frame_skip: int = 10,
        max_episode_steps: int = 1100,
        target_ascent: float = 1.50,
        render_mode: str | None = None,
        randomize_reset: bool = True,
        action_filter: float = 0.35,
    ) -> None:
        super().__init__()
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "assets",
                "unitree_g1",
                "scene_fixed_line_slope.xml",
            )
        self.model_path = os.path.abspath(model_path)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.target_ascent = float(target_ascent)
        self.render_mode = render_mode
        self.randomize_reset = bool(randomize_reset)
        self.action_filter = float(action_filter)
        self.policy_dt = self.frame_skip * self.model.opt.timestep
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        # PPO chooses each step and how hard to load the handled ascender.
        # Keeping the arm pull as an explicit policy action makes its causal
        # contribution measurable and prevents a hidden scripted hoist.
        self.action_dim = 3

        self.slope_angle = np.deg2rad(28.0)
        self.uphill = np.asarray(
            [np.cos(self.slope_angle), 0.0, np.sin(self.slope_angle)],
            dtype=np.float64,
        )
        self.slope_normal = np.asarray(
            [-np.sin(self.slope_angle), 0.0, np.cos(self.slope_angle)],
            dtype=np.float64,
        )
        # Keep the fixed line outside the robot's right hip instead of running
        # it through the sagittal plane.  The prepared right-hand grip and the
        # short harness tether both meet the line on this same side.
        self.rope_y = -0.32
        # A taut alpine handline sits about knee-to-waist high above the local
        # surface at this pitch; it follows the slope instead of hanging plumb.
        self.rope_normal_offset = 0.68
        # A handled ascender keeps the operator's knuckles outside the rope
        # channel.  These offsets register the wrist to the rubber handle,
        # not to the rope centerline; the compliant cam guide uses the exact
        # inverse transform below so its reaction remains force-balanced.
        self.hand_wrist_uphill_offset = -0.019
        self.hand_wrist_normal_offset = 0.050
        self.hand_wrist_lateral_offset = -0.040
        # Keep the Jumar at a compact waist-to-chest reach.  During a step it
        # locks at this rope coordinate while the body advances toward it.
        self.hand_ascender_reach = 0.22
        half_rope_rotation = 0.5 * (0.5 * np.pi - self.slope_angle)
        self.rope_quaternion = np.asarray(
            [np.cos(half_rope_rotation), 0.0, np.sin(half_rope_rotation), 0.0]
        )

        self.pelvis_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.left_foot_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        self.left_wrist_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"
        )
        self.right_wrist_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
        )
        self.ice_face_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "ice_face")
        self.fixed_rope_flex_id = self._id(
            mujoco.mjtObj.mjOBJ_FLEX, "fixed_rope"
        )
        rope_vertex_address = int(
            self.model.flex_vertadr[self.fixed_rope_flex_id]
        )
        rope_vertex_count = int(
            self.model.flex_vertnum[self.fixed_rope_flex_id]
        )
        self._rope_vertex_slice = slice(
            rope_vertex_address,
            rope_vertex_address + rope_vertex_count,
        )
        self._rope_vertex_body_ids = self.model.flex_vertbodyid[
            self._rope_vertex_slice
        ].astype(np.int32)
        self._rope_dynamic_body_ids = np.unique(
            self._rope_vertex_body_ids[self._rope_vertex_body_ids > 0]
        )
        self._rope_dof_ids = np.flatnonzero(
            np.isin(self.model.dof_bodyid, self._rope_dynamic_body_ids)
        )
        # A real kernmantle rope dissipates transverse oscillation through
        # sheath/core friction.  Flex edge damping is axial only, so damp the
        # generated vertex translation DOFs as well.
        self.model.dof_damping[self._rope_dof_ids] = 0.30
        self.model.dof_armature[self._rope_dof_ids] = 2e-5
        mujoco.mj_forward(self.model, self.data)
        self._rope_rest_vertices = self.data.flexvert_xpos[
            self._rope_vertex_slice
        ].copy()
        self._rope_progress_values = self._rope_rest_vertices @ self.uphill
        rope_edge_address = int(self.model.flex_edgeadr[self.fixed_rope_flex_id])
        rope_edge_count = int(self.model.flex_edgenum[self.fixed_rope_flex_id])
        self._rope_edge_slice = slice(
            rope_edge_address,
            rope_edge_address + rope_edge_count,
        )
        self._rope_rest_length = float(
            np.sum(self.model.flexedge_length0[self._rope_edge_slice])
        )
        protected_body_tokens = (
            "pelvis",
            "waist",
            "torso",
            "hip",
            "knee",
            "ankle",
            "head",
        )
        self._protected_rope_geom_ids = frozenset(
            geom_id
            for geom_id in range(self.model.ngeom)
            if any(
                token in (
                    mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(self.model.geom_bodyid[geom_id]),
                    )
                    or ""
                )
                for token in protected_body_tokens
            )
        )
        self._right_hand_rope_geom_ids = frozenset(
            geom_id
            for geom_id in range(self.model.ngeom)
            if (
                (
                    mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(self.model.geom_bodyid[geom_id]),
                    )
                    or ""
                ).startswith("right_hand_")
                or (
                    mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(self.model.geom_bodyid[geom_id]),
                    )
                    or ""
                )
                == "right_wrist_yaw_link"
            )
            and int(self.model.geom_contype[geom_id]) != 0
        )
        self._slope_friction = self.model.geom_friction[
            self.ice_face_geom_id
        ].copy()
        self._mocap_ids = {
            name: int(
                self.model.body_mocapid[
                    self._id(mujoco.mjtObj.mjOBJ_BODY, name)
                ]
            )
            for name in (
                "harness_visual",
                "chest_ascender_visual",
                "hand_ascender_visual",
                "chest_lanyard_visual",
                "left_crampon_visual",
                "right_crampon_visual",
            )
        }

        self._actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            for i in range(self.nu)
        ]
        self._actuator_ids = {
            name: index for index, name in enumerate(self._actuator_names)
        }
        self._actuator_qpos_addresses = self.model.jnt_qposadr[
            self.model.actuator_trnid[:, 0]
        ].astype(np.int32)
        self._actuator_dof_addresses = self.model.jnt_dofadr[
            self.model.actuator_trnid[:, 0]
        ].astype(np.int32)
        self._nominal_ctrl = np.zeros(self.nu, dtype=np.float64)
        self._configure_prepared_pose()

        arm_joints = (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
        self._arm_actuator_ids = {
            side: np.asarray(
                [self._actuator_ids[f"{side}_{joint}_joint"] for joint in arm_joints],
                dtype=np.int32,
            )
            for side in ("left", "right")
        }
        self._arm_qpos_addresses = {
            side: self._actuator_qpos_addresses[actuators]
            for side, actuators in self._arm_actuator_ids.items()
        }
        self._arm_dof_addresses = {
            side: self.model.jnt_dofadr[
                self.model.actuator_trnid[actuators, 0]
            ].astype(np.int32)
            for side, actuators in self._arm_actuator_ids.items()
        }
        self._distal_digit_body_ids = {
            side: np.asarray(
                [
                    self._id(
                        mujoco.mjtObj.mjOBJ_BODY,
                        f"{side}_hand_{digit}_link",
                    )
                    for digit in ("thumb_2", "index_1", "middle_1")
                ],
                dtype=np.int32,
            )
            for side in ("left", "right")
        }
        self._foot_body_ids = {
            side: self._id(
                mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link"
            )
            for side in ("left", "right")
        }

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        # Rotation, base velocity, actuator error/velocity, 10 gait/contact
        # scalars, and the previous two step plus arm-pull commands.
        self.obs_dim = 9 + 3 + 3 + 2 * self.nu + 10 + self.action_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Rope mass belongs to the anchored environment, not the robot when
        # normalizing measured boot load by bodyweight.
        rope_mass = float(
            np.sum(self.model.body_mass[self._rope_dynamic_body_ids])
        )
        self.total_mass = float(mujoco.mj_getTotalmass(self.model) - rope_mass)
        self.body_weight = self.total_mass * 9.81
        self.line_stiffness = 18_000.0
        self.line_damping = 520.0
        self.max_line_force = 700.0
        self.rope_guide_stiffness = 450.0
        self.rope_guide_damping = 18.0
        self.max_rope_guide_force = 80.0
        # The 18%-bodyweight cap stays inside the stock 25 Nm shoulder/elbow
        # limits while remaining large enough for PPO to discover and exploit.
        self.max_arm_pull_force = 0.18 * self.body_weight
        self.lateral_stiffness = 700.0
        self.lateral_damping = 100.0
        self.orientation_stiffness = 420.0
        self.orientation_damping = 46.0
        self.normal_balance_stiffness = 900.0
        self.normal_balance_damping = 90.0
        self.max_normal_balance_force = 100.0
        self.step_duration = 18
        self.step_request_hold = 4
        self._desired_base_quaternion = np.asarray(
            [np.cos(0.045), 0.0, np.sin(0.045), 0.0]
        )

        self._renderer: mujoco.Renderer | None = None
        self._line_enabled = True
        self._traction_enabled = True
        self._arm_pull_enabled = True
        self._step_count = 0
        self._completed_steps = 0
        self._expected_side = 0
        self._swing_side: int | None = None
        self._swing_phase = 0.0
        self._step_power = 0.0
        self._step_cooldown = 0
        self._request_hold_steps = 0
        self._start_progress = 0.0
        self._previous_progress = 0.0
        self._high_water_progress = 0.0
        self._line_ratchet_progress = 0.0
        self._hand_ascender_progress = 0.0
        self._stable_success_steps = 0
        self._line_loaded_steps = 0
        self._arm_pull_loaded_steps = 0
        self._rope_core_collision_steps = 0
        self._hand_rope_penetration_steps = 0
        self._grounded_steps = 0
        self._left_contact_steps = 0
        self._right_contact_steps = 0
        self._double_support_steps = 0
        self._airborne_streak = 0
        self._maximum_airborne_streak = 0
        self._last_action = np.zeros(self.action_dim, dtype=np.float64)
        self._arm_pull_command = 0.0
        self._last_line_force = 0.0
        self._last_rope_guide_force = 0.0
        self._last_arm_pull_force = 0.0
        self._arm_pull_impulse_ns = 0.0
        self._last_ground_load = 0.0
        self._wrist_target_quaternions = {
            "left": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "right": np.asarray([1.0, 0.0, 0.0, 0.0]),
        }

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        result = mujoco.mj_name2id(self.model, object_type, name)
        if result < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return result

    def _set_named_targets(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self._nominal_ctrl[self._actuator_ids[name]] = value

    def _configure_prepared_pose(self) -> None:
        for side in ("left", "right"):
            self._set_named_targets(
                {
                    f"{side}_hip_pitch_joint": -0.18,
                    f"{side}_knee_joint": 0.38,
                    f"{side}_ankle_pitch_joint": -0.52,
                    f"{side}_hip_roll_joint": 0.0,
                    f"{side}_hip_yaw_joint": 0.0,
                    f"{side}_ankle_roll_joint": 0.0,
                }
            )
        arms = {
            # The left arm stays bent and clear for balance; forcing it across
            # the torso to a second rope device looked unlike real handline
            # travel at this pitch.
            "left": (-0.42, 0.34, 0.05, 0.88, 0.0, 0.02, 0.0),
            "right": (-0.95, -0.08, -0.10, 0.62, 0.0, 0.08, 0.0),
        }
        joints = (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
        for side, values in arms.items():
            self._set_named_targets(
                {
                    f"{side}_{joint}_joint": value
                    for joint, value in zip(joints, values)
                }
            )
        self._set_named_targets(
            {
                "left_hand_thumb_0_joint": 0.05,
                "left_hand_thumb_1_joint": 0.22,
                "left_hand_thumb_2_joint": 0.20,
                "left_hand_middle_0_joint": -0.25,
                "left_hand_middle_1_joint": -0.35,
                "left_hand_index_0_joint": -0.25,
                "left_hand_index_1_joint": -0.35,
                "right_hand_thumb_0_joint": -0.10,
                "right_hand_thumb_1_joint": -0.70,
                "right_hand_thumb_2_joint": -0.85,
                "right_hand_middle_0_joint": 0.85,
                "right_hand_middle_1_joint": 1.10,
                "right_hand_index_0_joint": 0.85,
                "right_hand_index_1_joint": 1.10,
            }
        )

    def _rope_sample(
        self, progress: float
    ) -> tuple[np.ndarray, int, int, float]:
        """Interpolate a world point and reaction weights on the flex rope."""
        progress_clamped = float(
            np.clip(
                progress,
                self._rope_progress_values[0],
                self._rope_progress_values[-1],
            )
        )
        upper = int(
            np.clip(
                np.searchsorted(self._rope_progress_values, progress_clamped),
                1,
                len(self._rope_progress_values) - 1,
            )
        )
        lower = upper - 1
        interval = float(
            self._rope_progress_values[upper]
            - self._rope_progress_values[lower]
        )
        weight = (
            (progress_clamped - self._rope_progress_values[lower])
            / max(interval, 1e-9)
        )
        vertices = self.data.flexvert_xpos[self._rope_vertex_slice]
        point = (1.0 - weight) * vertices[lower] + weight * vertices[upper]
        return point, lower, upper, float(weight)

    def _rope_point(self, progress: float) -> np.ndarray:
        return self._rope_sample(progress)[0]

    def _progress(self) -> float:
        return float(np.dot(self.data.xpos[self.pelvis_body_id], self.uphill))

    def protected_rope_collision(self) -> bool:
        """Whether the physical flex rope contacts the robot's core or legs."""
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.fixed_rope_flex_id not in contact.flex:
                continue
            if any(
                int(geom_id) in self._protected_rope_geom_ids
                for geom_id in contact.geom
            ):
                return True
        return False

    def _hand_rope_contact_state(self) -> tuple[int, float]:
        """Return physical hand/sheath contacts and deepest penetration."""
        count = 0
        maximum_penetration = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.fixed_rope_flex_id not in contact.flex:
                continue
            if not any(
                int(geom_id) in self._right_hand_rope_geom_ids
                for geom_id in contact.geom
            ):
                continue
            count += 1
            maximum_penetration = max(
                maximum_penetration,
                max(-float(contact.dist), 0.0),
            )
        return count, maximum_penetration

    def _normal_height(self, position: np.ndarray) -> float:
        return float(np.dot(position, self.slope_normal))

    def _foot_contacts(self) -> tuple[bool, bool, float, float]:
        contacts = {"left": False, "right": False}
        loads = {"left": 0.0, "right": 0.0}
        contact_force = np.zeros(6, dtype=np.float64)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.ice_face_geom_id not in (contact.geom1, contact.geom2):
                continue
            other_geom = (
                contact.geom2
                if contact.geom1 == self.ice_face_geom_id
                else contact.geom1
            )
            other_body = int(self.model.geom_bodyid[other_geom])
            side = next(
                (
                    candidate
                    for candidate, body_id in self._foot_body_ids.items()
                    if other_body == body_id
                ),
                None,
            )
            if side is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, contact_force)
            contacts[side] = True
            loads[side] += max(float(contact_force[0]), 0.0)
        return contacts["left"], contacts["right"], loads["left"], loads["right"]

    def _rope_velocity(self, lower: int, upper: int, weight: float) -> np.ndarray:
        """Interpolate the world linear velocity of two flex vertices."""
        velocities = []
        spatial_velocity = np.zeros(6, dtype=np.float64)
        for vertex in (lower, upper):
            body_id = int(self._rope_vertex_body_ids[vertex])
            if body_id == 0:
                velocities.append(np.zeros(3, dtype=np.float64))
                continue
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                spatial_velocity,
                0,
            )
            velocities.append(spatial_velocity[3:].copy())
        return (1.0 - weight) * velocities[0] + weight * velocities[1]

    def _apply_rope_force(self, progress: float, force: np.ndarray) -> None:
        """Distribute a world-space force across adjacent flex vertices."""
        point, lower, upper, weight = self._rope_sample(progress)
        for vertex, share in ((lower, 1.0 - weight), (upper, weight)):
            body_id = int(self._rope_vertex_body_ids[vertex])
            if body_id == 0 or share <= 0.0:
                continue
            mujoco.mj_applyFT(
                self.model,
                self.data,
                share * force,
                np.zeros(3, dtype=np.float64),
                point,
                body_id,
                self.data.qfrc_applied,
            )

    def _apply_hand_ascender_guide(self, progress: float) -> float:
        """Guide the rope through the handled cam with balanced forces.

        The cam is free to slide along the rope but constrains motion across
        it.  A compliant transverse spring keeps the physical flex inside the
        device while applying the equal-and-opposite reaction at the wrist.
        """
        point, lower, upper, weight = self._rope_sample(progress)
        wrist = self.data.xpos[self.right_wrist_body_id]
        desired_rope_point = (
            wrist
            - self.hand_wrist_uphill_offset * self.uphill
            - self.hand_wrist_normal_offset * self.slope_normal
            - np.asarray([0.0, self.hand_wrist_lateral_offset, 0.0])
        )
        error = desired_rope_point - point
        error -= float(np.dot(error, self.uphill)) * self.uphill

        spatial_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.right_wrist_body_id,
            spatial_velocity,
            0,
        )
        wrist_velocity = spatial_velocity[3:].copy()
        relative_velocity = (
            self._rope_velocity(lower, upper, weight) - wrist_velocity
        )
        relative_velocity -= (
            float(np.dot(relative_velocity, self.uphill)) * self.uphill
        )
        rope_force = (
            self.rope_guide_stiffness * error
            - self.rope_guide_damping * relative_velocity
        )
        magnitude = float(np.linalg.norm(rope_force))
        if magnitude > self.max_rope_guide_force:
            rope_force *= self.max_rope_guide_force / magnitude
            magnitude = self.max_rope_guide_force

        self._apply_rope_force(progress, rope_force)
        mujoco.mj_applyFT(
            self.model,
            self.data,
            -rope_force,
            np.zeros(3, dtype=np.float64),
            wrist,
            self.right_wrist_body_id,
            self.data.qfrc_applied,
        )
        return magnitude

    def _apply_support_forces(self) -> tuple[float, float, float]:
        self.data.qfrc_applied.fill(0.0)
        pelvis = self.data.xpos[self.pelvis_body_id]
        velocity = self.data.qvel[:3]
        progress = float(np.dot(pelvis, self.uphill))
        uphill_velocity = float(np.dot(velocity, self.uphill))
        line_force = 0.0
        if self._line_enabled:
            # Advance the one-way chest cam only during a deliberate climbing
            # stroke. Otherwise tiny pose-controller oscillations would be
            # rectified into slow, uncommanded uphill travel.
            if self._swing_side is not None:
                self._line_ratchet_progress = max(
                    self._line_ratchet_progress, progress
                )
            slip = max(self._line_ratchet_progress - progress, 0.0)
            _point, lower, upper, weight = self._rope_sample(
                self._line_ratchet_progress + 0.05
            )
            rope_uphill_velocity = float(
                np.dot(self._rope_velocity(lower, upper, weight), self.uphill)
            )
            relative_uphill_velocity = uphill_velocity - rope_uphill_velocity
            # A slack tether cannot transmit a damping force.  Applying the
            # dashpot term before extension made the rope behave like an
            # invisible downhill velocity brake instead of a unilateral
            # tension element.
            line_force = float(
                np.clip(
                    (
                        self.line_stiffness * slip
                        - self.line_damping * min(relative_uphill_velocity, 0.0)
                    )
                    if slip > 0.0
                    else 0.0,
                    0.0,
                    self.max_line_force,
                )
            )
        lateral_force = (
            -self.lateral_stiffness * float(pelvis[1])
            - self.lateral_damping * float(velocity[1])
        )
        support_force = line_force * self.uphill
        support_force[1] += lateral_force
        left_contact, right_contact, _left_load, _right_load = self._foot_contacts()
        normal_height = self._normal_height(pelvis)
        normal_velocity = float(np.dot(velocity, self.slope_normal))
        normal_balance = 0.0
        if left_contact or right_contact:
            normal_balance = float(
                np.clip(
                    self.normal_balance_stiffness * (0.67 - normal_height)
                    - self.normal_balance_damping * normal_velocity,
                    0.0,
                    self.max_normal_balance_force,
                )
            )
            support_force += normal_balance * self.slope_normal
        orientation_error = np.zeros(3, dtype=np.float64)
        mujoco.mju_subQuat(
            orientation_error,
            self._desired_base_quaternion,
            self.data.qpos[3:7],
        )
        torque = (
            self.orientation_stiffness * orientation_error
            - self.orientation_damping * self.data.qvel[3:6]
        )
        mujoco.mj_applyFT(
            self.model,
            self.data,
            support_force,
            torque,
            pelvis,
            self.pelvis_body_id,
            self.data.qfrc_applied,
        )
        if line_force > 0.0:
            self._apply_rope_force(
                self._line_ratchet_progress + 0.05,
                -line_force * self.uphill,
            )
        guide_force = (
            self._apply_hand_ascender_guide(self._hand_ascender_progress)
            if self._line_enabled and self._swing_side is not None
            else 0.0
        )
        arm_pull_force = 0.0
        if (
            self._line_enabled
            and self._arm_pull_enabled
            and self._swing_side is not None
        ):
            # The positive-only third policy action contracts the loaded arm.
            # A smooth bell-shaped stroke avoids an impulsive cam engagement.
            activation = float(np.sin(np.pi * self._swing_phase) ** 2)
            arm_pull_force = (
                self.max_arm_pull_force
                * max(self._arm_pull_command, 0.0)
                * activation
            )
            if arm_pull_force > 0.0:
                wrist = self.data.xpos[self.right_wrist_body_id]
                wrist_force = arm_pull_force * self.uphill
                mujoco.mj_applyFT(
                    self.model,
                    self.data,
                    wrist_force,
                    np.zeros(3, dtype=np.float64),
                    wrist,
                    self.right_wrist_body_id,
                    self.data.qfrc_applied,
                )
                # Newton's third law: the anchored, deformable rope receives
                # the full opposite reaction at the locked Jumar coordinate.
                self._apply_rope_force(
                    self._hand_ascender_progress,
                    -wrist_force,
                )
                self._arm_pull_impulse_ns += (
                    arm_pull_force * self.model.opt.timestep
                )

        return line_force, guide_force, arm_pull_force

    def _start_step_if_requested(self, action: np.ndarray) -> None:
        if not self._traction_enabled:
            self._request_hold_steps = 0
            return
        if self._swing_side is not None or self._step_cooldown > 0:
            self._request_hold_steps = 0
            return
        expected = self._expected_side
        other = 1 - expected
        deliberate_request = (
            action[expected] > 0.22
            and action[expected] - action[other] > 0.20
        )
        self._request_hold_steps = (
            self._request_hold_steps + 1 if deliberate_request else 0
        )
        if self._request_hold_steps >= self.step_request_hold:
            self._swing_side = expected
            self._swing_phase = 0.0
            self._step_power = float(np.clip(0.55 + 0.45 * action[expected], 0.45, 1.0))
            # Slide the unloaded Jumar up, then lock it for this complete
            # stroke.  As the pelvis advances the arm visibly shortens toward
            # a fixed point on the rope instead of dragging a floating prop.
            self._hand_ascender_progress = max(
                self._hand_ascender_progress,
                self._progress() + self.hand_ascender_reach,
            )
            self._request_hold_steps = 0

    def _leg_targets(self, target: np.ndarray) -> None:
        for side_index, side in enumerate(("left", "right")):
            hip, knee, ankle = -0.18, 0.38, -0.52
            if self._swing_side is not None:
                if side_index == self._swing_side:
                    lift = float(np.sin(np.pi * self._swing_phase))
                    hip = -0.18 - 0.52 * lift
                    knee = 0.38 + 1.00 * lift
                    ankle = -0.52 - 0.24 * lift
                else:
                    hip, knee, ankle = -0.08, 0.22, -0.50
            for joint, value in zip(
                ("hip_pitch", "knee", "ankle_pitch"),
                (hip, knee, ankle),
            ):
                target[self._actuator_ids[f"{side}_{joint}_joint"]] = value

    def _solve_arm_ik(
        self,
        side: str,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
    ) -> np.ndarray:
        ik_data = self._ik_data
        ik_data.qpos[:] = self.data.qpos
        actuator_ids = self._arm_actuator_ids[side]
        qpos_addresses = self._arm_qpos_addresses[side]
        dof_addresses = self._arm_dof_addresses[side]
        joint_ids = self.model.actuator_trnid[actuator_ids, 0]
        lower = self.model.jnt_range[joint_ids, 0] + 1e-4
        upper = self.model.jnt_range[joint_ids, 1] - 1e-4
        wrist_body_id = (
            self.left_wrist_body_id if side == "left" else self.right_wrist_body_id
        )
        jacobian_position = np.zeros((3, self.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.nv), dtype=np.float64)
        orientation_error = np.zeros(3, dtype=np.float64)
        for _ in range(8):
            # IK only needs transforms and body Jacobians.  Running the full
            # forward dynamics pass here also factorizes the rope's very stiff
            # equality constraints at every iteration and can make an
            # otherwise kinematic wrist solve numerically rank-deficient.
            mujoco.mj_kinematics(self.model, ik_data)
            mujoco.mj_comPos(self.model, ik_data)
            mujoco.mju_subQuat(
                orientation_error,
                target_quaternion,
                ik_data.xquat[wrist_body_id],
            )
            error = np.concatenate(
                [
                    target_position - ik_data.xpos[wrist_body_id],
                    # Position takes priority so the digits stay on the
                    # inclined hardware; a light orientation term prevents
                    # implausible wrist roll without blocking cross-body reach.
                    0.05 * orientation_error,
                ]
            )
            if float(np.linalg.norm(error)) < 8e-4:
                break
            mujoco.mj_jacBody(
                self.model,
                ik_data,
                jacobian_position,
                jacobian_rotation,
                wrist_body_id,
            )
            jacobian = np.vstack(
                [
                    jacobian_position[:, dof_addresses],
                    jacobian_rotation[:, dof_addresses],
                ]
            )
            damping = 2e-3
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), error
            )
            ik_data.qpos[qpos_addresses] = np.clip(
                ik_data.qpos[qpos_addresses] + np.clip(delta, -0.18, 0.18),
                lower,
                upper,
            )
        return ik_data.qpos[qpos_addresses].copy()

    def _device_points(self) -> tuple[np.ndarray, np.ndarray]:
        progress = self._progress()
        return (
            self._rope_point(progress + 0.05),
            self._rope_point(self._hand_ascender_progress),
        )

    def _arm_target_positions(self) -> dict[str, np.ndarray]:
        _chest, hand = self._device_points()
        return {
            "right": hand
            + self.hand_wrist_uphill_offset * self.uphill
            + self.hand_wrist_normal_offset * self.slope_normal
            + np.asarray([0.0, self.hand_wrist_lateral_offset, 0.0]),
        }

    def _sync_equipment_visuals(self) -> None:
        pelvis = self.data.xpos[self.pelvis_body_id]
        chest, hand = self._device_points()
        positions = {
            "harness_visual": pelvis + np.asarray([0.0, 0.0, -0.08]),
            "chest_ascender_visual": chest,
            "hand_ascender_visual": hand,
            "chest_lanyard_visual": pelvis + np.asarray([0.0, 0.0, -0.04]),
        }
        for name, position in positions.items():
            mocap_id = self._mocap_ids[name]
            self.data.mocap_pos[mocap_id] = position
            self.data.mocap_quat[mocap_id] = (
                self.rope_quaternion
                if name in ("chest_ascender_visual", "hand_ascender_visual")
                else np.asarray([1.0, 0.0, 0.0, 0.0])
            )
        for side in ("left", "right"):
            mocap_id = self._mocap_ids[f"{side}_crampon_visual"]
            body_id = self._foot_body_ids[side]
            self.data.mocap_pos[mocap_id] = self.data.xpos[body_id]
            self.data.mocap_quat[mocap_id] = self.data.xquat[body_id]

    def _collision_flags(self) -> tuple[bool, bool]:
        cross_hand = False
        slope_hand = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            # Flex contacts encode one side as geom=-1. Core/leg and right-hand
            # rope contacts have dedicated checks, so this loop only handles
            # ordinary geom-to-geom hand collisions.
            if contact.geom1 < 0 or contact.geom2 < 0:
                continue
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            name1 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body1
            ) or ""
            name2 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body2
            ) or ""
            cross_hand = cross_hand or (
                name1.startswith("left_hand_") and name2.startswith("right_hand_")
            ) or (
                name1.startswith("right_hand_") and name2.startswith("left_hand_")
            )
            if self.ice_face_geom_id in (contact.geom1, contact.geom2):
                other = name2 if contact.geom1 == self.ice_face_geom_id else name1
                slope_hand = slope_hand or other.startswith(("left_hand_", "right_hand_"))
        return cross_hand, slope_hand

    def _metrics(self) -> dict[str, float]:
        pelvis = self.data.xpos[self.pelvis_body_id]
        torso_rotation = self.data.xmat[self.torso_body_id].reshape(3, 3)
        progress = self._progress() - self._start_progress
        left_contact, right_contact, left_load, right_load = self._foot_contacts()
        left_wrist = self.data.xpos[self.left_wrist_body_id]
        right_wrist = self.data.xpos[self.right_wrist_body_id]
        left_digit_positions = self.data.xpos[self._distal_digit_body_ids["left"]]
        right_digit_positions = self.data.xpos[self._distal_digit_body_ids["right"]]
        left_digits = np.mean(left_digit_positions, axis=0)
        right_digits = np.mean(right_digit_positions, axis=0)
        chest, hand = self._device_points()
        left_grip_error = float(np.linalg.norm(left_digits - chest))
        right_digit_device_error = float(np.linalg.norm(right_digits - hand))
        # The wrist is registered to the offset rubber handle; distal-digit
        # distance to the rope center is not a valid handle-grip error because
        # the closed fingers deliberately remain outside the cam channel.
        right_handle_target = self._arm_target_positions()["right"]
        right_grip_error = float(
            np.linalg.norm(right_wrist - right_handle_target)
        )
        # A true wrap means articulated distal digits span both sides of the
        # rope/handle centerline. Using their mean would incorrectly reject a
        # visibly wrapped hand whenever the palm and mean lie on one side.
        left_wrap = float(
            np.min(left_digit_positions[:, 1])
            < self.rope_y
            < np.max(left_digit_positions[:, 1])
        )
        right_wrap = float(
            np.min(right_digit_positions[:, 1])
            < self.rope_y
            < np.max(right_digit_positions[:, 1])
        )
        # At this slope angle the handled ascender is operated one-handed. The
        # left hand is intentionally free for balance, so only the right grip
        # is scored as a required equipment contact.
        grasp_score = float(
            0.25 * right_wrap
            + 0.75 * np.exp(-np.square(right_grip_error / 0.15))
        )
        cross_hand, slope_hand = self._collision_flags()
        rope_vertices = self.data.flexvert_xpos[self._rope_vertex_slice]
        rope_length = float(
            np.sum(self.data.flexedge_length[self._rope_edge_slice])
        )
        rope_contact_count = sum(
            int(self.fixed_rope_flex_id in self.data.contact[index].flex)
            for index in range(self.data.ncon)
        )
        hand_rope_contact_count, hand_rope_max_penetration = (
            self._hand_rope_contact_state()
        )
        left_foot = self.data.site_xpos[self.left_foot_site_id]
        right_foot = self.data.site_xpos[self.right_foot_site_id]
        ground_load = left_load + right_load
        return {
            "ascent": progress,
            "height_m": progress,
            "vertical_gain_m": float(pelvis[2] - self._start_pelvis_z),
            "uphill_speed": float(np.dot(self.data.qvel[:3], self.uphill)),
            "high_water_ascent": float(self._high_water_progress - self._start_progress),
            "descent_from_high_water": float(self._high_water_progress - self._progress()),
            "upright_score": float(torso_rotation[2, 2]),
            "lateral_offset": abs(float(pelvis[1])),
            "pelvis_normal_height": self._normal_height(pelvis),
            "left_boot_contact": float(left_contact),
            "right_boot_contact": float(right_contact),
            "any_boot_contact": float(left_contact or right_contact),
            "double_support": float(left_contact and right_contact),
            "left_boot_load_n": left_load,
            "right_boot_load_n": right_load,
            "ground_load_n": ground_load,
            "ground_load_bodyweight": ground_load / max(self.body_weight, 1e-6),
            "left_boot_clearance_m": self._normal_height(left_foot),
            "right_boot_clearance_m": self._normal_height(right_foot),
            "line_load_n": self._last_line_force,
            "line_slip_m": max(self._line_ratchet_progress - self._progress(), 0.0),
            "rope_length_m": rope_length,
            "rope_extension_m": rope_length - self._rope_rest_length,
            "rope_max_displacement_m": float(
                np.max(np.linalg.norm(rope_vertices - self._rope_rest_vertices, axis=1))
            ),
            "rope_contact_count": float(rope_contact_count),
            "rope_core_collision": float(self.protected_rope_collision()),
            "hand_rope_contact_count": float(hand_rope_contact_count),
            "hand_rope_max_penetration_m": hand_rope_max_penetration,
            "rope_guide_load_n": self._last_rope_guide_force,
            "arm_pull_load_n": self._last_arm_pull_force,
            "arm_pull_load_bodyweight": (
                self._last_arm_pull_force / max(self.body_weight, 1e-6)
            ),
            "arm_pull_impulse_ns": self._arm_pull_impulse_ns,
            "jumar_relative_progress_m": (
                self._hand_ascender_progress - self._progress()
            ),
            "completed_cycles": float(self._completed_steps),
            "expected_side": float(self._expected_side),
            "swing_side": float(-1 if self._swing_side is None else self._swing_side),
            "swing_phase": self._swing_phase,
            "left_grip_error": left_grip_error,
            "right_grip_error": right_grip_error,
            "right_digit_device_error": right_digit_device_error,
            "left_wrap_score": left_wrap,
            "right_wrap_score": right_wrap,
            "grasp_score": grasp_score,
            "hand_separation": float(np.linalg.norm(left_wrist - right_wrist)),
            "cross_hand_collision": float(cross_hand),
            "wall_hand_collision": float(slope_hand),
        }

    def _get_obs(self) -> np.ndarray:
        metrics = self._metrics()
        rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        joint_errors = self.data.qpos[self._actuator_qpos_addresses] - self.data.ctrl
        gait = np.asarray(
            [
                metrics["ascent"] / max(self.target_ascent, 1e-6),
                metrics["uphill_speed"] * 0.4,
                metrics["descent_from_high_water"] * 5.0,
                -1.0 if self._expected_side == 0 else 1.0,
                metrics["swing_side"],
                metrics["swing_phase"],
                metrics["left_boot_contact"],
                metrics["right_boot_contact"],
                metrics["ground_load_bodyweight"],
                metrics["completed_cycles"] / 10.0,
            ]
        )
        observation = np.concatenate(
            [
                rotation.ravel(),
                self.data.qvel[3:6] * 0.10,
                self.data.qvel[:3] * 0.20,
                joint_errors,
                self.data.qvel[self._actuator_dof_addresses] * 0.05,
                gait,
                self._last_action,
            ]
        )
        return observation.astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        randomized = bool(options.get("randomize", self.randomize_reset))
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        lateral = float(self.np_random.uniform(-0.012, 0.012)) if randomized else 0.0
        self.data.qpos[:3] = [0.0, lateral, 0.84]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self._actuator_qpos_addresses] = self._nominal_ctrl
        self.data.ctrl[:] = self._nominal_ctrl
        if randomized:
            self.data.qpos[self._actuator_qpos_addresses] += self.np_random.normal(
                0.0, 0.004, self.nu
            )
            # Domain-randomize the robot, not the anchored environment.  A
            # blanket qvel perturbation would inject arbitrary velocity into
            # every rope vertex and turn small reset noise into a lateral whip.
            self.data.qvel.fill(0.0)
            self.data.qvel[:6] = self.np_random.normal(0.0, 0.004, 6)
            self.data.qvel[self._actuator_dof_addresses] = self.np_random.normal(
                0.0, 0.004, self.nu
            )
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._completed_steps = 0
        self._expected_side = 0
        self._swing_side = None
        self._swing_phase = 0.0
        self._step_power = 0.0
        self._step_cooldown = 0
        self._request_hold_steps = 0
        initial_progress = self._progress()
        self._start_progress = initial_progress
        self._previous_progress = initial_progress
        self._high_water_progress = initial_progress
        self._line_ratchet_progress = initial_progress
        self._hand_ascender_progress = (
            initial_progress + self.hand_ascender_reach
        )
        self._stable_success_steps = 0
        self._line_loaded_steps = 0
        self._arm_pull_loaded_steps = 0
        self._rope_core_collision_steps = 0
        self._hand_rope_penetration_steps = 0
        self._grounded_steps = 0
        self._left_contact_steps = 0
        self._right_contact_steps = 0
        self._double_support_steps = 0
        self._airborne_streak = 0
        self._maximum_airborne_streak = 0
        self._last_action.fill(0.0)
        self._arm_pull_command = 0.0
        self._last_line_force = 0.0
        self._last_rope_guide_force = 0.0
        self._last_arm_pull_force = 0.0
        self._arm_pull_impulse_ns = 0.0
        self._last_ground_load = 0.0
        self._wrist_target_quaternions = {
            "left": self.data.xquat[self.left_wrist_body_id].copy(),
            "right": self.data.xquat[self.right_wrist_body_id].copy(),
        }
        for side, position in self._arm_target_positions().items():
            arm_targets = self._solve_arm_ik(
                side, position, self._wrist_target_quaternions[side]
            )
            self.data.qpos[self._arm_qpos_addresses[side]] = arm_targets
            self.data.ctrl[self._arm_actuator_ids[side]] = arm_targets
        self._sync_equipment_visuals()
        mujoco.mj_forward(self.model, self.data)

        # Settle into real boot contacts before measuring episode progress.
        for _ in range(160):
            line_force, guide_force, arm_pull_force = self._apply_support_forces()
            self._last_line_force = line_force
            self._last_rope_guide_force = guide_force
            self._last_arm_pull_force = arm_pull_force
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        settled_progress = self._progress()
        self._start_progress = settled_progress
        self._previous_progress = settled_progress
        self._high_water_progress = settled_progress
        self._line_ratchet_progress = settled_progress
        self._hand_ascender_progress = (
            settled_progress + self.hand_ascender_reach
        )
        self._start_pelvis_z = float(self.data.xpos[self.pelvis_body_id, 2])
        self._last_line_force = 0.0
        self._last_rope_guide_force = 0.0
        self._last_arm_pull_force = 0.0
        self._sync_equipment_visuals()
        mujoco.mj_forward(self.model, self.data)
        metrics = self._metrics()
        self._rope_core_collision_steps += int(metrics["rope_core_collision"] > 0.5)
        self._hand_rope_penetration_steps += int(
            metrics["hand_rope_max_penetration_m"] > 8e-4
        )
        metrics.update(
            {
                "success": False,
                "line_enabled": self._line_enabled,
                "traction_enabled": self._traction_enabled,
            }
        )
        return self._get_obs(), metrics

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        command = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        filtered = (
            command
            if self._step_count == 1
            else self.action_filter * self._last_action
            + (1.0 - self.action_filter) * command
        )
        self._arm_pull_command = float(max(filtered[2], 0.0))
        if self._step_cooldown > 0:
            self._step_cooldown -= 1
        if self._swing_side is None:
            # An unloaded cam slides with the hand. It becomes world-locked
            # only when a requested step begins, preventing neutral arm IK
            # from acting like an unintended winch.
            self._hand_ascender_progress = (
                self._progress() + self.hand_ascender_reach
            )
        self._start_step_if_requested(filtered)
        target = self._nominal_ctrl.copy()
        self._leg_targets(target)
        for side, position in self._arm_target_positions().items():
            if self._swing_side is None:
                # Hold the last joint-space grasp while the cam is unloaded.
                # Re-solving a world-frame target every idle frame can inject
                # tiny cyclic joint work that a ratcheting line accumulates.
                target[self._arm_actuator_ids[side]] = self.data.ctrl[
                    self._arm_actuator_ids[side]
                ]
            else:
                target[self._arm_actuator_ids[side]] = self._solve_arm_ik(
                    side, position, self._wrist_target_quaternions[side]
                )
        self.data.ctrl[:] = np.clip(
            target,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

        line_force_peak = 0.0
        guide_force_peak = 0.0
        arm_pull_force_peak = 0.0
        ground_load_peak = 0.0
        for _ in range(self.frame_skip):
            line_force, guide_force, arm_pull_force = self._apply_support_forces()
            line_force_peak = max(line_force_peak, line_force)
            guide_force_peak = max(guide_force_peak, guide_force)
            arm_pull_force_peak = max(arm_pull_force_peak, arm_pull_force)
            _lc, _rc, left_load, right_load = self._foot_contacts()
            ground_load_peak = max(ground_load_peak, left_load + right_load)
            mujoco.mj_step(self.model, self.data)
        self._last_line_force = line_force_peak
        self._last_rope_guide_force = guide_force_peak
        self._last_arm_pull_force = arm_pull_force_peak
        self._last_ground_load = ground_load_peak

        completed_step = False
        if self._swing_side is not None:
            self._swing_phase += 1.0 / self.step_duration
            if self._swing_phase >= 1.0:
                finished_side = self._swing_side
                self._swing_side = None
                self._swing_phase = 0.0
                self._step_cooldown = 3
                self._expected_side = 1 - finished_side
                self._completed_steps += 1
                completed_step = True

        self._sync_equipment_visuals()
        mujoco.mj_forward(self.model, self.data)
        progress_absolute = self._progress()
        self._high_water_progress = max(self._high_water_progress, progress_absolute)
        previous_ascent = self._previous_progress - self._start_progress
        capped_progress_delta = float(
            np.clip(
                progress_absolute - self._start_progress,
                0.0,
                self.target_ascent,
            )
            - np.clip(previous_ascent, 0.0, self.target_ascent)
        )
        metrics = self._metrics()
        self._rope_core_collision_steps += int(metrics["rope_core_collision"] > 0.5)
        self._hand_rope_penetration_steps += int(
            metrics["hand_rope_max_penetration_m"] > 8e-4
        )
        left_contact = bool(metrics["left_boot_contact"])
        right_contact = bool(metrics["right_boot_contact"])
        if left_contact or right_contact:
            self._grounded_steps += 1
            self._airborne_streak = 0
        else:
            self._airborne_streak += 1
        self._maximum_airborne_streak = max(
            self._maximum_airborne_streak, self._airborne_streak
        )
        self._left_contact_steps += int(left_contact)
        self._right_contact_steps += int(right_contact)
        self._double_support_steps += int(left_contact and right_contact)
        self._line_loaded_steps += int(line_force_peak > 40.0)
        self._arm_pull_loaded_steps += int(
            arm_pull_force_peak > 0.03 * self.body_weight
        )

        action_cost = float(np.mean(np.square(filtered)))
        smoothness_cost = float(np.mean(np.square(filtered - self._last_action)))
        reward = (
            35.0 * max(capped_progress_delta, 0.0)
            - 45.0 * max(-capped_progress_delta, 0.0)
            + 0.70
            * float(completed_step and previous_ascent < self.target_ascent)
            + 0.028 * metrics["any_boot_contact"]
            + 0.010 * metrics["double_support"]
            + 0.018 * float(np.clip(metrics["upright_score"], 0.0, 1.0))
            + 0.018 * metrics["grasp_score"]
            + 0.080 * min(
                arm_pull_force_peak / max(0.18 * self.body_weight, 1e-6), 1.0
            )
            - 0.020
            - 0.06 * float(not (left_contact or right_contact))
            - 0.003 * action_cost
            - 0.005 * smoothness_cost
        )

        elapsed = max(self._step_count, 1)
        left_contact_fraction = self._left_contact_steps / elapsed
        right_contact_fraction = self._right_contact_steps / elapsed
        grounded_fraction = self._grounded_steps / elapsed
        success_candidate = (
            metrics["ascent"] >= self.target_ascent
            and self._completed_steps >= 8
            and metrics["descent_from_high_water"] < 0.10
            and metrics["upright_score"] > 0.78
            and metrics["lateral_offset"] < 0.25
            and metrics["grasp_score"] > 0.32
            and metrics["hand_separation"] > 0.12
            and metrics["cross_hand_collision"] < 0.5
            and metrics["wall_hand_collision"] < 0.5
            and grounded_fraction > 0.90
            and left_contact_fraction > 0.30
            and right_contact_fraction > 0.30
            and self._maximum_airborne_streak <= 4
            and self._line_loaded_steps >= 5
            and self._arm_pull_loaded_steps >= 5
            and self._rope_core_collision_steps == 0
            and self._hand_rope_penetration_steps == 0
        )
        self._stable_success_steps = (
            self._stable_success_steps + 1 if success_candidate else 0
        )
        success = self._stable_success_steps >= 8
        failure = bool(
            metrics["ascent"] < -0.55
            or metrics["pelvis_normal_height"] < 0.42
            or metrics["lateral_offset"] > 0.70
            or metrics["upright_score"] < 0.05
            or self._airborne_streak > 20
            or not np.isfinite(self.data.qpos).all()
        )
        terminated = success or failure
        truncated = self._step_count >= self.max_episode_steps
        if success:
            reward += 120.0
        if failure:
            reward -= 30.0

        metrics.update(
            {
                "success": success,
                "failure": failure,
                "stable_success_steps": self._stable_success_steps,
                "completed_cycles": self._completed_steps,
                "grounded_fraction": grounded_fraction,
                "left_boot_contact_fraction": left_contact_fraction,
                "right_boot_contact_fraction": right_contact_fraction,
                "double_support_fraction": self._double_support_steps / elapsed,
                "maximum_airborne_streak": self._maximum_airborne_streak,
                "line_load_fraction": self._line_loaded_steps / elapsed,
                "arm_pull_load_fraction": (
                    self._arm_pull_loaded_steps / elapsed
                ),
                "arm_pull_impulse_ns": self._arm_pull_impulse_ns,
                "rope_core_collision_steps": self._rope_core_collision_steps,
                "rope_core_collision_fraction": (
                    self._rope_core_collision_steps / elapsed
                ),
                "hand_rope_penetration_steps": self._hand_rope_penetration_steps,
                "hand_rope_penetration_fraction": (
                    self._hand_rope_penetration_steps / elapsed
                ),
                "chest_load_fraction": self._line_loaded_steps / elapsed,
                "foot_loop_load_fraction": grounded_fraction,
                "line_enabled": self._line_enabled,
                "traction_enabled": self._traction_enabled,
                "foot_ascender_enabled": self._traction_enabled,
                "arm_pull_enabled": self._arm_pull_enabled,
            }
        )
        self._previous_progress = progress_absolute
        self._last_action = filtered.copy()
        return self._get_obs(), float(reward), terminated, truncated, metrics

    def set_line_enabled(self, enabled: bool) -> None:
        self._line_enabled = bool(enabled)

    def set_traction_enabled(self, enabled: bool) -> None:
        self._traction_enabled = bool(enabled)
        self.model.geom_friction[self.ice_face_geom_id] = self._slope_friction

    def set_foot_ascender_enabled(self, enabled: bool) -> None:
        """Compatibility alias: disables boot-derived uphill traction."""
        self.set_traction_enabled(enabled)

    def set_arm_pull_enabled(self, enabled: bool) -> None:
        """Enable the force-balanced handled-ascender traction path."""
        self._arm_pull_enabled = bool(enabled)

    def rebase_climb_progress(self) -> tuple[np.ndarray, dict[str, float]]:
        """Start a fresh ascent segment from the current physical stance.

        This resets task counters only.  It does not reset MuJoCo state,
        prescribe the floating base, or move any rope vertex, and is used
        after a fall/get-up handoff so the saved climbing PPO can continue.
        """

        progress = self._progress()
        self._step_count = 0
        self._completed_steps = 0
        self._expected_side = 0
        self._swing_side = None
        self._swing_phase = 0.0
        self._step_power = 0.0
        self._step_cooldown = 0
        self._request_hold_steps = 0
        self._start_progress = progress
        self._previous_progress = progress
        self._high_water_progress = progress
        self._line_ratchet_progress = progress
        self._hand_ascender_progress = progress + self.hand_ascender_reach
        self._start_pelvis_z = float(self.data.xpos[self.pelvis_body_id, 2])
        self._stable_success_steps = 0
        self._line_loaded_steps = 0
        self._arm_pull_loaded_steps = 0
        self._rope_core_collision_steps = 0
        self._hand_rope_penetration_steps = 0
        self._grounded_steps = 0
        self._left_contact_steps = 0
        self._right_contact_steps = 0
        self._double_support_steps = 0
        self._airborne_streak = 0
        self._maximum_airborne_streak = 0
        self._last_action.fill(0.0)
        self._arm_pull_command = 0.0
        self._last_line_force = 0.0
        self._last_rope_guide_force = 0.0
        self._last_arm_pull_force = 0.0
        self._arm_pull_impulse_ns = 0.0
        self._last_ground_load = 0.0
        self._sync_equipment_visuals()
        mujoco.mj_forward(self.model, self.data)
        metrics = self._metrics()
        metrics.update(
            {
                "success": False,
                "failure": False,
                "line_enabled": self._line_enabled,
                "traction_enabled": self._traction_enabled,
            }
        )
        return self._get_obs(), metrics

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = self.pelvis_body_id
        camera.distance = 2.8
        camera.azimuth = 118
        camera.elevation = -10
        self._renderer.update_scene(self.data, camera=camera)
        from fixed_line_visuals import add_braided_rope_visual

        add_braided_rope_visual(self._renderer, self.model, self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# The existing training/evaluation entry points import this conventional name.
G1FixedLineEnv = G1FixedLineSlopeEnv
