"""Stage 2: induced slip and recovery on the fixed line.

`fixed_line_slope_env.py` proves the robot can ascend. It never falls, because
nothing ever disturbs it. This subclass adds the disturbance, gives the episode
a way to continue past a stumble, and measures whether the robot gets going
again -- the "recovery from a large fall" that FIXED_LINE.md lists as out of
scope.

Why a subclass and not an edit
------------------------------
The published 100%-over-ten-resets result is a property of the env exactly as
it shipped. Editing that file in place would invalidate it. Everything here is
additive, and `G1FixedLineSlopeEnv` still reproduces its own numbers.

Why the action space stays 2-D
------------------------------
`obs_dim` is a function of `action_dim`, so widening the action space changes
the observation width and the shipped checkpoint
(`models/ppo_fixed_line_slope/g1_fixed_line_final.zip`) can no longer be
loaded. Keeping it at 2 means the existing policy can be evaluated on this task
immediately, with no training -- which is the honest first experiment. "The
shipped policy cannot recover" is a result, and it is the baseline any trained
recovery policy has to beat. Widening the action space (leg-target residuals,
or a third 'recover' request) is the follow-on, and `--action-dim` is left as
the obvious extension point rather than being guessed at now.

The disturbance
---------------
Two modes, both randomized in timing and magnitude:

  friction  a verglas patch -- boot friction collapses for a window. This is
            the faithful one: it is how a real slip on a fixed line starts, and
            it is the same failure this whole project is about.
  impulse   a downslope+lateral shove on the pelvis. Less faithful, but the
            magnitude is directly controllable, which makes it the better
            instrument for a dose-response curve.

Why the balance assist has to be dialled down
---------------------------------------------
The parent applies an UNCAPPED orientation PD (gain 420) holding the pelvis at
a desired quaternion, plus an UNCAPPED lateral spring (700 N/m) pinning the
robot to y=0. Righting a stumbling humanoid is exactly what those two do. Run a
disturbance against them at full strength and the PD controller performs the
recovery -- you would be measuring a hand-tuned stabilizer, not a policy. So
this env exposes `set_balance_assist_scale`, and any recovery claim must report
the assist scale it was measured at. At scale 1.0 the number is not about the
policy.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from fixed_line_slope_env import G1FixedLineSlopeEnv


class G1SlipRecoveryEnv(G1FixedLineSlopeEnv):
    """Fixed-line ascent with an induced slip and a recovery phase."""

    def __init__(
        self,
        *args: Any,
        disturb: bool = True,
        slip_mode: str = "friction",
        slip_step_range: tuple[int, int] = (140, 420),
        slip_duration_range: tuple[int, int] = (10, 26),
        slip_friction_scale: float = 0.03,
        # Calibrated against the shipped checkpoint at full assist: 200 N is a
        # 2 cm nudge it barely notices, 700 N is a 0.35 m slip it recovers
        # from, 1100 N is a 1.02 m slide it never recovers from and which ends
        # the episode off the bottom of the slope. This range straddles the
        # interesting part of that curve instead of sitting below it.
        slip_impulse_range: tuple[float, float] = (350.0, 850.0),
        balance_assist_scale: float = 1.0,
        recovery_window: int = 220,
        recovery_progress_m: float = 0.06,
        device_lead_m: float = 0.36,
        recovery_hold_steps: int = 60,
        recovery_hold_progress_m: float = 0.10,
        recovery_required_slip_m: float = 0.08,
        **kwargs: Any,
    ) -> None:
        # A slip costs time by construction, and the parent's 1100-step budget
        # has almost none to give: measured on the shipped checkpoint, an
        # undisturbed ascent crosses the 1.5 m target at step 1075 and needs 8
        # consecutive holding steps, finishing at 1082 -- 18 steps of margin.
        # A disturbed run crossed at 1095 and truncated with 6 of the 8 steps
        # banked, so it scored as a failure purely on the clock while ascending
        # 1.521 m and recovering cleanly. Give the recovery task room, or every
        # measurement is really a stopwatch.
        kwargs.setdefault("max_episode_steps", 1500)
        super().__init__(*args, **kwargs)
        self.disturb = bool(disturb)
        self.slip_mode = str(slip_mode)
        self.slip_step_range = slip_step_range
        self.slip_duration_range = slip_duration_range
        self.slip_friction_scale = float(slip_friction_scale)
        self.slip_impulse_range = slip_impulse_range
        self.recovery_window = int(recovery_window)
        self.recovery_progress_m = float(recovery_progress_m)
        self.device_lead_m = float(device_lead_m)
        self.recovery_hold_steps = int(recovery_hold_steps)
        self.recovery_hold_progress_m = float(recovery_hold_progress_m)
        self.recovery_required_slip_m = float(recovery_required_slip_m)

        # Geoms that actually touch the snow, so the verglas patch is applied to
        # the boots rather than to the whole robot.
        self._boot_geom_ids = np.asarray(
            [
                gid
                for gid in range(self.model.ngeom)
                if int(self.model.geom_bodyid[gid]) in set(self._foot_body_ids.values())
            ],
            dtype=np.int32,
        )
        self._boot_friction_default = self.model.geom_friction[
            self._boot_geom_ids
        ].copy()

        # Snapshot the parent's assist gains once, so a scale is always applied
        # to the shipped values and never compounds across calls.
        self._assist_defaults = {
            name: float(getattr(self, name))
            for name in (
                "lateral_stiffness",
                "lateral_damping",
                "orientation_stiffness",
                "orientation_damping",
                "normal_balance_stiffness",
                "normal_balance_damping",
                "max_normal_balance_force",
            )
        }
        self._balance_assist_scale = 1.0
        self.set_balance_assist_scale(balance_assist_scale)
        # The left hand ships with its finger targets at ~0, i.e. fully open.
        # It is also the only limb with no IK target at all (`_arm_target_positions`
        # returns "right" only), so it is parked -- 3.2 cm of motion in the torso
        # frame across an entire climb, versus 1.02 m for the right. The result
        # renders as a lifeless splayed hand held out in front of the rope, which
        # is the single thing that most makes the robot look broken.
        #
        # FIXED_LINE.md describes it as "bent, open, and clear for balance". It is
        # not doing balance -- nothing commands it -- but a relaxed climber's hand
        # is curled, not splayed. No DOF and no contacts change; the pose is
        # otherwise cosmetic. It is NOT dynamically free, though, and claiming so
        # would be wrong: curling the fingers shifts the forearm inertia. Matched
        # comparison, seed 5, lead 0.24, no disturbance:
        #     shipped open : ascent 1.5105 m, 1315 steps, success
        #     relaxed      : ascent 1.5239 m, 1311 steps, success
        # ~0.9% apart, both succeeding. Small, but not zero -- quote it as a
        # cosmetic change with a measurable footprint, not a free one.
        # The left ARM pose is the "wave", not the fingers. Shipped nominal:
        #   shoulder_pitch -0.420  <- NEGATIVE pitch swings the arm back and UP
        #   shoulder_roll  +0.340  <- abducted outward
        #   elbow          +0.880
        # which parks it raised behind the head with an open hand, for the whole
        # climb, because `_arm_target_positions()` returns "right" only and
        # nothing ever commands the left arm (measured: 3.2 cm of motion in the
        # torso frame across a full ascent, vs 1.02 m for the right).
        # A climber's free hand hangs low and slightly out, elbow soft.
        RELAXED_LEFT = {
        # Pitch -0.20, not the -0.22-forward pose that first looked right.
        # The left arm is NOT cosmetic under disturbance: its inertia matters
        # for slip recovery. Measured at a 700 N shove, seed 5, lead 0.24:
        #     pitch -0.42 (shipped wave) : recovered, ascent 1.519 m
        #     pitch -0.20 (this)         : recovered, ascent 2.075 m
        #     pitch +0.22 (arm down)     : NOT recovered, ascent 0.863 m,
        #                                  episode over at step 355
        # Dropping the arm fully looked better standing still and cost the
        # recovery outright. This lowers it out of the wave and keeps the
        # recovery, which is the only version that is both.
            "left_shoulder_pitch_joint": -0.20,
            "left_shoulder_roll_joint": 0.30,
            "left_shoulder_yaw_joint": 0.00,
            "left_elbow_joint": 0.75,
            "left_wrist_pitch_joint": 0.05,
            "left_hand_thumb_0_joint": 0.10,
            "left_hand_thumb_1_joint": 0.30,
            "left_hand_thumb_2_joint": 0.45,
            "left_hand_index_0_joint": -0.55,
            "left_hand_index_1_joint": -0.85,
            "left_hand_middle_0_joint": -0.55,
            "left_hand_middle_1_joint": -0.85,
        }
        for name, value in RELAXED_LEFT.items():
            i = self._actuator_ids.get(name)
            if i is not None:
                lo, hi = self.model.actuator_ctrlrange[i]
                self._nominal_ctrl[i] = float(np.clip(value, lo, hi))

        self.torso_body_id_for_grip = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        self._reset_slip_state()

    def _device_points(self):
        """Ascender lead along the rope. Defaults to the shipped 0.36 m.

        Exposed because the right lead depends on the LOAD, which was not
        obvious and which I first got wrong. A sweep that appeared to show the
        shipped 0.36 m lead failing (40 cm grip error) had a 12 kg pack still
        attached. Re-measured properly, seed 5, after settling:

            unloaded, lead 0.36 : 11.7 cm mean grip error, grasp 0.524, ok
            12 kg,    lead 0.36 : 40.2 cm mean, grasp 0.269, ascender detaches
            12 kg,    lead 0.24 : 11.9 cm mean, grasp 0.494
            12 kg,    lead 0.16 : 12.0 cm mean, grasp 0.461

        So the shipped value is correct for the unloaded task, and the arm
        tracking is a second casualty of load: the pack pitches the torso, the
        IK target moves relative to the shoulder, and at 0.36 m it goes out of
        reach. Carrying a load therefore needs a shorter lead as well as
        whatever fixes the gait.

        Default unchanged so nothing silently differs from the parent.
        """
        progress = self._progress()
        return (self._rope_point(progress + 0.05),
                self._rope_point(progress + self.device_lead_m))

    # ---- balance assist ------------------------------------------------

    def set_balance_assist_scale(self, scale: float) -> None:
        """Scale every external stabilizer term. 1.0 = shipped, 0.0 = off."""
        scale = float(max(scale, 0.0))
        self._balance_assist_scale = scale
        for name, default in self._assist_defaults.items():
            setattr(self, name, default * scale)

    def set_balance_assist_enabled(self, enabled: bool) -> None:
        """The ablation toggle the parent never had."""
        self.set_balance_assist_scale(1.0 if enabled else 0.0)

    @property
    def balance_assist_scale(self) -> float:
        return self._balance_assist_scale

    # ---- disturbance ---------------------------------------------------

    def _reset_slip_state(self) -> None:
        self._slip_at = -1
        self._slip_duration = 0
        self._slip_impulse = 0.0
        self._slip_lateral_sign = 1.0
        self._slip_active = False
        self._slip_triggered = False
        self._slip_started_progress = 0.0
        self._slip_depth = 0.0
        self._recovered = False
        self._recovered_at = -1
        self._recovery_candidate_at = -1
        self._progress_at_candidate = 0.0
        self._recovery_deadline = -1
        self._progress_at_slip_end = 0.0
        self._boot_friction_restored = True
        self._airborne_max_before_slip = 0
        self._airborne_credited = False

    def _apply_boot_friction(self, scale: float) -> None:
        self.model.geom_friction[self._boot_geom_ids] = (
            self._boot_friction_default * scale
        )

    def _apply_support_forces(self):
        """Parent assist, plus the impulse-mode shove while the slip is live."""
        # Signature-agnostic: the parent's return arity changed in b92ab58
        # (the fall-recovery rework). Pass whatever it gives back through
        # unchanged rather than unpacking a fixed number of values.
        result = super()._apply_support_forces()
        if self._slip_active and self.slip_mode == "impulse":
            # Downslope plus a lateral component, so the slip is not perfectly
            # in-plane -- a purely downslope shove is the one case the lateral
            # spring is best at rejecting.
            shove = -self._slip_impulse * self.uphill
            shove[1] += 0.35 * self._slip_impulse * self._slip_lateral_sign
            mujoco.mj_applyFT(
                self.model,
                self.data,
                shove,
                np.zeros(3),
                self.data.xpos[self.pelvis_body_id],
                self.pelvis_body_id,
                self.data.qfrc_applied,
            )
        return result

    # ---- episode -------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        self._apply_boot_friction(1.0)
        self._reset_slip_state()
        observation, metrics = super().reset(seed=seed, options=options)
        if self.disturb:
            self._slip_at = int(self.np_random.integers(*self.slip_step_range))
            self._slip_duration = int(self.np_random.integers(*self.slip_duration_range))
            self._slip_impulse = float(self.np_random.uniform(*self.slip_impulse_range))
            self._slip_lateral_sign = 1.0 if self.np_random.random() < 0.5 else -1.0
        metrics.update(self._slip_metrics())
        return observation, metrics

    # NOTE: no wrist-orientation override. The visible hand twist was traced
    # to `_wrist_target_quaternions` being identity [1,0,0,0] -- a fixed WORLD
    # orientation held while the robot climbs a pitching 28 deg slope, leaving
    # the wrist 12.4-26.1 deg off the torso and drifting 12 deg across a climb.
    #
    # That diagnosis did not survive testing. Two replacements, seed 5:
    #     rope-aligned target : twist range 13.7 -> 176.4 deg, success LOST
    #     torso-relative      : twist range 13.7 -> 13.8 deg, no change
    #
    # Torso-relative is the semantically correct target and changes nothing,
    # which means the orientation command is not what determines wrist pose
    # here: `_solve_arm_ik` is damped least squares weighted to POSITION, so it
    # reaches the device point and lets orientation fall where it will. Fixing
    # the twist means reweighting or constraining that solve, not relabelling
    # the target frame. Reverted to the parent rather than ship a change that
    # measured as a no-op.

    def _slip_metrics(self) -> dict[str, float]:
        return {
            "slip_scheduled_step": float(self._slip_at),
            "slip_triggered": float(self._slip_triggered),
            "slip_active": float(self._slip_active),
            "slip_depth_m": float(self._slip_depth),
            "recovered": float(self._recovered),
            "steps_to_recover": float(
                self._recovered_at - (self._slip_at + self._slip_duration)
                if self._recovered
                else -1
            ),
            "balance_assist_scale": float(self._balance_assist_scale),
            "airborne_streak_excused": float(self._airborne_credited),
            "recovery_required": float(
                self._slip_triggered
                and self._slip_depth >= self.recovery_required_slip_m
            ),
        }

    def step(self, action):
        # Open/close the disturbance window BEFORE physics, so the flag the
        # substep loop reads is the one this step intends.
        step_index = self._step_count + 1
        if self.disturb and self._slip_at >= 0:
            in_window = self._slip_at <= step_index < self._slip_at + self._slip_duration
            if in_window and not self._slip_active:
                self._slip_active = True
                self._slip_triggered = True
                self._slip_started_progress = self._progress()
                # The parent's success gate needs max_airborne_streak <= 3 as a
                # MAXIMUM OVER THE WHOLE EPISODE. A real slip blows straight
                # through it -- measured at a 700 N shove, the streak hits 14 --
                # and because it is a running max it can never be un-tripped.
                # So a robot that slipped 0.35 m, recovered, and went on to
                # climb 2.66 m scored as a failure on that gate alone. Snapshot
                # the pre-slip value here and restore it once recovery lands.
                self._airborne_max_before_slip = self._maximum_airborne_streak
                if self.slip_mode == "friction":
                    self._apply_boot_friction(self.slip_friction_scale)
                    self._boot_friction_restored = False
            elif not in_window and self._slip_active:
                self._slip_active = False
                if not self._boot_friction_restored:
                    self._apply_boot_friction(1.0)
                    self._boot_friction_restored = True
                self._progress_at_slip_end = self._progress()
                self._recovery_deadline = step_index + self.recovery_window

        progress_before = self._progress()
        observation, reward, terminated, truncated, metrics = super().step(action)
        progress_after = self._progress()

        if self._slip_triggered:
            self._slip_depth = max(
                self._slip_depth, self._slip_started_progress - progress_after
            )

        base_success = bool(metrics.get("success", False))
        base_failure = bool(metrics.get("failure", False))

        # 1. The induced slip is not the policy's fault. The parent charges
        #    -45 per metre of backward motion; refund it inside the window so
        #    the disturbance itself is not the dominant term in the return.
        if self._slip_active:
            reward += 45.0 * max(-(progress_after - progress_before), 0.0)

        # 2. A stumble must not end the episode before recovery can be tried.
        #    The parent's own gates stay in force for the unrecoverable ones
        #    (slid off the slope, non-finite state); only the postural
        #    failures are suspended, and only inside the recovery window.
        recoverable = bool(
            metrics.get("pelvis_normal_height", 1.0) >= 0.28
            and metrics.get("lateral_offset", 0.0) <= 0.70
            and metrics.get("ascent", 0.0) > -0.55
            and np.isfinite(self.data.qpos).all()
        )
        in_recovery = self._slip_triggered and (
            self._slip_active or 0 <= step_index <= self._recovery_deadline
        )
        if base_failure and in_recovery and recoverable:
            terminated = False
            reward += 30.0  # undo the parent's failure charge
            base_failure = False

        # 3. Recovery credit: both boots back down, and real uphill progress
        #    made from where the slip left the robot. Both halves matter --
        #    standing still upright is not recovery.
        # Recovery is provisional first, then confirmed. Regaining double
        # support and 6 cm is only a CANDIDATE: measured at an 800 N shove, the
        # robot cleared that bar 16 steps after the slip, was credited as
        # recovered, then collapsed and ended the episode at step 352 having
        # climbed 1.031 m of a 1.5 m target. Crediting that as a recovery is
        # how you publish a number for something the robot did not do. The task
        # is walk -> slip -> recover -> KEEP MOVING, so confirmation requires
        # surviving `recovery_hold_steps` more and adding real distance on top.
        if (
            self._slip_triggered
            and not self._slip_active
            and not self._recovered
            and self._recovery_candidate_at < 0
            and metrics.get("double_support", 0.0) > 0.5
            and (progress_after - self._progress_at_slip_end)
            >= self.recovery_progress_m
        ):
            self._recovery_candidate_at = step_index
            self._progress_at_candidate = progress_after
            reward += 5.0

        if (
            self._recovery_candidate_at > 0
            and not self._recovered
            and step_index - self._recovery_candidate_at >= self.recovery_hold_steps
            and (progress_after - self._progress_at_candidate)
            >= self.recovery_hold_progress_m
        ):
            self._recovered = True
            self._recovered_at = step_index
            reward += 25.0
            # The gate exists to catch a policy that hops through its normal
            # gait, not to punish an externally induced stumble it then
            # recovered from. Rewind to the pre-slip maximum; any instability
            # AFTER recovery still accumulates from that baseline and still
            # counts against success.
            self._maximum_airborne_streak = self._airborne_max_before_slip
            self._airborne_credited = True

        # A candidate that stops making progress is not a recovery in
        # progress -- drop it so a later, genuine one can still qualify.
        if (
            self._recovery_candidate_at > 0
            and not self._recovered
            and step_index - self._recovery_candidate_at >= self.recovery_hold_steps
        ):
            self._recovery_candidate_at = -1

        # An ABSORBED disturbance must also have the airborne gate rewound.
        # The rewind originally fired only on confirmed recovery, so a 400 N
        # shove -- too small to require a recovery, but big enough to break the
        # parent's `max_airborne_streak <= 3` whole-episode gate -- could climb
        # 2.063 m and still never score a success. Nothing recovers it, because
        # nothing needed recovering.
        if (
            self._slip_triggered
            and not self._slip_active
            and not self._airborne_credited
            and self._slip_depth < self.recovery_required_slip_m
        ):
            self._maximum_airborne_streak = self._airborne_max_before_slip
            self._airborne_credited = True

        # 4. Ran out of recovery window without getting going again.
        #    Only applies to a slip worth recovering FROM. A 400 N shove
        #    displaces the robot 3.8 cm and it walks on untroubled -- but the
        #    deadline was armed by any disturbance at all, so that episode was
        #    killed at step 545 with 0.692 m of ascent and scored a failure,
        #    while a 700 N shove that genuinely slipped 0.35 m scored 1.520 m.
        #    A smaller shove producing a worse result is the signature of a
        #    metric bug, not a policy limit. You do not recover from a nudge
        #    you absorbed.
        if (
            self._slip_triggered
            and self._slip_depth >= self.recovery_required_slip_m
            and not self._recovered
            and self._recovery_deadline > 0
            and step_index > self._recovery_deadline
        ):
            terminated = True
            base_failure = True
            reward -= 30.0

        # 5. The parent disqualifies an episode for one airborne streak > 3
        #    steps, anywhere. Under a deliberate slip that gate can never be
        #    met, so success here additionally requires recovery when a slip
        #    was induced -- ascent alone is not enough.
        needs_recovery = (
            self._slip_triggered
            and self._slip_depth >= self.recovery_required_slip_m
        )
        success = bool(base_success and (self._recovered or not needs_recovery))
        if base_success and not success:
            reward -= 100.0  # withdraw the parent's bonus; it did not recover
            terminated = terminated and not base_success

        metrics.update(self._slip_metrics())
        metrics["success"] = success
        metrics["failure"] = base_failure
        metrics["recovery_task"] = True
        return observation, float(reward), bool(terminated), bool(truncated), metrics


def load(**kwargs) -> G1SlipRecoveryEnv:
    return G1SlipRecoveryEnv(**kwargs)
