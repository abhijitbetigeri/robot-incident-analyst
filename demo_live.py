"""Live interactive demo: run the policy in a MuJoCo viewer, shove it on demand.

Real-time, not a video. A full rollout costs 5.5 s of wall clock for 23.9 s of
simulation on 8 CPU cores, so there is ample headroom to pace to real time and
still stay responsive.

macOS REQUIRES mjpython for an interactive viewer -- plain python will fail:

    .venv/bin/mjpython demo_live.py
    .venv/bin/mjpython demo_live.py --pack 12 --impulse 700

Keys
    SPACE   shove the robot (triggers the slip)
    R       reset
    ESC     quit
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

import sim_bridge
import slip_recovery_env


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/ppo_fixed_line_slope/g1_fixed_line_final.zip")
    p.add_argument("--impulse", type=float, default=700.0)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--speed", type=float, default=1.0, help="1.0 = real time.")
    a = p.parse_args()

    policy = PPO.load(a.model, device="cpu")
    # slip_step_range is set far out so nothing fires until SPACE is pressed.
    env = slip_recovery_env.load(
        disturb=True, slip_mode="impulse",
        slip_impulse_range=(a.impulse, a.impulse),
        slip_duration_range=(24, 25),
        slip_step_range=(10**8, 10**8 + 1),
        device_lead_m=0.24,
    )
    obs, _ = env.reset(seed=a.seed)

    state = {"shove": False, "reset": False}
    bridge = sim_bridge.CommandBridge()   # voice agent writes here

    def on_key(keycode: int) -> None:
        if keycode == 32:                       # SPACE
            state["shove"] = True
        elif keycode in (82, 114):              # R
            state["reset"] = True

    print(__doc__)
    with mujoco.viewer.launch_passive(
        env.model, env.data, key_callback=on_key
    ) as viewer:
        obs, _ = env.reset(seed=a.seed)
        last_phase = ""
        while viewer.is_running():
            tick = time.time()

            if state["reset"]:
                obs, _ = env.reset(seed=a.seed)
                state.update(shove=False, reset=False)
                print("\n-- reset --", flush=True)

            # Drain any voice command. Non-blocking and latest-wins: the sim
            # steps at 50 Hz and must never wait on a speech round trip.
            cmd = bridge.take()
            if cmd:
                act = cmd.get("action")
                if act == "shove":
                    state["shove"] = True
                    a.impulse = float(cmd.get("newtons", a.impulse))
                elif act == "reset":
                    state["reset"] = True
                elif act == "assist":
                    env.set_balance_assist_scale(float(cmd.get("scale", 1.0)))
                    print(f"\n>> balance assist -> {cmd.get('scale')}", flush=True)

            if state["shove"]:
                # Schedule the slip for the very next step.
                env._slip_at = env._step_count + 1
                env._slip_duration = 24
                env._slip_impulse = a.impulse
                env._slip_lateral_sign = 1.0
                state["shove"] = False
                print(f"\n>> {a.impulse:.0f} N shove", flush=True)

            action, _ = policy.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, info = env.step(action)

            phase = ("SLIP" if info.get("slip_active", 0) > 0.5
                     else "MOVING" if info.get("recovered", 0) > 0.5
                     else "RECOVER" if info.get("slip_triggered", 0) > 0.5
                     else "WALK")
            if phase != last_phase:
                print(f"   [{phase}]", flush=True)
                last_phase = phase
            if env._step_count % 25 == 0:
                print(f"\r   ascent {info['ascent']:5.2f} m   "
                      f"slip {info.get('slip_depth_m', 0):4.2f} m   {phase:8s}",
                      end="", flush=True)

            bridge.publish({**info, "phase": phase,
                            "balance_assist_scale": env.balance_assist_scale})
            viewer.sync()
            if terminated or truncated:
                print("\n-- episode over, resetting --", flush=True)
                obs, _ = env.reset(seed=a.seed)
                last_phase = ""

            # Pace to wall clock so the audience sees real-time motion.
            lag = env.policy_dt / max(a.speed, 1e-6) - (time.time() - tick)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
