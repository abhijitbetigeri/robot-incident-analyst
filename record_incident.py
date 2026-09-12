"""Record the closed loop as a video: robot view on the left, analyst on the right.

Runs the same four stages as incident_loop.py, but renders every rollout step
from the MuJoCo tracking camera and composes it with a live text panel, so the
recording IS the demo rather than a screen capture of it.

    .venv/bin/python record_incident.py                # writes submission/demo.mp4
    .venv/bin/python record_incident.py --no-issue     # skip filing on GitHub
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import textwrap
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

import incident_loop as il
import slip_recovery_env

W_ROBOT, H = 1280, 720
W_PANEL = 640
FPS = 25
BG = (20, 28, 38)
INK = (242, 245, 248)
MUTED = (159, 176, 192)
TEAL = (63, 179, 176)
AMBER = (232, 154, 74)
RED = (235, 90, 90)
GREEN = (98, 200, 120)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size, index=1 if bold else 0)
    except OSError:
        return ImageFont.load_default()


F_TITLE, F_HEAD, F_BODY, F_SMALL = font(26, True), font(21, True), font(19), font(16)


def panel(lines: list[tuple[str, tuple[int, int, int], ImageFont.FreeTypeFont]]) -> np.ndarray:
    """Render a list of (text, color, font) lines into the right-hand panel."""
    img = Image.new("RGB", (W_PANEL, H), BG)
    d = ImageDraw.Draw(img)
    y = 36
    for text, color, f in lines:
        if text == "":
            y += 12
            continue
        d.text((36, y), text, fill=color, font=f)
        y += int(f.size * 1.45)
    return np.asarray(img)


def wrap(text: str, width: int = 44) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


def compose(robot: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([robot, right], axis=1)


def live_lines(stage: str, cfg_scale: float, seed: int, row: dict | None, status: tuple[str, tuple] | None):
    lines = [("ROBOT INCIDENT ANALYST", INK, F_TITLE), ("", INK, F_BODY),
             (stage, TEAL, F_HEAD),
             (f"seed={seed}  balance_assist_scale={cfg_scale}", MUTED, F_SMALL), ("", INK, F_BODY)]
    if row:
        lines += [
            (f"step            {row['step']:>6d}", INK, F_BODY),
            (f"ascent          {row['ascent']:>6.3f} m", INK, F_BODY),
            (f"pelvis height   {row['pelvis_normal_height']:>6.3f} m   (fail < 0.42)", INK, F_BODY),
            (f"upright score   {row['upright_score']:>6.3f}", INK, F_BODY),
            (f"boot contact    L {int(row['left_boot_contact'])}  R {int(row['right_boot_contact'])}", INK, F_BODY),
        ]
    if status:
        lines += [("", INK, F_BODY), (status[0], status[1], F_HEAD)]
    return lines


def rollout_render(env, policy, seed, writer, stage, cfg_scale, every=1, hold=40, end_color=RED):
    obs, _ = env.reset(seed=seed)
    log, tail, info, steps = [], [], {}, 0
    last_frame = None
    for _ in range(env.max_episode_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, info = env.step(action)
        steps += 1
        row = {"step": steps, **{k: round(float(info.get(k, 0.0)), 3) for k in il.LOG_KEYS}}
        tail.append(row); tail = tail[-10:]
        if steps % 5 == 0:
            log.append(row)
        if steps % every == 0:
            last_frame = env.render()
            writer.append_data(compose(last_frame, panel(live_lines(stage, cfg_scale, seed, row, None))))
        if terminated or truncated:
            break
    for r in tail:
        if r not in log:
            log.append(r)
    log.sort(key=lambda r: r["step"])
    result = {
        "seed": seed, "steps": steps,
        "success": bool(info.get("success", False)), "failure": bool(info.get("failure", False)),
        "ascent_m": round(float(info.get("ascent", 0.0)), 3),
        "upright_score": round(float(info.get("upright_score", 0.0)), 3),
        "pelvis_normal_height": round(float(info.get("pelvis_normal_height", 0.0)), 3),
        "lateral_offset": round(float(info.get("lateral_offset", 0.0)), 3),
        "slip_triggered": bool(info.get("slip_triggered", 0.0)),
        "recovered": bool(info.get("recovered", 0.0)),
        "balance_assist_scale": round(float(env.balance_assist_scale), 2),
    }
    if last_frame is None:
        last_frame = env.render()
    status = (il.outcome(result), GREEN if result["success"] else end_color)
    frame = compose(last_frame, panel(live_lines(stage, cfg_scale, seed, tail[-1], status)))
    for _ in range(hold):
        writer.append_data(frame)
    return result, log, last_frame


def hold(writer, frame, seconds):
    for _ in range(int(seconds * FPS)):
        writer.append_data(frame)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/ppo_fixed_line_slope/g1_fixed_line_final.zip")
    p.add_argument("--assist", type=float, default=0.5)
    p.add_argument("--impulse", type=float, default=700.0)
    p.add_argument("--seed", type=int, default=4100)
    p.add_argument("--llm", default=il.DEFAULT_MODEL)
    p.add_argument("--repo", default=il.os.environ.get("GITHUB_REPO", "abhijitbetigeri/robot-incident-analyst"))
    p.add_argument("--no-issue", action="store_true")
    p.add_argument("--out", default="submission/demo.mp4")
    a = p.parse_args()

    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    art = pathlib.Path("runs") / f"recording_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    art.mkdir(parents=True, exist_ok=True)

    policy = PPO.load(a.model, device="cpu")
    config = {"balance_assist_scale": a.assist, "slip_mode": "impulse",
              "slip_impulse_n": a.impulse, "policy": a.model, "seed": a.seed}
    env = slip_recovery_env.load(disturb=True, slip_mode="impulse", render_mode="rgb_array",
                                 slip_impulse_range=(a.impulse, a.impulse),
                                 balance_assist_scale=a.assist)
    writer = imageio.get_writer(str(out), fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    t0 = time.time()

    # 1. fail
    print("[1/4] rendering the failing run", flush=True)
    before, log, fall_frame = rollout_render(env, policy, a.seed, writer,
                                             "[1/4] Running episode", a.assist, every=1)
    print(f"      {il.outcome(before)}", flush=True)

    # 2. diagnose
    print(f"[2/4] diagnosing via Respan / {a.llm}", flush=True)
    thinking = compose(fall_frame, panel(live_lines("[2/4] Diagnosing", a.assist, a.seed, None,
                                                    (f"sending {len(log)} telemetry rows to", MUTED))
                                         + [(f"{a.llm}", TEAL, F_BODY), ("via Respan gateway ...", MUTED, F_BODY)]))
    hold(writer, thinking, 2.0)
    report, meta = il.ask_gemma(il.build_prompt(config, before, log), a.llm)
    fix = report["fix"]
    lines = [("ROBOT INCIDENT ANALYST", INK, F_TITLE), ("", INK, F_BODY),
             ("[2/4] Diagnosis", TEAL, F_HEAD),
             (f"{meta['model']}", MUTED, F_SMALL),
             (f"{meta['latency_s']} s   {meta['prompt_tokens']} in / {meta['completion_tokens']} out", MUTED, F_SMALL),
             ("", INK, F_BODY), ("ROOT CAUSE", AMBER, F_SMALL)]
    lines += [(l, INK, F_BODY) for l in wrap(report["root_cause"])]
    lines += [("", INK, F_BODY), ("EVIDENCE", AMBER, F_SMALL)]
    for e in report.get("evidence", [])[:3]:
        ws = wrap(e, 42)
        lines += [("- " + ws[0], INK, F_SMALL)] + [("  " + w, INK, F_SMALL) for w in ws[1:]]
    lines += [("", INK, F_BODY), ("FIX", AMBER, F_SMALL),
              (f"{fix['tunable']} = {fix['value']}", GREEN, F_HEAD)]
    hold(writer, compose(fall_frame, panel(lines)), 8.0)
    print(f"      fix: {fix['tunable']} = {fix['value']}", flush=True)

    # 3. fix and re-run
    value = float(fix["value"])
    if fix.get("tunable") == "balance_assist_scale":
        env.set_balance_assist_scale(max(0.0, min(1.0, value)))
    elif fix.get("tunable") == "slip_impulse_n":
        env._slip_impulse = value
    print("[3/4] rendering the re-run", flush=True)
    after, _log2, ok_frame = rollout_render(env, policy, a.seed, writer,
                                            f"[3/4] Re-running with fix", value, every=2, hold=40)
    print(f"      {il.outcome(after)}", flush=True)

    # 4. file
    where = "not filed (--no-issue)"
    if not a.no_issue:
        print("[4/4] filing issue via Nango", flush=True)
        where = il.file_issue(a.repo, report.get("issue_title", "Robot incident"),
                              il.issue_body(report, before, after, meta, config), art)
        print(f"      {where}", flush=True)
    lines = [("ROBOT INCIDENT ANALYST", INK, F_TITLE), ("", INK, F_BODY),
             ("[4/4] Issue filed via Nango", TEAL, F_HEAD)]
    lines += [(l, MUTED, F_SMALL) for l in wrap(where, 48)]
    lines += [("", INK, F_BODY),
              (f"{'':16}{'before':>10}{'after':>10}", MUTED, F_SMALL),
              (f"{'assist scale':16}{before['balance_assist_scale']:>10}{after['balance_assist_scale']:>10}", INK, F_BODY),
              (f"{'ascent (m)':16}{before['ascent_m']:>10.3f}{after['ascent_m']:>10.3f}", INK, F_BODY),
              (f"{'steps':16}{before['steps']:>10}{after['steps']:>10}", INK, F_BODY),
              (f"{'outcome':16}{'FALL':>10}{'SUCCESS':>10}", INK, F_BODY),
              ("", INK, F_BODY),
              ("Respan gateway  ->  Gemma 4 31B", MUTED, F_SMALL),
              ("Nango           ->  GitHub issue", MUTED, F_SMALL),
              ("Lambda          ->  student fine-tune (next)", MUTED, F_SMALL)]
    hold(writer, compose(ok_frame, panel(lines)), 6.0)
    writer.close()
    env.close()
    (art / "summary.json").write_text(json.dumps({"before": before, "after": after, "report": report,
                                                  "gateway": meta, "issue": where}, indent=2) + "\n")
    print(f"\nwrote {out}  ({time.time() - t0:.0f} s)  artifacts in {art}", flush=True)


if __name__ == "__main__":
    main()
