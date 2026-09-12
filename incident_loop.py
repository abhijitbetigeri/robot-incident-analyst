"""Closed-loop incident analyst: fail, diagnose, fix, re-run, file.

One command runs the whole loop the demo is about:

  1. Roll out the G1 on the fixed line with a misconfigured balance assist.
     It falls. Every step's telemetry is written to a run log.
  2. Send the run log and the environment's tunables to Gemma through the
     Respan gateway. Gemma returns a root cause and a concrete fix as JSON.
  3. Apply the fix to the live environment, re-run the SAME seed, and check
     the robot now completes the ascent.
  4. File the incident as a GitHub issue through Nango, with the before and
     after numbers in the body. If Nango is not configured, the issue payload
     is printed and saved instead so the demo never dies on a missing key.

    .venv/bin/python incident_loop.py                  # full loop
    .venv/bin/python incident_loop.py --assist 0.3     # choose the bad config
    .venv/bin/python incident_loop.py --live           # also push fix to viewer

Environment:
    RESPAN_API_KEY          falls back to ~/.respan/credentials.json
    RESPAN_MODEL            default openrouter/google/gemma-4-31b-it
    NANGO_SECRET_KEY        Nango secret key (issue step is dry-run without it)
    NANGO_CONNECTION_ID     Nango connection id for the GitHub integration
    NANGO_PROVIDER_KEY      Nango provider config key, default "github"
    GITHUB_REPO             owner/name for the issue, default abhijitbetigeri/robot-incident-analyst
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import sys
import time

import requests
from stable_baselines3 import PPO

import slip_recovery_env

def _load_env_local() -> None:
    """Read KEY=VALUE lines from .env.local next to this file. Never overrides a set var."""
    path = pathlib.Path(__file__).with_name(".env.local")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k.strip() and not v.startswith("PASTE_") and k.strip() not in os.environ:
            os.environ[k.strip()] = v


_load_env_local()

RESPAN_URL = "https://api.respan.ai/api/chat/completions"
NANGO_PROXY = "https://api.nango.dev/proxy"
DEFAULT_MODEL = "openrouter/google/gemma-4-31b-it"
FALLBACK_MODELS = ["deepinfra/google/gemma-4-31B-it", "novita/google/gemma-3-27b-it"]

# Metrics worth showing an analyst. Kept short so the prompt stays small.
LOG_KEYS = (
    "ascent", "upright_score", "pelvis_normal_height", "lateral_offset",
    "left_boot_contact", "right_boot_contact", "ground_load_bodyweight",
    "slip_active", "slip_depth_m",
)

FAILURE_RULES = {
    "ascent < -0.55 m": "slid back down the line",
    "pelvis_normal_height < 0.42 m": "pelvis dropped to the slope: a fall",
    "lateral_offset > 0.70 m": "drifted sideways off the line",
    "upright_score < 0.05": "torso horizontal: a fall",
    "airborne > 20 steps": "lost all ground contact",
}


# ---------------------------------------------------------------- rollout --

def rollout_with_log(env, policy, seed: int, every: int = 5) -> tuple[dict, list[dict]]:
    """Run one episode and keep a downsampled telemetry log plus the last 10 steps."""
    obs, _ = env.reset(seed=seed)
    log: list[dict] = []
    tail: list[dict] = []
    info: dict = {}
    steps = 0
    for _ in range(env.max_episode_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, info = env.step(action)
        steps += 1
        row = {"step": steps, **{k: round(float(info.get(k, 0.0)), 3) for k in LOG_KEYS}}
        tail.append(row)
        tail = tail[-10:]
        if steps % every == 0:
            log.append(row)
        if terminated or truncated:
            break
    for row in tail:
        if row not in log:
            log.append(row)
    log.sort(key=lambda r: r["step"])
    result = {
        "seed": seed,
        "steps": steps,
        "success": bool(info.get("success", False)),
        "failure": bool(info.get("failure", False)),
        "ascent_m": round(float(info.get("ascent", 0.0)), 3),
        "upright_score": round(float(info.get("upright_score", 0.0)), 3),
        "pelvis_normal_height": round(float(info.get("pelvis_normal_height", 0.0)), 3),
        "lateral_offset": round(float(info.get("lateral_offset", 0.0)), 3),
        "slip_triggered": bool(info.get("slip_triggered", 0.0)),
        "recovered": bool(info.get("recovered", 0.0)),
        "balance_assist_scale": round(float(env.balance_assist_scale), 2),
    }
    return result, log


def outcome(r: dict) -> str:
    if r["success"]:
        return "SUCCESS: reached the 1.5 m target"
    if r["failure"]:
        return f"FAILURE at step {r['steps']}"
    return f"timed out at step {r['steps']}"


# ---------------------------------------------------------------- respan ---

def respan_key() -> str:
    key = os.environ.get("RESPAN_API_KEY", "")
    if key:
        return key
    creds = pathlib.Path.home() / ".respan" / "credentials.json"
    if creds.exists():
        data = json.loads(creds.read_text())
        return data.get("default", {}).get("apiKey", "")
    return ""


def build_prompt(config: dict, before: dict, log: list[dict]) -> list[dict]:
    system = (
        "You are the incident analyst for a Unitree G1 humanoid climbing a fixed "
        "line on a 35-degree slope in MuJoCo. You read run telemetry, name the root "
        "cause, and propose ONE concrete fix the operator can apply by changing a "
        "tunable. Answer with a single JSON object and nothing else."
    )
    user = f"""## Episode outcome
{outcome(before)}. Final state: {json.dumps({k: before[k] for k in ('steps','ascent_m','upright_score','pelvis_normal_height','lateral_offset','slip_triggered')})}

## Configuration at the time
{json.dumps(config, indent=2)}

## Tunables you may set
- balance_assist_scale: float in [0.0, 1.0]. Scales an orientation PD on the pelvis (gain 420 at 1.0) and a lateral spring (700 N/m at 1.0). The shipped policy checkpoint was trained and validated with this at 1.0. Below about 0.5 the policy has no stabilizer to lean on.
- slip_impulse_n: shove magnitude in newtons, [0, 1200]. Lowering it makes the test easier, it does not fix the robot.

## Failure rules the environment applies
{json.dumps(FAILURE_RULES, indent=2)}

## Telemetry (policy steps at 50 Hz, downsampled, last 10 steps in full)
{json.dumps(log)}

## Respond with exactly this JSON shape
{{
  "root_cause": "one sentence",
  "evidence": ["short bullet citing a metric and step", "..."],
  "failing_step": <int>,
  "severity": "low|medium|high",
  "fix": {{"tunable": "balance_assist_scale", "value": <float>}},
  "issue_title": "short GitHub issue title",
  "expected_after_fix": "one sentence"
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def ask_gemma(messages: list[dict], model: str) -> tuple[dict, dict]:
    key = respan_key()
    if not key:
        sys.exit("No Respan API key. Set RESPAN_API_KEY or run: npx @respan/cli setup")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_err = None
    for m in [model, *FALLBACK_MODELS]:
        t0 = time.time()
        try:
            r = requests.post(RESPAN_URL, headers=headers, timeout=90, json={
                "model": m, "messages": messages, "max_tokens": 700, "temperature": 0.1,
                "metadata": {"source": "incident_loop", "robot": "g1"},
            })
            data = r.json()
            if "choices" not in data:
                last_err = data.get("error", data)
                print(f"   {m}: {str(last_err)[:120]}", flush=True)
                continue
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            meta = {"model": m, "latency_s": round(time.time() - t0, 2),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "log_id": data.get("id")}
            return parse_json(text), meta
        except (requests.RequestException, ValueError) as e:
            last_err = e
            print(f"   {m}: {e}", flush=True)
    sys.exit(f"All gateway models failed: {last_err}")


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


# ----------------------------------------------------------------- nango ---

def file_issue(repo: str, title: str, body: str, out_dir: pathlib.Path) -> str:
    payload = {"title": title, "body": body, "labels": ["incident", "auto-triage"]}
    (out_dir / "issue.json").write_text(json.dumps(payload, indent=2) + "\n")
    secret = os.environ.get("NANGO_SECRET_KEY", "")
    conn = os.environ.get("NANGO_CONNECTION_ID", "")
    if not (secret and conn):
        return "dry-run (set NANGO_SECRET_KEY and NANGO_CONNECTION_ID to file it)"
    r = requests.post(
        f"{NANGO_PROXY}/repos/{repo}/issues",
        headers={
            "Authorization": f"Bearer {secret}",
            "Provider-Config-Key": os.environ.get("NANGO_PROVIDER_KEY", "github"),
            "Connection-Id": conn,
            "Content-Type": "application/json",
        },
        json=payload, timeout=30,
    )
    if r.status_code >= 300:
        return f"Nango error {r.status_code}: {r.text[:200]}"
    return r.json().get("html_url", "filed")


def issue_body(report: dict, before: dict, after: dict, meta: dict, config: dict) -> str:
    fix = report["fix"]
    rows = [
        "| | before | after fix |", "|---|---:|---:|",
        f"| balance_assist_scale | {before['balance_assist_scale']} | {after['balance_assist_scale']} |",
        f"| outcome | {outcome(before)} | {outcome(after)} |",
        f"| steps | {before['steps']} | {after['steps']} |",
        f"| ascent | {before['ascent_m']} m | {after['ascent_m']} m |",
        f"| upright score | {before['upright_score']} | {after['upright_score']} |",
    ]
    ev = "\n".join(f"- {e}" for e in report.get("evidence", []))
    return f"""**Root cause:** {report['root_cause']}

**Evidence**
{ev}

**Fix applied:** `{fix['tunable']} = {fix['value']}` (was {config['balance_assist_scale']})

{chr(10).join(rows)}

_Diagnosed by `{meta['model']}` via Respan in {meta['latency_s']} s ({meta['prompt_tokens']} in / {meta['completion_tokens']} out). Seed {before['seed']}, impulse {config['slip_impulse_n']} N._
"""


# ------------------------------------------------------------------ main ---

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/ppo_fixed_line_slope/g1_fixed_line_final.zip")
    p.add_argument("--assist", type=float, default=0.5, help="the misconfigured assist scale")
    p.add_argument("--impulse", type=float, default=700.0)
    p.add_argument("--seed", type=int, default=4100)
    p.add_argument("--llm", default=os.environ.get("RESPAN_MODEL", DEFAULT_MODEL))
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPO", "abhijitbetigeri/robot-incident-analyst"))
    p.add_argument("--live", action="store_true", help="also send the fix to a running demo_live.py viewer")
    p.add_argument("--out", default="runs")
    a = p.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = pathlib.Path(a.out) / f"incident_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = PPO.load(a.model, device="cpu")
    config = {"balance_assist_scale": a.assist, "slip_mode": "impulse",
              "slip_impulse_n": a.impulse, "policy": a.model, "seed": a.seed}
    env = slip_recovery_env.load(disturb=True, slip_mode="impulse",
                                 slip_impulse_range=(a.impulse, a.impulse),
                                 balance_assist_scale=a.assist)

    # 1. run with the bad config
    print(f"\n[1/4] Running episode  seed={a.seed}  balance_assist_scale={a.assist}", flush=True)
    t0 = time.time()
    before, log = rollout_with_log(env, policy, a.seed)
    (out_dir / "run_before.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n")
    print(f"      {outcome(before)}  ascent {before['ascent_m']} m  "
          f"upright {before['upright_score']}  ({time.time() - t0:.1f} s, {len(log)} log rows)", flush=True)
    if before["success"]:
        print("      Episode succeeded with this config, nothing to diagnose. Lower --assist.")
        return

    # 2. diagnose through Respan
    print(f"\n[2/4] Diagnosing with {a.llm} via Respan gateway", flush=True)
    report, meta = ask_gemma(build_prompt(config, before, log), a.llm)
    (out_dir / "report.json").write_text(json.dumps({"report": report, "gateway": meta}, indent=2) + "\n")
    print(f"      root cause : {report['root_cause']}")
    for e in report.get("evidence", []):
        print(f"                   - {e}")
    print(f"      fix        : {report['fix']['tunable']} = {report['fix']['value']}")
    print(f"      gateway    : {meta['model']}  {meta['latency_s']} s  "
          f"{meta['prompt_tokens']} in / {meta['completion_tokens']} out", flush=True)

    # 3. apply the fix and re-run the same seed
    fix = report["fix"]
    value = float(fix["value"])
    if fix.get("tunable") == "balance_assist_scale":
        env.set_balance_assist_scale(max(0.0, min(1.0, value)))
    elif fix.get("tunable") == "slip_impulse_n":
        env._slip_impulse = value
    else:
        sys.exit(f"Gemma proposed an unknown tunable: {fix}")
    if a.live:
        import sim_bridge
        sim_bridge.send("assist", scale=value)
        print(f"      sent assist={value} to the live viewer", flush=True)
    print(f"\n[3/4] Re-running seed={a.seed} with {fix['tunable']}={value}", flush=True)
    t0 = time.time()
    after, log_after = rollout_with_log(env, policy, a.seed)
    (out_dir / "run_after.jsonl").write_text("\n".join(json.dumps(r) for r in log_after) + "\n")
    print(f"      {outcome(after)}  ascent {after['ascent_m']} m  "
          f"upright {after['upright_score']}  ({time.time() - t0:.1f} s)", flush=True)

    # 4. file it
    print(f"\n[4/4] Filing issue on {a.repo} via Nango", flush=True)
    body = issue_body(report, before, after, meta, config)
    where = file_issue(a.repo, report.get("issue_title", "G1 incident"), body, out_dir)
    print(f"      {where}", flush=True)

    summary = {"config": config, "before": before, "after": after, "report": report,
               "gateway": meta, "issue": where, "fixed": bool(after["success"])}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 64)
    print(f"  before   assist {before['balance_assist_scale']:<4}  {outcome(before)}")
    print(f"  after    assist {after['balance_assist_scale']:<4}  {outcome(after)}")
    print(f"  {'FIXED' if after['success'] else 'NOT FIXED'}   artifacts in {out_dir}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
