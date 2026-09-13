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
    .venv/bin/python incident_loop.py --scenario traction   # or: assist, line
    .venv/bin/python incident_loop.py --live           # also push fix to viewer

Environment:
    RESPAN_API_KEY          falls back to ~/.respan/credentials.json
    RESPAN_MODEL            default openrouter/google/gemma-4-31b-it
    NANGO_SECRET_KEY        Nango secret key (issue step is dry-run without it)
    NANGO_CONNECTION_ID     Nango connection id for the GitHub integration
    NANGO_PROVIDER_KEY      Nango provider config key, default "github"
    GITHUB_REPO             owner/name for the issue, default abhijitbetigeri/robot-incident-analyst
    LAMBDA_HOST / LAMBDA_ENABLED   reviewer model served on a Lambda GPU instance, see lambda_reviewer.sh
    LAMBDA_URL              OpenAI-compatible URL of that model, default http://127.0.0.1:11434/v1 (ssh tunnel)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import sys
import textwrap
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
    "ascent", "uphill_speed", "upright_score", "pelvis_normal_height", "lateral_offset",
    "left_boot_contact", "right_boot_contact", "ground_load_bodyweight", "line_load_n",
    "slip_active", "slip_depth_m",
)

# Misconfigurations the demo can start from. Each is a real failure with a
# real fix: the analyst has to find which tunable is wrong from the telemetry.
SCENARIOS = {
    "assist": {
        "title": "Balance assist dialled down",
        "overrides": {"balance_assist_scale": 0.5},
    },
    "traction": {
        "title": "Boot traction disabled",
        "overrides": {"boot_traction_enabled": False},
    },
    "line": {
        "title": "Fixed line disconnected",
        "overrides": {"fixed_line_enabled": False},
    },
}

TUNABLES_DOC = """- balance_assist_scale: float in [0.0, 1.0]. Scales an orientation PD on the pelvis (gain 420 at 1.0) and a lateral spring (700 N/m at 1.0). The shipped policy checkpoint was trained and validated with this at 1.0. Below about 0.5 the policy has no stabilizer to lean on.
- boot_traction_enabled: bool. Crampon-style uphill traction of the boots on the ice face. When false the boots have only bare ice friction and the robot cannot push uphill.
- fixed_line_enabled: bool. The rope and ascender connection to the fixed line. When false the robot is not clipped in: line_load_n stays 0 and nothing arrests a slide.
- slip_impulse_n: shove magnitude in newtons, [0, 1200]. Lowering it makes the test easier, it does not fix the robot."""


def make_env(scenario: str, impulse: float, **kw):
    """Build the env with the scenario's misconfiguration applied."""
    ov = SCENARIOS[scenario]["overrides"]
    env = slip_recovery_env.load(disturb=True, slip_mode="impulse",
                                 slip_impulse_range=(impulse, impulse),
                                 balance_assist_scale=ov.get("balance_assist_scale", 1.0), **kw)
    if not ov.get("boot_traction_enabled", True):
        env.set_traction_enabled(False)
    if not ov.get("fixed_line_enabled", True):
        env.set_line_enabled(False)
    return env


def config_of(env, impulse: float, seed: int, policy_path: str) -> dict:
    return {
        "balance_assist_scale": round(float(env.balance_assist_scale), 2),
        "boot_traction_enabled": bool(getattr(env, "_traction_enabled", True)),
        "fixed_line_enabled": bool(getattr(env, "_line_enabled", True)),
        "slip_mode": "impulse", "slip_impulse_n": impulse, "policy": policy_path, "seed": seed,
    }


def _as_bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on", "enabled")


def apply_fix(env, fix: dict):
    """Apply the analyst's fix to the live env. Returns the normalized value."""
    t = fix.get("tunable")
    if t == "balance_assist_scale":
        v = max(0.0, min(1.0, float(fix["value"]))); env.set_balance_assist_scale(v); return v
    if t == "boot_traction_enabled":
        v = _as_bool(fix["value"]); env.set_traction_enabled(v); return v
    if t == "fixed_line_enabled":
        v = _as_bool(fix["value"]); env.set_line_enabled(v); return v
    if t == "slip_impulse_n":
        v = float(fix["value"]); env._slip_impulse = v; return v
    sys.exit(f"Analyst proposed an unknown tunable: {fix}")

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

## Tunables you may set (pick the ONE whose current value explains the failure)
{TUNABLES_DOC}

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
  "fix": {{"tunable": "<one of the tunables above>", "value": <float or bool>}},
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


# ---------------------------------------------------------------- lambda ---

# OpenAI-compatible endpoint of the reviewer model served on a Lambda GPU
# instance (Ollama or vLLM), reached through an SSH tunnel by default.
LAMBDA_URL = os.environ.get("LAMBDA_URL", "http://127.0.0.1:11434/v1").rstrip("/")
# Preference order for the reviewer model. First one the server lists wins;
# substrings, matched case-insensitively.
LAMBDA_PREFERRED = ("gemma", "llama-3.3-70b", "llama3.3-70b", "llama3.1-70b", "qwen", "deepseek", "hermes")


def lambda_key() -> str:
    # Ollama ignores the bearer token; vLLM may require one. Any non-empty
    # value enables the review stage.
    return os.environ.get("LAMBDA_API_KEY", "") or os.environ.get("LAMBDA_ENABLED", "")


def lambda_pick_model() -> str | None:
    forced = os.environ.get("LAMBDA_MODEL")
    if forced:
        return forced
    try:
        r = requests.get(f"{LAMBDA_URL}/models", headers={"Authorization": f"Bearer {lambda_key()}"}, timeout=20)
        ids = [m["id"] for m in r.json().get("data", [])]
    except (requests.RequestException, ValueError, KeyError):
        return None
    for pref in LAMBDA_PREFERRED:
        for i in ids:
            if pref in i.lower():
                return i
    return ids[0] if ids else None


def review_on_lambda(report: dict, before: dict, after: dict, log_before: list[dict],
                     log_after: list[dict]) -> tuple[dict | None, dict | None]:
    """Independent second opinion from an open model served by Lambda Inference.

    The reviewer never sees the analyst's fix as ground truth: it gets the
    diagnosis as a claim plus the raw before/after telemetry, and must say
    whether the re-run actually supports the claim."""
    if not lambda_key():
        return None, None
    model = lambda_pick_model()
    if not model:
        return None, {"error": "no models visible to this Lambda key"}
    system = ("You are a second-opinion reviewer for robot incident reports. You are given a "
              "diagnosis another model produced and the telemetry from the failed run and the "
              "re-run after its fix was applied. Judge whether the re-run supports the diagnosis. "
              "Answer with a single JSON object and nothing else.")
    user = f"""## Claimed diagnosis
{json.dumps({k: report.get(k) for k in ('root_cause', 'evidence', 'fix')}, indent=2)}

## Failed run: final state and downsampled telemetry
{json.dumps(before)}
{json.dumps(log_before[-12:])}

## Re-run with the fix: final state and downsampled telemetry
{json.dumps(after)}
{json.dumps(log_after[-12:])}

## Respond with exactly this JSON shape
{{"verdict": "confirmed|rejected|inconclusive", "confidence": <0.0-1.0>,
  "reasoning": "two sentences citing metrics", "residual_risk": "one sentence"}}"""
    t0 = time.time()
    try:
        r = requests.post(f"{LAMBDA_URL}/chat/completions", timeout=90,
                          headers={"Authorization": f"Bearer {lambda_key()}", "Content-Type": "application/json"},
                          json={"model": model, "max_tokens": 400, "temperature": 0.1,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": user}]})
        data = r.json()
        if "choices" not in data:
            return None, {"model": model, "error": str(data.get("error", data))[:200]}
        usage = data.get("usage", {})
        meta = {"model": model, "latency_s": round(time.time() - t0, 2),
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
        return parse_json(data["choices"][0]["message"]["content"]), meta
    except (requests.RequestException, ValueError) as e:
        return None, {"model": model, "error": str(e)[:200]}


# ----------------------------------------------------------------- nango ---

def file_issue(repo: str, title: str, body: str, out_dir: pathlib.Path) -> str:
    payload = {"title": title, "body": body, "labels": ["incident", "auto-triage"]}
    out_dir.mkdir(parents=True, exist_ok=True)
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


def issue_body(report: dict, before: dict, after: dict, meta: dict, config: dict,
               review: dict | None = None, review_meta: dict | None = None) -> str:
    fix = report["fix"]
    review_md = ""
    if review and review_meta and "error" not in review_meta:
        review_md = (f"\n**Independent review** (`{review_meta['model']}` on Lambda Inference, "
                     f"{review_meta['latency_s']} s): **{review.get('verdict', '?').upper()}** "
                     f"at confidence {review.get('confidence', '?')}. {review.get('reasoning', '')} "
                     f"Residual risk: {review.get('residual_risk', '')}\n")
    rows = [
        "| | before | after fix |", "|---|---:|---:|",
        f"| {fix['tunable']} | {config.get(fix['tunable'])} | {fix['value']} |",
        f"| outcome | {outcome(before)} | {outcome(after)} |",
        f"| steps | {before['steps']} | {after['steps']} |",
        f"| ascent | {before['ascent_m']} m | {after['ascent_m']} m |",
        f"| upright score | {before['upright_score']} | {after['upright_score']} |",
    ]
    ev = "\n".join(f"- {e}" for e in report.get("evidence", []))
    return f"""**Root cause:** {report['root_cause']}

**Evidence**
{ev}

**Fix applied:** `{fix['tunable']} = {fix['value']}` (was {config.get(fix['tunable'])})

{chr(10).join(rows)}
{review_md}
_Diagnosed by `{meta['model']}` via Respan in {meta['latency_s']} s ({meta['prompt_tokens']} in / {meta['completion_tokens']} out). Seed {before['seed']}, impulse {config['slip_impulse_n']} N._
"""


# ------------------------------------------------------------------ main ---

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/ppo_fixed_line_slope/g1_fixed_line_final.zip")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="assist",
                   help="which misconfiguration to start from")
    p.add_argument("--assist", type=float, default=None,
                   help="override the assist scale (implies --scenario assist)")
    p.add_argument("--impulse", type=float, default=700.0)
    p.add_argument("--seed", type=int, default=4100)
    p.add_argument("--llm", default=os.environ.get("RESPAN_MODEL", DEFAULT_MODEL))
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPO", "abhijitbetigeri/robot-incident-analyst"))
    p.add_argument("--live", action="store_true", help="also send the fix to a running demo_live.py viewer")
    p.add_argument("--no-issue", action="store_true", help="print the issue payload instead of filing it")
    p.add_argument("--out", default="runs")
    a = p.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = pathlib.Path(a.out) / f"incident_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = PPO.load(a.model, device="cpu")
    if a.assist is not None:
        a.scenario = "assist"
        SCENARIOS["assist"]["overrides"]["balance_assist_scale"] = a.assist
    env = make_env(a.scenario, a.impulse)
    config = config_of(env, a.impulse, a.seed, a.model)
    bad = SCENARIOS[a.scenario]["overrides"]

    # 1. run with the bad config
    print(f"\n[1/5] Running episode  seed={a.seed}  scenario={a.scenario}  "
          + "  ".join(f"{k}={v}" for k, v in bad.items()), flush=True)
    t0 = time.time()
    before, log = rollout_with_log(env, policy, a.seed)
    (out_dir / "run_before.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n")
    print(f"      {outcome(before)}  ascent {before['ascent_m']} m  "
          f"upright {before['upright_score']}  ({time.time() - t0:.1f} s, {len(log)} log rows)", flush=True)
    if before["success"]:
        print("      Episode succeeded with this config, nothing to diagnose. Try another --scenario.")
        return

    # 2. diagnose through Respan
    print(f"\n[2/5] Diagnosing with {a.llm} via Respan gateway", flush=True)
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
    value = apply_fix(env, fix)
    if a.live and fix.get("tunable") == "balance_assist_scale":
        import sim_bridge
        sim_bridge.send("assist", scale=value)
        print(f"      sent assist={value} to the live viewer", flush=True)
    print(f"\n[3/5] Re-running seed={a.seed} with {fix['tunable']}={value}", flush=True)
    t0 = time.time()
    after, log_after = rollout_with_log(env, policy, a.seed)
    (out_dir / "run_after.jsonl").write_text("\n".join(json.dumps(r) for r in log_after) + "\n")
    print(f"      {outcome(after)}  ascent {after['ascent_m']} m  "
          f"upright {after['upright_score']}  ({time.time() - t0:.1f} s)", flush=True)

    # 4. independent review on Lambda Inference
    print("\n[4/5] Independent review on Lambda GPU", flush=True)
    review, review_meta = review_on_lambda(report, before, after, log, log_after)
    if review and review_meta and "error" not in review_meta:
        print(f"      {review_meta['model']}  {review_meta['latency_s']} s")
        print(f"      verdict    : {str(review.get('verdict', '?')).upper()}  "
              f"(confidence {review.get('confidence', '?')})")
        for line in textwrap.wrap(str(review.get("reasoning", "")), 70):
            print(f"                   {line}")
    elif review_meta:
        print(f"      skipped: {review_meta.get('error')}", flush=True)
    else:
        print("      skipped (set LAMBDA_HOST in .env.local and run ./lambda_reviewer.sh tunnel)", flush=True)
    (out_dir / "review.json").write_text(json.dumps({"review": review, "lambda": review_meta}, indent=2) + "\n")

    # 5. file it
    print(f"\n[5/5] Filing issue on {a.repo} via Nango", flush=True)
    body = issue_body(report, before, after, meta, config, review, review_meta)
    if a.no_issue:
        (out_dir / "issue.json").write_text(json.dumps({"title": report.get("issue_title"), "body": body}, indent=2) + "\n")
        where = f"not filed (--no-issue); payload in {out_dir / 'issue.json'}"
    else:
        where = file_issue(a.repo, report.get("issue_title", "G1 incident"), body, out_dir)
    print(f"      {where}", flush=True)

    summary = {"config": config, "before": before, "after": after, "report": report,
               "gateway": meta, "review": review, "lambda": review_meta,
               "issue": where, "fixed": bool(after["success"])}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 64)
    print(f"  before   {fix['tunable']}={config.get(fix['tunable'])}  {outcome(before)}")
    print(f"  after    {fix['tunable']}={value}  {outcome(after)}")
    print(f"  {'FIXED' if after['success'] else 'NOT FIXED'}   artifacts in {out_dir}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
