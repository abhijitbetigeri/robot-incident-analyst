# Robotic Simulation Incident Analysis

A failed robot run gets a root cause, a fix, a re-run that proves the fix, and a
ticket. One command, about 13 seconds, no human reads the log.

Built for the Lambda / Nango / Respan / Gemma hackathon. The robot is a Unitree
G1 humanoid climbing a fixed line on a 35-degree slope in MuJoCo, driven by a
trained PPO policy. The environment, the policy checkpoint, and the analyst are
all in this repo.

- Demo video (2 min, three scenarios): [submission/demo.mp4](submission/demo.mp4)
- Slide: [submission/g1-incident-analyst.pptx](submission/g1-incident-analyst.pptx)
- Script: [incident_loop.py](incident_loop.py), recorder: [record_incident.py](record_incident.py)
- Example ticket the agent filed: [issue #1](https://github.com/abhijitbetigeri/robot-incident-analyst/issues/1)

## The loop

```
[1/4] Running episode  seed=4100  balance_assist_scale=0.5
      FAILURE at step 126  ascent 0.309 m  upright 0.841
[2/4] Diagnosing with openrouter/google/gemma-4-31b-it via Respan gateway
      root cause : The robot lacks sufficient stabilization to maintain pelvis
                   height on the slope, leading to a gradual sink until the
                   failure threshold is hit.
      fix        : balance_assist_scale = 1.0
      gateway    : 5.4 s  4571 in / 250 out
[3/4] Re-running seed=4100 with balance_assist_scale=1.0
      SUCCESS: reached the 1.5 m target  ascent 1.53 m  upright 0.977
[4/4] Filing issue on abhijitbetigeri/robot-incident-analyst via Nango
      https://github.com/abhijitbetigeri/robot-incident-analyst/issues/1
```

1. **Run fails.** The G1 climbs with the balance assist misconfigured. Pelvis
   height decays until the environment's fall rule fires. Every step's
   telemetry is logged.
2. **Diagnose.** The log, the config, and the list of tunables go to Gemma 4
   31B through the Respan gateway. Gemma returns a root cause, evidence citing
   metrics and steps, and a fix as JSON.
3. **Fix and re-run.** The fix is applied to the live environment and the same
   seed is rolled out again. The robot reaches the 1.5 m target.
4. **File the issue.** A GitHub issue is created through Nango with the
   evidence and a before/after table.

Each run leaves `runs/incident_<timestamp>/` with `run_before.jsonl`,
`run_after.jsonl`, `report.json`, `issue.json`, and `summary.json`.

## Scenarios

The analyst is never told which setting is wrong. Each scenario injects one
misconfiguration and the model has to find it in the telemetry. All three are
verified: the re-run with the proposed fix completes the ascent.

| `--scenario` | Injected fault | What happens | Fix Gemma found |
|---|---|---|---|
| `assist` | `balance_assist_scale = 0.5` | pelvis height decays, fall at step 126 | `balance_assist_scale = 1.0` |
| `traction` | `boot_traction_enabled = false` | no uphill progress, slides and falls at step 495 | `boot_traction_enabled = true` |
| `line` | `fixed_line_enabled = false` | not clipped in, slides 0.56 m back and falls at step 65 | `fixed_line_enabled = true` |

An independent reviewer model served on a Lambda GPU (see
`lambda_reviewer.sh`) checks each diagnosis against the before and after
telemetry and adds its verdict to the issue. Example with the review:
[issue #10](https://github.com/abhijitbetigeri/robot-incident-analyst/issues/10),
where Gemma 3 12B on a GH200 confirmed the traction diagnosis at 0.95 confidence.

```bash
./lambda_reviewer.sh setup    # install Ollama on the instance, pull gemma3:12b
./lambda_reviewer.sh tunnel   # forward localhost:11434 to it, keep open
```

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
npx @respan/cli setup            # stores the Respan key in ~/.respan
cp .env.local.example .env.local # then fill in the Nango values
.venv/bin/python incident_loop.py
```

Without Nango credentials the loop still completes and prints the issue
payload instead of filing it.

Options: `--scenario traction` or `--scenario line` picks a different fault,
`--seed` changes the episode, `--llm` swaps the model behind the gateway,
`--no-issue` prints the ticket instead of filing it, and `--live` pushes the
fix into a running `mjpython demo_live.py` viewer so the audience sees the
robot recover. `record_incident.py` renders all scenarios into
`submission/demo.mp4`.

## Files

| File | What it is |
|---|---|
| `incident_loop.py` | The analyst: run, diagnose, fix, re-run, review, file |
| `record_incident.py` | Renders the scenarios into the demo video |
| `lambda_reviewer.sh` | Sets up and tunnels the reviewer model on a Lambda GPU instance |
| `fixed_line_slope_env.py` | MuJoCo fixed-line ascent environment for the G1 |
| `slip_recovery_env.py` | Adds the induced slip and the tunable balance assist |
| `demo_live.py`, `sim_bridge.py` | Live MuJoCo viewer and the file bridge the analyst can push fixes through |
| `models/ppo_fixed_line_slope/` | Trained PPO climbing policy |
| `assets/` | Unitree G1 model and the slope scene |

## How each sponsor is used

| Sponsor | Role |
|---|---|
| Respan | Gateway for every model call. Traces of each diagnosis are the training set; evals gate the student model. |
| Gemma | Gemma 4 31B is the analyst today. Gemma 4 E4B is the student to distill into. |
| Lambda | Serves the independent reviewer (Gemma 3 12B via Ollama on a GH200) that verifies every diagnosis before it is filed. Next: fine-tune the E4B student on the Respan traces there. |
| Nango | GitHub and Slack tools with auth handled. The agent files the ticket. |

## Why this matters

Robot training and evaluation produce hundreds of failed episodes, each a log
of thousands of lines. Almost none are read. This turns every failure into a
triaged, reproducible ticket with a verified fix, so the team looks at root
causes instead of scrolling telemetry.
