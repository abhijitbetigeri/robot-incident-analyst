# Nobody reads the robot logs. So we built something that does.

*How we turned a failed humanoid climb into a root cause, a fix, a verified re-run, and a ticket, in thirteen seconds, at the Lambda / Nango / Respan / Gemma hackathon.*

![Architecture](submission/architecture.png)

## The problem we kept running into

If you train or evaluate robot policies, you know the shape of the day. You launch a few hundred simulation episodes, come back, and a good chunk of them failed. Each failure left a log: thousands of lines of joint states, contact flags, torques, and a step number where it all ended.

Almost none of those logs ever get read. You open two or three, scroll to the crash, form a theory, change something, and re-run. The rest of the failures are treated as noise. In our own work on a Unitree G1 humanoid climbing a fixed rope on an ice slope, we'd lost real time to exactly this: the policy was falling around step 250 and it took a while to notice that short episodes were censoring the survival numbers. A log reader would have flagged it the first time.

So the question for the hackathon was simple. Can an open model read a failed run, name the cause, and prove the fix, without a human in the loop?

## What we built

One command, `incident_loop.py`, runs a closed loop:

1. **Run fails.** The G1 climbs a 35-degree slope on a fixed line in MuJoCo with one setting deliberately misconfigured. It falls. Every step's telemetry is logged.
2. **Diagnose.** The telemetry, the current configuration, and a list of tunables go to Gemma 4 31B through the Respan gateway. The model returns a root cause, evidence citing specific metrics and steps, and a single fix as JSON.
3. **Fix and re-run.** The fix is applied to the live environment and the same seed is rolled out again. The robot has to reach the 1.5 m target for the fix to count.
4. **Independent review.** A second, different open model, Gemma 3 12B served on a Lambda GH200, is shown the diagnosis as a claim plus the raw before and after telemetry. It has to say whether the re-run actually supports the claim.
5. **File the ticket.** Nango creates a GitHub issue with the root cause, the evidence, a before/after table, and the reviewer's verdict.

The whole loop takes about thirteen seconds. The analyst is never told which setting is wrong.

![The failing run with live telemetry](submission/failing-run.jpg)

## Three faults, three correct diagnoses

We injected three different misconfigurations. Each is a real failure with a real fix, and the model had to find the right tunable among four candidates from the numbers alone.

| Injected fault | What the robot did | What Gemma found |
|---|---|---|
| Balance assist at 0.5 | pelvis height decayed from 0.64 m to 0.42 m, fall at step 126 | `balance_assist_scale = 1.0` |
| Boot traction off | walked in place, ascent never left zero, fall at step 495 | `boot_traction_enabled = true` |
| Fixed line disconnected | slid 0.56 m backward, fall at step 65 | `fixed_line_enabled = true` |

All three re-runs reached the summit at 1.53 m. The reviewer on Lambda confirmed all three diagnoses at 0.95 confidence, citing the pelvis height and ascent numbers from both runs.

![The reviewer's verdict card](submission/review-card.jpg)

The evidence is on this repo. Issues [#14](https://github.com/abhijitbetigeri/robot-incident-analyst/issues/14), [#15](https://github.com/abhijitbetigeri/robot-incident-analyst/issues/15), and [#16](https://github.com/abhijitbetigeri/robot-incident-analyst/issues/16) were filed by the agent during the final video render, one per fault, each with the review paragraph in the body. The [2.5 minute demo video](submission/demo.mp4) shows the fall and the recovery on the same seed for each scenario.

## How each piece of the stack earned its place

**Respan** is the gateway every model call goes through. That gave us one endpoint and one key for Gemma across three hosting providers, automatic fallback when one was slow, and a full trace of every diagnosis with tokens, cost, and latency. A diagnosis costs about three hundredths of a cent. Those traces are not just observability. They are a labeled dataset of telemetry paired with expert reports, which matters for what's next.

**Gemma** does the reasoning. Gemma 4 31B is the analyst. It read structured telemetry it had never seen, reasoned about a physical failure mode, and produced valid JSON with a correct fix on every attempt we made. Gemma 3 12B is the reviewer, deliberately a different model so the check is independent.

**Lambda** serves the reviewer. We ran Ollama on a single GH200, reached it over an SSH tunnel, and the review came back in two to seven seconds. The bigger job for that GPU is the next step below.

**Nango** files the ticket. The agent calls one proxy endpoint with the issue payload and Nango handles the GitHub authentication. We never touched OAuth.

## Things that went wrong, honestly

**Lambda's hosted inference API no longer exists.** We'd planned to call it directly. The docs page redirects away and the hostnames don't resolve. Standing up Ollama on a GPU instance took about ten minutes and turned out to be a better story anyway: a model on hardware we control, checking a model we don't.

**The SSH tunnel died mid-render.** The video renderer runs for thirteen minutes per pass. The laptop slept, the tunnel dropped, and two of three scenarios silently skipped the review. The loop now re-opens the tunnel by itself when the reviewer is unreachable, and renders run under `caffeinate`.

**We pasted the wrong credentials three times.** Environment ID instead of secret key, account key instead of environment key, an SSH public key where an API key should go. The fix was a `.env.local` with comments saying exactly which dashboard page each value comes from. Boring, and it would have saved twenty minutes.

**The first render crashed at the last second.** The issue filer wrote a JSON copy into a folder that didn't exist yet. One `mkdir`. Thirteen minutes lost.

## What's next

The Respan traces from every run are already a training set: telemetry in, expert report out. The next step is to distill the 31B analyst into Gemma 4 E4B with a LoRA fine-tune on Lambda, serve the student with vLLM, and use Respan's evals to gate the swap, promoting the student only when it matches the teacher on held-out incidents. At that point the analyst is small enough to leave running on every failed episode, not just the ones someone remembers to look at.

The other thing we want is more faults. Three tunables is a demo. A real training pipeline has dozens of knobs, and the interesting failures are the ones where two of them interact.

## Try it

```bash
git clone https://github.com/abhijitbetigeri/robot-incident-analyst
cd robot-incident-analyst
python -m venv .venv && .venv/bin/pip install -r requirements.txt
npx @respan/cli setup
.venv/bin/python incident_loop.py --scenario traction --no-issue
```

Without Nango or Lambda credentials the loop still runs end to end and prints the ticket instead of filing it. The README covers the rest.

*Built in one day on a Unitree G1 MuJoCo environment. Every issue on this repo was filed by the agent.*
