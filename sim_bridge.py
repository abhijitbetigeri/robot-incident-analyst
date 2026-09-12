"""File-based bridge between a voice agent and the running simulation.

Deliberately a JSON file and not a socket or HTTP server. Under a deadline the
failure modes matter more than the elegance: a file has no port to collide, no
server to crash, no async to deadlock, and it can be driven from a shell with
`echo` before any audio exists. If the voice layer dies mid-demo the sim keeps
running.

The simulation loop owns the physics and must never block on speech: MuJoCo
steps at 50 Hz while a voice round trip is 300 ms-2 s. So the agent WRITES
intents here and the sim DRAINS them once per control step, latest-wins.
Telemetry goes the other way through a second file the agent reads on demand.

    from sim_bridge import CommandBridge
    bridge = CommandBridge()
    cmd = bridge.take()          # None, or a dict, once per sim step
    bridge.publish(metrics)      # after each step
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import time

DIR = pathlib.Path(tempfile.gettempdir())
CMD = DIR / "g1_cmd.json"
STATUS = DIR / "g1_status.json"


class CommandBridge:
    def __init__(self) -> None:
        self._seen = 0.0
        CMD.unlink(missing_ok=True)

    def take(self) -> dict | None:
        """Return a pending command once, or None. Never raises."""
        try:
            if not CMD.exists():
                return None
            stamp = CMD.stat().st_mtime
            if stamp <= self._seen:
                return None
            self._seen = stamp
            return json.loads(CMD.read_text())
        except Exception:
            # A half-written file is normal: the writer is another process.
            # Silently skip; the next poll picks it up.
            return None

    def publish(self, metrics: dict) -> None:
        try:
            payload = {
                "ascent_m": round(float(metrics.get("ascent", 0.0)), 3),
                "slip_depth_m": round(float(metrics.get("slip_depth_m", 0.0)), 3),
                "recovered": bool(metrics.get("recovered", 0.0)),
                "phase": metrics.get("phase", "walk"),
                "pack_kg": metrics.get("pack_kg", 0.0),
                "assist_scale": round(float(metrics.get("balance_assist_scale", 1.0)), 2),
                "t": time.time(),
            }
            # Write-then-rename so a reader never sees a partial file.
            tmp = STATUS.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(STATUS)
        except Exception:
            pass


def send(action: str, **kwargs) -> dict:
    """Used by the voice agent (and by hand, for testing)."""
    payload = {"action": action, **kwargs}
    tmp = CMD.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(CMD)
    return payload


def status() -> dict:
    try:
        return json.loads(STATUS.read_text())
    except Exception:
        return {"phase": "unknown"}


if __name__ == "__main__":
    import sys
    print(send(sys.argv[1], **dict(a.split("=") for a in sys.argv[2:]))
          if len(sys.argv) > 1 else status())
