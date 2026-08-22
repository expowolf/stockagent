# notify.py
"""Phone push notification via ntfy.sh (free, no account required).

Set these env vars:
  NTFY_TOPIC   - your private topic name, e.g. "kronos-alerts-abc123"
  NTFY_SERVER  - optional, defaults to https://ntfy.sh
"""

import os
import httpx

NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")


def push(title: str, message: str, priority: str = "default", tags: list[str] | None = None) -> bool:
    """
    Send a push notification to your phone via ntfy.sh.
    Returns True on success, False if NTFY_TOPIC is unset or the request fails.
    """
    if not NTFY_TOPIC:
        print("[notify] NTFY_TOPIC not set — skipping push")
        return False

    try:
        headers = {
            "Title": title[:250],
            "Priority": priority,  # min / low / default / high / urgent
        }
        if tags:
            headers["Tags"] = ",".join(tags)

        resp = httpx.post(
            f"{NTFY_SERVER}/{NTFY_TOPIC}",
            content=message.encode(),
            headers=headers,
            timeout=8,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[notify] push failed: {exc}")
        return False
