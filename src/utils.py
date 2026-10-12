"""Small helpers shared across modules."""
import hashlib
import json
from datetime import datetime


def format_inr(value: float) -> str:
    """Format a number as Indian Rupees with Indian digit grouping (12,34,567)."""
    if value is None:
        return "-"
    negative = value < 0
    value = abs(round(float(value)))
    digits = str(int(value))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups) + "," + tail
    return ("-₹" if negative else "₹") + digits


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stable_hash(obj) -> str:
    """Deterministic short hash of any JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
