"""Stable hardware fingerprint for this machine (4.0.382).

One value that identifies the PHYSICAL box, needs no operator input, and survives
an OS reinstall, a container recreate, and a `docker compose down -v`.

Why this exists: anything derived from software identity is wrong here.
  - `/etc/machine-id` inside the container is the CONTAINER's, regenerated on every
    `docker compose up` — it identifies nothing durable.
  - `/etc/machine-id` on the host is copied verbatim by a disk clone, so every clone
    of the golden image would claim the same identity.
  - A hand-typed site name is a manual step that gets skipped or duplicated.

SMBIOS/DMI is read from the motherboard, so a disk cloned onto different hardware
reports a DIFFERENT fingerprint. That is precisely the property the support-credential
clone guard needs: a cloned golden image cannot silently keep posting into the room
belonging to the machine it was cloned from.

Readable from inside the MVE container because it runs privileged with /sys mounted
(verified on vteam12 2026-09-01: product_uuid + board_serial both present).
"""
import hashlib
import logging
import os
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# DMI fields in priority order. product_uuid is per-motherboard and the most reliable;
# board_serial is a good second. product_serial / chassis_serial / asset_tag are NOT
# used as primaries — on the boxes we ship they are OEM placeholders (vteam12 returns
# "System Serial Number" and "Default string"), which would collide across the fleet.
_DMI_FIELDS = ("product_uuid", "board_serial")
_DMI_DIR = "/sys/class/dmi/id"

# Values OEMs ship as filler. Any of these is treated as "no value", never as identity.
_PLACEHOLDERS = {
    "", "none", "default string", "to be filled by o.e.m.",
    "to be filled by oem", "system serial number", "not specified",
    "not applicable", "unknown", "0", "invalid",
    "00000000-0000-0000-0000-000000000000",
    "ffffffff-ffff-ffff-ffff-ffffffffffff",
}

_CACHE: Dict[str, Optional[str]] = {}
_LOCK = threading.Lock()


def _usable(value: Optional[str]) -> bool:
    if not value:
        return False
    v = value.strip().lower()
    if v in _PLACEHOLDERS:
        return False
    # a value that is only punctuation/zeros carries no identity
    return len(v) >= 6 and any(c.isalnum() and c != "0" for c in v)


def _read_dmi(field: str) -> Optional[str]:
    try:
        with open(os.path.join(_DMI_DIR, field), "r") as fh:
            return fh.read().strip()
    except Exception:
        return None


def _mac_addresses() -> List[str]:
    """Sorted permanent-looking MAC addresses, as a last-resort fallback."""
    out = []
    try:
        base = "/sys/class/net"
        for iface in sorted(os.listdir(base)):
            if iface == "lo" or iface.startswith(("veth", "docker", "br-", "virbr")):
                continue
            try:
                with open(os.path.join(base, iface, "address"), "r") as fh:
                    mac = fh.read().strip().lower()
            except Exception:
                continue
            if mac and mac != "00:00:00:00:00:00":
                out.append(mac)
    except Exception:
        pass
    return sorted(out)


def fingerprint_sources() -> Dict[str, Optional[str]]:
    """What the fingerprint was built from — for diagnostics and support tickets."""
    src: Dict[str, Optional[str]] = {}
    for field in _DMI_FIELDS:
        raw = _read_dmi(field)
        src[field] = raw if _usable(raw) else None
    macs = _mac_addresses()
    src["macs"] = ",".join(macs) if macs else None
    return src


def machine_fingerprint(length: int = 12) -> str:
    """Stable lowercase hex id for this physical machine.

    Same machine → same value, forever. Different machine → different value, even
    from a byte-identical disk image. Cached: the underlying files never change
    while the process runs.
    """
    key = "fp{}".format(length)
    with _LOCK:
        cached = _CACHE.get(key)
    if cached:
        return cached

    parts: List[str] = []
    for field in _DMI_FIELDS:
        raw = _read_dmi(field)
        if _usable(raw):
            parts.append("{}={}".format(field, raw.strip()))

    if not parts:
        # No usable DMI (a VM, or a locked-down kernel). MACs are weaker — a NIC swap
        # changes them — but they still beat having no identity at all.
        macs = _mac_addresses()
        if macs:
            parts.append("macs=" + ",".join(macs))
            logger.warning("[machine-id] no usable DMI fields; falling back to MAC addresses")

    if not parts:
        logger.error("[machine-id] no hardware identity available on this host")
        return ""

    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:length]
    with _LOCK:
        _CACHE[key] = digest
    logger.info("[machine-id] fingerprint %s from %s", digest,
                ", ".join(p.split("=", 1)[0] for p in parts))
    return digest
