"""Ejection procedures routes."""

from fastapi import APIRouter, Request
from fastapi import Depends as Depends
from routers._body import json_body as _json_body_dep, body as _req_body  # 4.0.425
from fastapi.responses import JSONResponse
from config import load_data_file, save_data_file, save_config_key
from config import timeline_config_locked as _tl_locked   # 4.0.443
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/procedures")
def get_procedures(request: Request):
    """Get current ejection procedures."""
    config = getattr(request.app.state, 'timeline_config', {})
    return {'procedures': config.get('procedures', [])}


def _num(v, kind, field, where):
    """Coerce a numeric rule field; ValueError with an operator-readable message."""
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: '{field}' must be a number (got {v!r})")
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"{where}: '{field}' must be a finite number")
    return int(x) if kind == "int" else x


_RULE_NUMS = {"count": "int", "area": "float", "threshold": "float", "min_confidence": "float",
              "marginal_confidence": "float", "max_delta_e": "float"}


def _clean_procedures(procs):
    """4.0.438 — validate + normalise before saving. Nothing was checked: a rule with
    {"count": "2"} made the evaluator raise TypeError on EVERY frame, which the inference
    path catches per frame — so ALL procedures silently stopped ejecting while this API had
    answered 200. A missing `enabled` was read as True in one place and False in another."""
    if not isinstance(procs, list):
        raise ValueError("'procedures' must be a list")
    out = []
    for i, p in enumerate(procs):
        where = f"procedure {i + 1}"
        if not isinstance(p, dict):
            raise ValueError(f"{where}: must be an object")
        p = dict(p)
        if p.get("name"):
            where = f"procedure '{p.get('name')}'"
        p["enabled"] = bool(p.get("enabled", False))
        cams = p.get("cameras", [])
        if cams in (None, ""):
            cams = []
        if not isinstance(cams, list):
            raise ValueError(f"{where}: 'cameras' must be a list")
        try:
            p["cameras"] = [int(c) for c in cams if str(c).strip() != ""]
        except (TypeError, ValueError):
            raise ValueError(f"{where}: 'cameras' must be camera numbers")
        rules = p.get("rules", [])
        if rules in (None, ""):
            rules = []
        if not isinstance(rules, list):
            raise ValueError(f"{where}: 'rules' must be a list")
        clean_rules = []
        for j, r in enumerate(rules):
            rw = f"{where}, rule {j + 1}"
            if not isinstance(r, dict):
                raise ValueError(f"{rw}: must be an object")
            r = dict(r)
            for f, kind in _RULE_NUMS.items():
                if f in r:
                    v = _num(r.get(f), kind, f, rw)
                    if v is None:
                        r.pop(f, None)
                    else:
                        r[f] = v
            clean_rules.append(r)
        p["rules"] = clean_rules
        out.append(p)
    return out


@router.post("/api/procedures")
@_tl_locked   # 4.0.443
def update_procedures(request: Request, _mve_body=Depends(_json_body_dep)):
    """Update ejection procedures configuration."""
    try:
        data = _req_body(_mve_body)
        # 4.0.438 — a body without 'procedures' (or a typo) used to save [] = every rule gone
        if not isinstance(data, dict) or "procedures" not in data:
            return JSONResponse(status_code=400, content={'success': False, 'error': "body must be {\"procedures\": [...]}"})
        try:
            procedures = _clean_procedures(data.get('procedures'))
        except ValueError as ve:
            return JSONResponse(status_code=400, content={'success': False, 'error': str(ve)})

        config = getattr(request.app.state, 'timeline_config', {})
        # 4.0.438 — booted on DEFAULT timeline_config (DB not ready at boot): saving defaults +
        # these procedures replaced the whole stored timeline_config (camera order, num_rows,
        # mark-ejects, bbox padding, …) and stopped the self-heal. Start from the STORED one
        # when there is one (that also recovers those settings); a fresh install has none.
        if getattr(request.app.state, 'timeline_config_is_default', False):
            try:
                from config import invalidate_config_cache, load_data_file as _ldf
                invalidate_config_cache()
                stored = (_ldf() or {}).get('timeline_config')
                if isinstance(stored, dict) and stored:
                    config = dict(stored)
                    request.app.state.timeline_config_is_default = False
                    logger.warning("[BOOT-RACE] procedures save recovered the stored timeline_config first")
            except Exception as _le:
                logger.warning(f"procedures: stored timeline_config read failed: {_le}")
        config['procedures'] = procedures
        request.app.state.timeline_config = config

        # 4.0.355 — persist ONLY timeline_config so a concurrent service_config
        # save can't wipe these ejection procedures (lost-update race fix).
        ok = save_config_key('timeline_config', request.app.state.timeline_config)
        if ok is False:
            # 4.0.438 — neither DB nor disk took it: say so instead of "success"
            return JSONResponse(status_code=500, content={'success': False, 'error': 'procedures are active now but could NOT be saved (database and disk write failed) - they will be lost on restart'})

        logger.info(f"Procedures updated: {len(procedures)} procedure(s)")
        return {'success': True, 'procedures': procedures}
    except Exception as e:
        logger.error(f"Error updating procedures: {e}")
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})


@router.post("/api/color-reference/{class_name}")
def set_color_reference(class_name: str, request: Request, _mve_body=Depends(_json_body_dep)):
    """Set a fixed L*a*b* color reference for a class.

    Body: {"capture": true} to use latest detection, or {"lab": [L, a, b]}.
    """
    try:
        from services.detection import get_color_reference, set_fixed_color_reference
        data = _req_body(_mve_body)

        if data.get('capture', False):
            lab = get_color_reference(class_name, "previous")
            if lab is None:
                return JSONResponse(
                    status_code=404,
                    content={'success': False, 'error': f'No color data for "{class_name}" yet'}
                )
        else:
            lab = data.get('lab')
            if not lab or not isinstance(lab, list) or len(lab) != 3:
                return JSONResponse(
                    status_code=400,
                    content={'success': False, 'error': 'Expected {"lab": [L, a, b]}'}
                )
            # 4.0.438 — numbers only: ["40","1","2"] passed and then broke the ΔE maths per frame
            try:
                lab = [float(x) for x in lab]
                if any(x != x or x in (float("inf"), float("-inf")) for x in lab):
                    raise ValueError
            except (TypeError, ValueError):
                return JSONResponse(status_code=400, content={'success': False, 'error': 'L, a, b must be numbers'})

        success = set_fixed_color_reference(class_name, lab)
        if success:
            return {'success': True, 'class': class_name, 'lab': lab}
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Redis write failed'})
    except Exception as e:
        logger.error(f"Error setting color reference: {e}")
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})


@router.get("/api/color-reference/{class_name}")
def get_color_ref_endpoint(class_name: str, mode: str = "fixed"):
    """Get current color reference. Query: ?mode=fixed|previous|running_avg"""
    try:
        from services.detection import get_color_reference
        lab = get_color_reference(class_name, mode)
        if lab is None:
            return JSONResponse(
                status_code=404,
                content={'success': False, 'error': f'No {mode} reference for "{class_name}"'}
            )
        return {'success': True, 'class': class_name, 'mode': mode, 'lab': lab}
    except Exception as e:
        logger.error(f"Error getting color reference: {e}")
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})
