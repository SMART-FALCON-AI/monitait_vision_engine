"""Ejection procedures routes."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from config import load_data_file, save_data_file
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/procedures")
async def get_procedures(request: Request):
    """Get current ejection procedures."""
    config = getattr(request.app.state, 'timeline_config', {})
    return {'procedures': config.get('procedures', [])}


@router.post("/api/procedures")
async def update_procedures(request: Request):
    """Update ejection procedures configuration."""
    try:
        data = await request.json()
        procedures = data.get('procedures', [])

        # Update in app.state
        config = getattr(request.app.state, 'timeline_config', {})
        config['procedures'] = procedures
        request.app.state.timeline_config = config

        # Persist to DATA_FILE
        file_data = load_data_file()
        file_data['timeline_config'] = request.app.state.timeline_config
        save_data_file(file_data)

        logger.info(f"Procedures updated: {len(procedures)} procedure(s)")
        return {'success': True, 'procedures': procedures}
    except Exception as e:
        logger.error(f"Error updating procedures: {e}")
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})


@router.post("/api/color-reference/{class_name}")
async def set_color_reference(class_name: str, request: Request):
    """Set a fixed L*a*b* color reference for a class.

    Body: {"capture": true} to use latest detection, or {"lab": [L, a, b]}.
    """
    try:
        from services.detection import get_color_reference, set_fixed_color_reference
        data = await request.json()

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

        success = set_fixed_color_reference(class_name, lab)
        if success:
            return {'success': True, 'class': class_name, 'lab': lab}
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Redis write failed'})
    except Exception as e:
        logger.error(f"Error setting color reference: {e}")
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})


@router.get("/api/color-reference/{class_name}")
async def get_color_ref_endpoint(class_name: str, mode: str = "fixed"):
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
