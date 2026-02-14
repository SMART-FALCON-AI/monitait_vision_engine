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
