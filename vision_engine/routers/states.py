from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from config import save_service_config, load_service_config, DATA_FILE
from services.state_machine import State, CapturePhase
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/states")
async def get_states(request: Request):
    """Get all configured states and current state status."""
    sm = request.app.state.state_manager
    if sm is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)
    return JSONResponse(content=sm.get_status())


@router.get("/api/states/{state_name}")
async def get_state(request: Request, state_name: str):
    """Get a specific state configuration."""
    sm = request.app.state.state_manager
    if sm is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)

    if state_name not in sm.states:
        return JSONResponse(content={"error": f"State '{state_name}' not found"}, status_code=404)

    return JSONResponse(content=sm.states[state_name].to_dict())


@router.post("/api/states")
async def create_or_update_state(request: Request):
    """Create or update a state configuration.

    Request body example:
    {
        "name": "uplight_capture",
        "active_cameras": [1, 2],
        "delay": 0.1,
        "light_mode": "U_ON_B_OFF",
        "encoder_threshold": 100,
        "analog_threshold": -1,
        "light_status_check": false
    }
    """
    sm = request.app.state.state_manager
    if sm is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)

    try:
        data = await request.json()

        if "name" not in data:
            return JSONResponse(content={"error": "State name is required"}, status_code=400)

        state = State.from_dict(data)
        success = sm.add_state(state)

        if success:
            # Auto-save states to DATA_FILE
            try:
                config = load_service_config() or {}
                config["states"] = {name: s.to_dict() for name, s in sm.states.items()}
                config["current_state_name"] = sm.current_state.name if sm.current_state else "default"
                save_service_config(config)
                logger.info(f"State '{state.name}' created/updated and saved to DATA_FILE")
            except Exception as save_error:
                logger.error(f"Failed to auto-save states: {save_error}")

            return JSONResponse(content={
                "success": True,
                "state": state.to_dict(),
                "message": f"State '{state.name}' created/updated"
            })
        else:
            return JSONResponse(content={"error": "Failed to add state"}, status_code=500)

    except Exception as e:
        logger.error(f"Error creating state: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.delete("/api/states/{state_name}")
async def delete_state(request: Request, state_name: str):
    """Delete a state configuration."""
    sm = request.app.state.state_manager
    if sm is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)

    if state_name == "default":
        return JSONResponse(content={"error": "Cannot delete default state"}, status_code=400)

    success = sm.remove_state(state_name)
    if success:
        # Auto-save states to DATA_FILE after deletion
        try:
            config = load_service_config() or {}
            config["states"] = {name: s.to_dict() for name, s in sm.states.items()}
            config["current_state_name"] = sm.current_state.name if sm.current_state else "default"
            save_service_config(config)
            logger.info(f"State '{state_name}' deleted and changes saved to DATA_FILE")
        except Exception as save_error:
            logger.error(f"Failed to auto-save states after deletion: {save_error}")

        return JSONResponse(content={"success": True, "message": f"State '{state_name}' deleted"})
    else:
        return JSONResponse(content={"error": f"State '{state_name}' not found"}, status_code=404)


@router.post("/api/states/{state_name}/activate")
async def activate_state(request: Request, state_name: str):
    """Activate a specific state as the current state."""
    sm = request.app.state.state_manager
    if sm is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)

    success = sm.set_current_state(state_name)
    if success:
        # Auto-save states to DATA_FILE after activation (to save current_state_name)
        try:
            config = load_service_config() or {}
            config["states"] = {name: s.to_dict() for name, s in sm.states.items()}
            config["current_state_name"] = sm.current_state.name if sm.current_state else "default"
            save_service_config(config)
            logger.info(f"State '{state_name}' activated and saved to DATA_FILE")
        except Exception as save_error:
            logger.error(f"Failed to auto-save states after activation: {save_error}")

        return JSONResponse(content={
            "success": True,
            "message": f"State '{state_name}' activated",
            "current_state": sm.current_state.to_dict() if sm.current_state else None
        })
    else:
        return JSONResponse(content={"error": f"Failed to activate state '{state_name}'"}, status_code=400)


@router.post("/api/states/trigger-capture")
async def trigger_capture(request: Request):
    """Manually trigger a capture cycle."""
    sm = request.app.state.state_manager
    if sm is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)

    success = sm.trigger_capture()
    if success:
        return JSONResponse(content={
            "success": True,
            "message": "Capture triggered",
            "capture_count": sm.capture_count
        })
    else:
        return JSONResponse(content={
            "error": "Failed to trigger capture (capture already in progress?)",
            "capture_state": sm.capture_state.value
        }, status_code=400)


@router.post("/api/states/save")
async def save_states(request: Request):
    """Save all states to the main config file (.env.prepared)."""
    sm = request.app.state.state_manager
    if sm is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)

    try:
        # Load existing config and update states section
        config = load_service_config() or {}
        config["states"] = {name: s.to_dict() for name, s in sm.states.items()}
        config["current_state_name"] = sm.current_state.name if sm.current_state else "default"

        if save_service_config(config):
            return JSONResponse(content={
                "success": True,
                "message": f"States saved to {DATA_FILE}",
                "states_saved": len(config["states"])
            })
        else:
            return JSONResponse(content={"error": "Failed to save states"}, status_code=500)
    except Exception as e:
        logger.error(f"Error saving states: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/states/load")
async def load_states(request: Request):
    """Load states from the main config file (.env.prepared)."""
    sm = request.app.state.state_manager
    if sm is None:
        return JSONResponse(content={"error": "State manager not initialized"}, status_code=503)

    try:
        config = load_service_config()
        if not config or "states" not in config:
            return JSONResponse(content={"error": "No saved states found in config"}, status_code=404)

        states_loaded = 0
        for state_name, state_data in config["states"].items():
            state = State.from_dict(state_data)
            sm.add_state(state)
            states_loaded += 1

        # Set current state if specified
        current_state_name = config.get("current_state_name", "default")
        if current_state_name in sm.states:
            sm.set_current_state(current_state_name)

        return JSONResponse(content={
            "success": True,
            "message": f"States loaded from {DATA_FILE}",
            "states_count": states_loaded
        })
    except Exception as e:
        logger.error(f"Error loading states: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
