"""Command catch-all route. MUST be registered last."""

from fastapi import APIRouter, Request, HTTPException
from typing import Optional
from datetime import datetime
from config import USER_COMMANDS, COMMANDS_WITH_VALUE
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/{command}")
async def send_command(command: str, request: Request, value: Optional[int] = None):
    """Send command to watcher device.

    Examples:
        POST /U_ON_B_OFF - Turn U light on, B light off
        POST /SET_VERBOSE?value=1 - Enable verbose mode
        POST /OK_OFFSET_DELAY?value=100 - Set OK offset delay to 100ms
    """
    watcher = request.app.state.watcher_instance
    if watcher is None:
        raise HTTPException(status_code=503, detail="Watcher device not initialized")

    try:
        # Convert user-friendly command to actual command
        if command in USER_COMMANDS:
            cmd = USER_COMMANDS[command]
            # Add value for commands that require it
            if command in COMMANDS_WITH_VALUE:
                if value is None:
                    raise HTTPException(status_code=400, detail=f"Command '{command}' requires a value parameter")
                cmd = f"{cmd},{value}"
        else:
            cmd = command

        watcher._send_message(f"{cmd}\n")
        return {"status": "ok", "command": cmd, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send command: {str(e)}")
