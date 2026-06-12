from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB, load_data_file, save_data_file,
    YOLO_INFERENCE_URL, GRADIO_MODEL, GRADIO_CONFIDENCE_THRESHOLD,
    CAPTURE_MODE, EJECTOR_ENABLED, HISTOGRAM_ENABLED,
    CHECK_CLASS_COUNTS_ENABLED, CHECK_CLASS_COUNTS_CLASSES,
    PARENT_OBJECT_LIST, STORE_ANNOTATION_ENABLED,
)
from services.db import db_connection_pool, get_db_connection, release_db_connection
from redis import Redis
from datetime import datetime
from typing import Dict, Any
import json
import time
import logging
import os
import psycopg2

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# AI MODEL CONFIGURATION HELPERS
# =============================================================================

def _get_ai_models_from_redis():
    """Load all AI models from Redis. Returns dict: {"models": {...}, "active": "name"}."""
    try:
        r = Redis("redis", 6379, db=REDIS_DB)
        raw = r.get("ai_models")
        if raw:
            data = json.loads(raw.decode('utf-8') if isinstance(raw, bytes) else raw)
            if "models" in data:
                return data
        # Migration: check legacy single-model keys
        legacy_model = r.get("ai_model")
        legacy_key = r.get("ai_api_key")
        if legacy_model and legacy_key:
            provider = legacy_model.decode('utf-8') if isinstance(legacy_model, bytes) else legacy_model
            api_key = legacy_key.decode('utf-8') if isinstance(legacy_key, bytes) else legacy_key
            name = provider.capitalize()
            data = {"models": {name: {"provider": provider, "api_key": api_key}}, "active": name}
            r.set("ai_models", json.dumps(data))
            return data
    except Exception as e:
        logger.warning(f"Failed to load AI models from Redis: {e}")
    return {"models": {}, "active": None}


def _save_ai_models_to_redis(data):
    """Save all AI models to Redis."""
    try:
        r = Redis("redis", 6379, db=REDIS_DB)
        r.set("ai_models", json.dumps(data))
        # Also set legacy keys for backward compatibility with ai_query
        active_name = data.get("active")
        if active_name and active_name in data.get("models", {}):
            m = data["models"][active_name]
            r.set("ai_model", m["provider"])
            r.set("ai_api_key", m["api_key"])
        return True
    except Exception as e:
        logger.error(f"Failed to save AI models to Redis: {e}")
        return False


# =============================================================================
# AI CONFIGURATION ENDPOINTS (multi-model)
# =============================================================================

@router.get("/api/ai_config")
async def get_ai_config():
    """Get all configured AI models and active model."""
    try:
        data = _get_ai_models_from_redis()
        # Strip API keys for display (show only last 4 chars). Surface the
        # 3.21.23 base_url and model_id fields so the UI can pre-fill them.
        safe_models = {}
        for name, cfg in data.get("models", {}).items():
            key = cfg.get("api_key", "")
            masked = f"***{key[-4:]}" if len(key) > 4 else "***"
            safe_models[name] = {
                "provider": cfg.get("provider", ""),
                "api_key_masked": masked,
                "base_url": cfg.get("base_url", ""),
                "model_id": cfg.get("model_id", ""),
            }
        return JSONResponse(content={"models": safe_models, "active": data.get("active")})
    except Exception as e:
        logger.error(f"Error getting AI config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/ai_config")
async def save_ai_config(config: Dict[str, Any]):
    """Save/update an AI model configuration.

    3.21.23 — new optional fields:
      - base_url: override the SDK's default endpoint (lets you point at
                  any OpenAI-compatible relay, e.g. rucode.rdemos.com/v1,
                  ollama, etc).
      - model_id: override the hard-coded model name in the chat call
                  (e.g. "kimi-k2.6", "claude-3-5-sonnet-20241022", "llama3:8b").

    Both fields default to empty (sdk defaults preserved).
    """
    try:
        name = config.get("name", "").strip()
        provider = config.get("provider", "")
        api_key = config.get("api_key", "")
        base_url = (config.get("base_url") or "").strip()
        model_id = (config.get("model_id") or "").strip()

        if not name:
            return JSONResponse(content={"error": "Model name is required"}, status_code=400)
        if not provider:
            return JSONResponse(content={"error": "Provider is required"}, status_code=400)
        if not api_key:
            return JSONResponse(content={"error": "API key is required"}, status_code=400)

        data = _get_ai_models_from_redis()
        data["models"][name] = {
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,  # empty = sdk default
            "model_id": model_id,  # empty = sdk default
        }

        # Auto-activate if it's the first model
        if not data.get("active") or len(data["models"]) == 1:
            data["active"] = name

        if _save_ai_models_to_redis(data):
            logger.info(f"AI model saved: {name} ({provider})")
            return JSONResponse(content={"success": True, "message": f"Model '{name}' saved", "active": data["active"]})
        else:
            return JSONResponse(content={"error": "Failed to save to Redis"}, status_code=500)
    except Exception as e:
        logger.error(f"Error saving AI config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/ai_config/activate")
async def activate_ai_model(config: Dict[str, Any]):
    """Activate a specific AI model by name."""
    try:
        name = config.get("name", "").strip()
        if not name:
            return JSONResponse(content={"error": "Model name is required"}, status_code=400)

        data = _get_ai_models_from_redis()
        if name not in data.get("models", {}):
            return JSONResponse(content={"error": f"Model '{name}' not found"}, status_code=404)

        data["active"] = name
        if _save_ai_models_to_redis(data):
            logger.info(f"AI model activated: {name}")
            return JSONResponse(content={"success": True, "message": f"Model '{name}' activated"})
        else:
            return JSONResponse(content={"error": "Failed to save to Redis"}, status_code=500)
    except Exception as e:
        logger.error(f"Error activating AI model: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.delete("/api/ai_config/{model_name}")
async def delete_ai_model(model_name: str):
    """Delete an AI model configuration."""
    try:
        data = _get_ai_models_from_redis()
        if model_name not in data.get("models", {}):
            return JSONResponse(content={"error": f"Model '{model_name}' not found"}, status_code=404)

        del data["models"][model_name]

        # If deleted model was active, activate first remaining or set None
        if data.get("active") == model_name:
            remaining = list(data["models"].keys())
            data["active"] = remaining[0] if remaining else None

        if _save_ai_models_to_redis(data):
            logger.info(f"AI model deleted: {model_name}")
            return JSONResponse(content={"success": True, "message": f"Model '{model_name}' deleted", "active": data["active"]})
        else:
            return JSONResponse(content={"error": "Failed to save to Redis"}, status_code=500)
    except Exception as e:
        logger.error(f"Error deleting AI model: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# =============================================================================
# DATABASE CONFIGURATION HELPERS
# =============================================================================

def _get_db_profiles_from_redis():
    """Load all DB profiles from Redis. Returns dict: {"profiles": {...}, "active": "name"}."""
    try:
        r = Redis("redis", 6379, db=REDIS_DB)
        raw = r.get("db_profiles")
        if raw:
            data = json.loads(raw.decode('utf-8') if isinstance(raw, bytes) else raw)
            if "profiles" in data:
                return data
        # Default: create a profile from current hardcoded values
        default_profile = {
            "host": "timescaledb", "port": 5432,
            "database": "monitaqc", "user": "monitaqc", "password": "monitaqc2024"
        }
        data = {"profiles": {"Default": default_profile}, "active": "Default"}
        r.set("db_profiles", json.dumps(data))
        return data
    except Exception as e:
        logger.warning(f"Failed to load DB profiles from Redis: {e}")
    return {"profiles": {}, "active": None}


def _save_db_profiles_to_redis(data):
    """Save all DB profiles to Redis."""
    try:
        r = Redis("redis", 6379, db=REDIS_DB)
        r.set("db_profiles", json.dumps(data))
        return True
    except Exception as e:
        logger.error(f"Failed to save DB profiles to Redis: {e}")
        return False


def _reconnect_db_pool():
    """Reconnect the database pool using the active profile."""
    import services.db as db_module
    data = _get_db_profiles_from_redis()
    active_name = data.get("active")
    if not active_name or active_name not in data.get("profiles", {}):
        return False
    p = data["profiles"][active_name]
    host = p.get("host", "timescaledb")
    port = int(p.get("port", 5432))
    database = p.get("database", "monitaqc")
    user = p.get("user", "monitaqc")
    password = p.get("password", "monitaqc2024")
    # Close existing pool
    if db_module.db_connection_pool:
        try:
            db_module.db_connection_pool.closeall()
        except Exception:
            pass
        db_module.db_connection_pool = None
    # New pool will be created on next get_db_connection() call
    logger.info(f"Database pool will reconnect to {host}:{port}/{database}")
    return True


# =============================================================================
# DATABASE CONFIGURATION ENDPOINTS (multi-profile)
# =============================================================================

@router.get("/api/db_config")
async def get_db_config():
    """Get all configured database profiles and active profile."""
    try:
        data = _get_db_profiles_from_redis()
        # Mask passwords
        safe_profiles = {}
        for name, cfg in data.get("profiles", {}).items():
            pwd = cfg.get("password", "")
            masked = f"***{pwd[-2:]}" if len(pwd) > 2 else "***"
            safe_profiles[name] = {
                "host": cfg.get("host", ""),
                "port": cfg.get("port", 5432),
                "database": cfg.get("database", ""),
                "user": cfg.get("user", ""),
                "password_masked": masked
            }
        return JSONResponse(content={"profiles": safe_profiles, "active": data.get("active")})
    except Exception as e:
        logger.error(f"Error getting DB config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/db_config")
async def save_db_config(config: Dict[str, Any]):
    """Save/update a database profile."""
    try:
        name = config.get("name", "").strip()
        host = config.get("host", "").strip()
        port = int(config.get("port", 5432))
        database = config.get("database", "").strip()
        user = config.get("user", "").strip()
        password = config.get("password", "").strip()

        if not name:
            return JSONResponse(content={"error": "Profile name is required"}, status_code=400)
        if not host:
            return JSONResponse(content={"error": "Host is required"}, status_code=400)

        data = _get_db_profiles_from_redis()
        data["profiles"][name] = {
            "host": host, "port": port, "database": database,
            "user": user, "password": password
        }
        if not data.get("active") or len(data["profiles"]) == 1:
            data["active"] = name

        if _save_db_profiles_to_redis(data):
            logger.info(f"DB profile saved: {name} ({host}:{port}/{database})")
            return JSONResponse(content={"success": True, "message": f"Profile '{name}' saved", "active": data["active"]})
        else:
            return JSONResponse(content={"error": "Failed to save to Redis"}, status_code=500)
    except Exception as e:
        logger.error(f"Error saving DB config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/db_config/activate")
async def activate_db_profile(config: Dict[str, Any]):
    """Activate a database profile and reconnect."""
    try:
        name = config.get("name", "").strip()
        if not name:
            return JSONResponse(content={"error": "Profile name is required"}, status_code=400)

        data = _get_db_profiles_from_redis()
        if name not in data.get("profiles", {}):
            return JSONResponse(content={"error": f"Profile '{name}' not found"}, status_code=404)

        data["active"] = name
        if _save_db_profiles_to_redis(data):
            _reconnect_db_pool()
            logger.info(f"DB profile activated: {name}")
            return JSONResponse(content={"success": True, "message": f"Profile '{name}' activated. DB reconnecting."})
        else:
            return JSONResponse(content={"error": "Failed to save to Redis"}, status_code=500)
    except Exception as e:
        logger.error(f"Error activating DB profile: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.delete("/api/db_config/{profile_name}")
async def delete_db_profile(profile_name: str):
    """Delete a database profile."""
    try:
        data = _get_db_profiles_from_redis()
        if profile_name not in data.get("profiles", {}):
            return JSONResponse(content={"error": f"Profile '{profile_name}' not found"}, status_code=404)

        del data["profiles"][profile_name]
        if data.get("active") == profile_name:
            remaining = list(data["profiles"].keys())
            data["active"] = remaining[0] if remaining else None

        if _save_db_profiles_to_redis(data):
            if data.get("active"):
                _reconnect_db_pool()
            logger.info(f"DB profile deleted: {profile_name}")
            return JSONResponse(content={"success": True, "message": f"Profile '{profile_name}' deleted", "active": data["active"]})
        else:
            return JSONResponse(content={"error": "Failed to save to Redis"}, status_code=500)
    except Exception as e:
        logger.error(f"Error deleting DB profile: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# =============================================================================
# AI TOOLS (for agentic AI query with tool use)
# =============================================================================

def get_ai_tools():
    """Define tools that AI can use to query data autonomously."""
    return [
        {
            "name": "query_database",
            "description": "Execute SQL query on TimescaleDB. Tables: production_metrics (time, encoder_value, ok_counter, ng_counter, shipment, is_moving, downtime_seconds), inference_results (time, shipment, image_path, detections JSONB, detection_count, inference_time_ms, model_used, pipeline_name, module_id, phase_id). Both are time-partitioned hypertables.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "PostgreSQL SQL query. Use NOW() - INTERVAL for time ranges. Detections is JSONB array with {class, confidence, bbox}."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_redis_data",
            "description": "Get real-time values from Redis. Keys: encoder, ok_counter, ng_counter, shipment, is_moving, downtime_seconds. List keys: inference_times, frame_intervals, capture_timestamps, detection_events.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Redis keys to retrieve (string keys use GET, list keys use LRANGE)"
                    }
                },
                "required": ["keys"]
            }
        },
        {
            "name": "get_system_status",
            "description": "Get comprehensive system status: cameras, inference config, pipelines, states, system metrics (CPU/memory/disk), and current configuration.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Which sections to include: cameras, inference, pipelines, states, metrics, config. Omit for all."
                    }
                },
                "required": []
            }
        },
        {
            "name": "call_api_endpoint",
            "description": "Call any internal MonitaQC API endpoint. GET endpoints: /api/cameras, /api/inference, /api/pipelines, /api/pipelines/current, /api/states, /api/inference/stats, /api/system/metrics, /api/latest_detections, /api/timeline_count, /api/cameras/config, /api/models, /api/gradio/models, /api/conf_baselines, /api/color_drift, /api/area_stats, /api/active_classes, /api/config/db_status, /api/shipment_quality_score, /api/audio_settings, /api/timeline_config. POST endpoints: /api/cameras/discover, /api/timeline_clear, /api/why (payload: {mode, metric, value, timestamp, camera, encoder, shipment, window_seconds, language, extra}).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "HTTP method: GET or POST",
                        "enum": ["GET", "POST"]
                    },
                    "endpoint": {
                        "type": "string",
                        "description": "API path, e.g., /api/cameras or /api/inference/stats"
                    },
                    "body": {
                        "type": "object",
                        "description": "Optional JSON body for POST requests"
                    }
                },
                "required": ["method", "endpoint"]
            }
        }
    ]


def execute_tool(tool_name: str, tool_input: dict, watcher_instance=None) -> str:
    """Execute a tool and return results."""
    try:
        if tool_name == "query_database":
            # Use active DB profile for connection
            db_data = _get_db_profiles_from_redis()
            active = db_data.get("active")
            if active and active in db_data.get("profiles", {}):
                p = db_data["profiles"][active]
                db_host, db_port = p.get("host", "timescaledb"), p.get("port", 5432)
                db_name, db_user = p.get("database", "monitaqc"), p.get("user", "monitaqc")
                db_pass = p.get("password", "monitaqc2024")
            else:
                db_host, db_port, db_name, db_user, db_pass = "timescaledb", 5432, "monitaqc", "monitaqc", "monitaqc2024"

            conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_pass)
            cursor = conn.cursor()
            query = tool_input["query"]
            # Safety: limit results to prevent massive responses
            if "LIMIT" not in query.upper():
                query = query.rstrip(";") + " LIMIT 100"
            cursor.execute(query)

            if cursor.description:
                results = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                formatted_results = [dict(zip(column_names, row)) for row in results]
                output = json.dumps(formatted_results, default=str)
            else:
                output = json.dumps({"message": "Query executed successfully", "rowcount": cursor.rowcount})
            cursor.close()
            conn.close()
            return output

        elif tool_name == "get_redis_data":
            # Use direct Redis connection (not broken watcher chain)
            r = Redis("redis", 6379, db=REDIS_DB)
            list_keys = {"inference_times", "frame_intervals", "capture_timestamps", "detection_events"}
            results = {}
            for key in tool_input["keys"]:
                if key in list_keys:
                    raw = r.lrange(key, 0, 49)
                    results[key] = [v.decode('utf-8') if isinstance(v, bytes) else v for v in raw]
                else:
                    value = r.get(key)
                    results[key] = value.decode('utf-8') if isinstance(value, bytes) and value else None
            return json.dumps(results, default=str)

        elif tool_name == "get_system_status":
            sections = tool_input.get("sections") or ["cameras", "inference", "pipelines", "states", "metrics", "config"]
            status = {}

            if "cameras" in sections and watcher_instance:
                cams = {}
                for cid in sorted(watcher_instance.cameras.keys()):
                    cam = watcher_instance.cameras[cid]
                    cams[str(cid)] = {
                        "has_frame": hasattr(cam, 'frame') and cam.frame is not None,
                        "frame_shape": list(cam.frame.shape) if hasattr(cam, 'frame') and cam.frame is not None else None,
                        "source": getattr(cam, 'source', 'unknown'),
                    }
                status["cameras"] = cams

            if "inference" in sections:
                status["inference"] = {
                    "yolo_url": YOLO_INFERENCE_URL,
                    "gradio_model": GRADIO_MODEL,
                    "gradio_confidence": GRADIO_CONFIDENCE_THRESHOLD,
                    "service_type": "YOLO" if YOLO_INFERENCE_URL else "Gradio",
                }

            if "metrics" in sections:
                try:
                    import psutil
                    status["system_metrics"] = {
                        "cpu_percent": psutil.cpu_percent(interval=0.1),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_percent": psutil.disk_usage('/').percent,
                    }
                except Exception:
                    status["system_metrics"] = {"error": "psutil unavailable"}

            if "config" in sections:
                status["config"] = {
                    "capture_mode": CAPTURE_MODE,
                    "ejector_enabled": EJECTOR_ENABLED,
                    "histogram_enabled": HISTOGRAM_ENABLED,
                    "check_class_counts_enabled": CHECK_CLASS_COUNTS_ENABLED,
                    "check_class_counts_classes": CHECK_CLASS_COUNTS_CLASSES,
                    "parent_object_list": PARENT_OBJECT_LIST,
                    "store_annotation_enabled": STORE_ANNOTATION_ENABLED,
                }

            if "states" in sections and watcher_instance:
                states_config = getattr(watcher_instance, 'states_config', {})
                current_state = getattr(watcher_instance, 'current_state', None)
                status["states"] = {
                    "current_state": current_state.name if current_state and hasattr(current_state, 'name') else str(current_state),
                    "available_states": list(states_config.keys()) if states_config else [],
                }

            return json.dumps(status, default=str)

        elif tool_name == "call_api_endpoint":
            # Call internal API endpoint via HTTP
            import requests as req
            method = tool_input.get("method", "GET").upper()
            endpoint = tool_input.get("endpoint", "")
            body = tool_input.get("body")
            url = f"http://127.0.0.1:5050{endpoint}"

            if method == "POST":
                resp = req.post(url, json=body, timeout=10)
            else:
                resp = req.get(url, timeout=10)

            # Truncate large responses
            text = resp.text
            if len(text) > 8000:
                text = text[:8000] + "\n... [truncated]"
            return text

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        logger.error(f"Tool execution error ({tool_name}): {e}")
        return json.dumps({"error": str(e)})


# =============================================================================
# AI QUERY ENDPOINT
# =============================================================================

async def call_ai_model(model: str, api_key: str, system_prompt: str, user_query: str, watcher_instance=None, base_url: str = "", model_id: str = "", tools_enabled: bool = True, max_tokens: int = 4096) -> str:
    """Call the appropriate AI model API.

    3.24.3 — `tools_enabled` (default True for backward-compat with the chat
    endpoint /api/ai_query). When set False, we skip the agentic loop entirely:
    one direct completion, no function calling. /api/why uses this — it
    pre-loads all the context the AI needs, so letting Kimi call tools was
    only adding latency and blocking the event loop on every sync
    `execute_tool(...)` call. With tools off, a Why? request is one
    `client.chat.completions.create(...)` round-trip — much faster, and the
    event loop stays free for other endpoints while we wait on the API.
    """
    try:
        if model == "claude":
            # Anthropic Claude API
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            tools = get_ai_tools() if tools_enabled else []
            messages = [{"role": "user", "content": user_query}]

            # Agentic loop - allow multiple tool calls
            # 3.24.1 — the anthropic SDK's `.create()` is synchronous; calling
            # it bare from an async handler blocks the event loop and starves
            # other concurrent requests (e.g. a second operator clicking 🤔
            # Why? while the first call is in flight). Hand off to a worker
            # thread so the FastAPI loop stays responsive.
            import asyncio as _asyncio
            max_iterations = 5 if tools_enabled else 1
            for iteration in range(max_iterations):
                _create_kwargs = {
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": max_tokens,
                    "system": system_prompt,
                    "messages": messages,
                }
                if tools_enabled and tools:
                    _create_kwargs["tools"] = tools
                response = await _asyncio.to_thread(client.messages.create, **_create_kwargs)

                # Check if Claude wants to use tools
                if response.stop_reason == "end_turn":
                    # No more tools, return final answer
                    for block in response.content:
                        if hasattr(block, 'text'):
                            return block.text
                    return "No response generated"

                elif response.stop_reason == "tool_use":
                    # Add assistant response to messages
                    messages.append({"role": "assistant", "content": response.content})

                    # Execute each tool call
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            tool_result = execute_tool(block.name, block.input, watcher_instance=watcher_instance)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": tool_result
                            })

                    # Add tool results to messages
                    messages.append({"role": "user", "content": tool_results})
                    # Continue loop to let Claude process results
                else:
                    # Unexpected stop reason
                    return f"Unexpected response: {response.stop_reason}"

            return "Maximum iterations reached. Please simplify your query."

        elif model == "chatgpt":
            # OpenAI ChatGPT API with function calling.
            # 3.21.23 — if a base_url was configured (for an OpenAI-compatible
            # relay like ai-trainer.monitait.com / rucode.rdemos.com / Ollama /
            # vLLM / LiteLLM), use it. Otherwise the SDK hits openai.com.
            import openai
            _openai_kwargs = {"api_key": api_key}
            if base_url:
                _openai_kwargs["base_url"] = base_url
            client = openai.OpenAI(**_openai_kwargs)

            # Convert tools to OpenAI format
            functions = []
            if tools_enabled:
                for tool in get_ai_tools():
                    functions.append({
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"]
                    })

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]

            # Agentic loop
            # 3.24.1 — same reasoning as the Claude branch: the openai SDK's
            # `.create()` is sync, hand off to a thread so concurrent /api/why
            # callers don't queue behind each other.
            # 3.24.3 — when tools_enabled=False (the /api/why path) we skip
            # the agentic loop and do exactly one round trip. No function
            # calling, no execute_tool() blocking, no retry.
            import asyncio as _asyncio
            max_iterations = 5 if tools_enabled else 1
            for iteration in range(max_iterations):
                _create_kwargs = {
                    "model": (model_id or "gpt-4o"),  # 3.21.23 — operator-configurable model id
                    "messages": messages,
                    "max_tokens": max_tokens,
                }
                if tools_enabled and functions:
                    _create_kwargs["functions"] = functions
                response = await _asyncio.to_thread(client.chat.completions.create, **_create_kwargs)

                message = response.choices[0].message

                if message.function_call:
                    # Execute function
                    function_name = message.function_call.name
                    function_args = json.loads(message.function_call.arguments)
                    function_result = execute_tool(function_name, function_args, watcher_instance=watcher_instance)

                    # Add to messages
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": function_name,
                            "arguments": message.function_call.arguments
                        }
                    })
                    messages.append({
                        "role": "function",
                        "name": function_name,
                        "content": function_result
                    })
                else:
                    # No function call, return answer
                    return message.content

            return "Maximum iterations reached. Please simplify your query."

        elif model == "gemini":
            # Gemini doesn't support function calling in same way, use basic response
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel('gemini-pro')
            full_prompt = f"{system_prompt}\n\nUser Question: {user_query}"
            response = model_instance.generate_content(full_prompt)
            return response.text

        elif model == "local":
            # Local model (Ollama) - basic response
            import requests
            endpoint = api_key
            response = requests.post(
                f"{endpoint}/api/generate",
                json={
                    "model": "llama2",
                    "prompt": f"{system_prompt}\n\nUser: {user_query}\n\nAssistant:",
                    "stream": False
                }
            )
            return response.json().get("response", "No response from local model")
        else:
            return f"Unsupported model: {model}"

    except ImportError as e:
        return f"Required AI library not installed: {str(e)}. Please install: pip install anthropic openai google-generativeai"
    except Exception as e:
        logger.error(f"AI model call failed: {e}")
        return f"AI request failed: {str(e)}"


@router.post("/api/why")
async def why_for_dot(request: Request, payload: Dict[str, Any]):
    """3.23.0 — Explain a single chart dot.

    Operator clicks a dot on the Charts tab → drawer opens → 🤔 Why? button
    posts {metric, value, timestamp, camera, encoder, shipment, image_path,
    window_seconds} here. We gather a focused context (surrounding detection
    counts + class breakdown over ±window_seconds, capture/inference health,
    state + pipeline name) and ask the active AI model for a 2-sentence
    diagnosis. Response is small + fast — the goal is a quick "huh, that's
    why" not a full RCA.
    """
    try:
        metric = str(payload.get("metric") or "defect")
        value  = payload.get("value")
        ts_in  = payload.get("timestamp")
        camera = payload.get("camera")
        encoder = payload.get("encoder")
        shipment = payload.get("shipment") or "no_shipment"
        win_s   = int(payload.get("window_seconds") or 300)
        # 3.23.1 — entry-point hint:
        #   "dot"     (default)  — explain THIS specific dot at THIS time
        #   "class"             — explain the current behavior of a per-class card
        #   "score"             — explain the shipment quality score
        #   "verdict"           — explain why a shipment got RELEASE/REINSPECT/HOLD
        mode = str(payload.get("mode") or "dot").lower()
        extra = payload.get("extra") or {}
        # 3.23.1 — Respond in the operator's UI language (en/fa/ar/de/tr/ja/es).
        language = str(payload.get("language") or "en").lower().strip()
        _LANG_NAMES = {
            "en": "English", "fa": "Persian (فارسی)", "ar": "Arabic (العربية)",
            "de": "German (Deutsch)", "tr": "Turkish (Türkçe)",
            "ja": "Japanese (日本語)", "es": "Spanish (Español)",
        }
        language_name = _LANG_NAMES.get(language, "English")

        # ----- active AI model -----
        ai_data = _get_ai_models_from_redis()
        active_name = ai_data.get("active")
        if not active_name or active_name not in ai_data.get("models", {}):
            return JSONResponse(
                content={"error": "AI not configured. Add and activate a model in AI Configuration."},
                status_code=400,
            )
        active_model = ai_data["models"][active_name]
        provider = active_model["provider"]
        api_key  = active_model["api_key"]
        base_url = (active_model.get("base_url") or "").strip()
        model_id = (active_model.get("model_id") or "").strip()

        # ----- gather surrounding context from inference_results -----
        # 3.24.2 — mode=class (and any caller that didn't pass a timestamp)
        # defaults to "now". Without this fix the SQL block was skipped and
        # the AI got an empty context, which is why it kept saying "no
        # recent samples" even though the card showed 200k+ samples.
        import time as _time_mod
        if not ts_in:
            ts_in = int(_time_mod.time() * 1000)

        from services.db import get_db_connection, release_db_connection
        ctx_lines = []
        conn = None
        try:
            conn = get_db_connection()
            if conn is not None:
                cur = conn.cursor()
                # 1. detection-count timeline for THIS class over the window
                cur.execute(
                    """
                    SELECT date_trunc('minute', time) AS m, COUNT(*)
                    FROM inference_results, LATERAL jsonb_array_elements(detections) det
                    WHERE time BETWEEN to_timestamp(%s/1000.0) - make_interval(secs => %s)
                                   AND to_timestamp(%s/1000.0) + make_interval(secs => %s)
                      AND (det->>'name') = %s
                    GROUP BY 1 ORDER BY 1 DESC LIMIT 30
                    """,
                    (ts_in, win_s, ts_in, win_s, metric),
                )
                rows = list(reversed(cur.fetchall()))
                if rows:
                    total = sum(int(r[1] or 0) for r in rows)
                    ctx_lines.append(
                        f"Counts of '{metric}' over last {win_s}s ({total:,} total, per-minute):\n"
                        + "\n".join(f"  {r[0].strftime('%H:%M')}  {r[1]}" for r in rows)
                    )
                # 2. top-5 other classes in the same window
                cur.execute(
                    """
                    SELECT (det->>'name') AS cls, COUNT(*) AS n
                    FROM inference_results, LATERAL jsonb_array_elements(detections) det
                    WHERE time BETWEEN to_timestamp(%s/1000.0) - make_interval(secs => %s)
                                   AND to_timestamp(%s/1000.0) + make_interval(secs => %s)
                      AND (det->>'name') IS NOT NULL
                      AND (det->>'name') <> %s
                    GROUP BY 1 ORDER BY n DESC LIMIT 5
                    """,
                    (ts_in, win_s, ts_in, win_s, metric),
                )
                co = [(r[0], int(r[1])) for r in cur.fetchall()]
                if co:
                    ctx_lines.append(
                        "Other classes firing in the same window: "
                        + ", ".join(f"{c}={n}" for c, n in co)
                    )
                cur.close()
        except Exception as _db_err:
            logger.debug(f"why ctx db lookup failed: {_db_err}")
        finally:
            if conn is not None:
                try:
                    release_db_connection(conn)
                except Exception:
                    pass

        # 3.24.2 — for mode=class, also pre-load the per-class analytics
        # blocks the operator stares at on the Process card: confidence
        # baselines (overall + per-camera), CIELAB color, absolute E, and
        # bbox area. Same data, just textualized so the AI can quote it.
        if mode == "class":
            try:
                from routers.timeline import _compute_conf_baselines
                bl = _compute_conf_baselines().get(metric)
                if bl:
                    line = (
                        f"📊 confidence baseline for '{metric}' (last 7d):\n"
                        f"  overall  p5={bl.get('p5')} p50={bl.get('p50')} p95={bl.get('p95')} n={bl.get('n')}"
                    )
                    for cam, cb in (bl.get("by_camera") or {}).items():
                        line += (
                            f"\n  cam {cam}  p5={cb.get('p5')} p50={cb.get('p50')} "
                            f"p95={cb.get('p95')} n={cb.get('n')}"
                        )
                    ctx_lines.append(line)
            except Exception as _e:
                logger.debug(f"why mode=class baseline fetch failed: {_e}")

            try:
                # Reuse the same SQL helpers the endpoints use, in-process.
                import urllib.request, json as _json
                # color_drift + area_stats are JSON-only — call our own
                # FastAPI handlers directly to avoid the HTTP round trip.
                from routers.timeline import get_color_drift, get_area_stats
                try:
                    _cd_resp = await get_color_drift(window="7d")
                    cd = _json.loads(_cd_resp.body.decode()).get("classes", {}).get(metric)
                    if cd:
                        line = f"🎨 CIELAB color for '{metric}' (last 7d): n={cd.get('n')}"
                        if cd.get("L"):
                            L = cd["L"]; a = cd["a"]; b = cd["b"]; E = cd.get("E", {})
                            line += (
                                f"\n  overall  L p5={L['p5']} p50={L['p50']} p95={L['p95']}  "
                                f"a p5={a['p5']} p50={a['p50']} p95={a['p95']}  "
                                f"b p5={b['p5']} p50={b['p50']} p95={b['p95']}  "
                                f"E p5={E.get('p5')} p50={E.get('p50')} p95={E.get('p95')}"
                            )
                        for cam, cc in (cd.get("by_camera") or {}).items():
                            line += (
                                f"\n  cam {cam}  L p50={cc['L']['p50']} a p50={cc['a']['p50']} "
                                f"b p50={cc['b']['p50']} n={cc.get('n')}"
                            )
                        ctx_lines.append(line)
                except Exception as _e:
                    logger.debug(f"why mode=class color fetch failed: {_e}")
                try:
                    _ar_resp = await get_area_stats(window="7d")
                    ar = _json.loads(_ar_resp.body.decode()).get("classes", {}).get(metric)
                    if ar:
                        line = (
                            f"📐 bbox area for '{metric}' (last 7d, px²): "
                            f"p5={ar.get('p5')} p50={ar.get('p50')} p95={ar.get('p95')} n={ar.get('n')}"
                        )
                        for cam, ca in (ar.get("by_camera") or {}).items():
                            line += (
                                f"\n  cam {cam}  p5={ca.get('p5')} p50={ca.get('p50')} "
                                f"p95={ca.get('p95')} n={ca.get('n')}"
                            )
                        ctx_lines.append(line)
                except Exception as _e:
                    logger.debug(f"why mode=class area fetch failed: {_e}")
            except Exception:
                pass

        # ----- real-time system snapshot from Redis -----
        try:
            r = Redis("redis", 6379, db=REDIS_DB)
            def _g(k, d="?"):
                v = r.get(k)
                return v.decode() if v else d
            sys_state = (
                f"encoder={_g('encoder','0')}  "
                f"moving={_g('is_moving','?')}  "
                f"ok={_g('ok_counter','0')}  ng={_g('ng_counter','0')}  "
                f"downtime={_g('downtime_seconds','0')}s"
            )
        except Exception:
            sys_state = "(redis unavailable)"

        # ----- active capture state + pipeline -----
        try:
            from config import load_service_config as _load_svc
            svc = _load_svc() or {}
            current_state = svc.get("current_state_name") or "?"
            current_pipeline = (svc.get("pipeline_config") or {}).get("current_pipeline") or "?"
        except Exception:
            current_state = current_pipeline = "?"

        # ----- compact prompt — we want a 2-sentence answer, not a treatise -----
        from datetime import datetime as _dt
        when_str = _dt.fromtimestamp((ts_in or 0) / 1000.0).strftime("%Y-%m-%d %H:%M:%S") if ts_in else "?"
        cam_str = f"cam {camera}" if camera is not None else "(unknown camera)"
        enc_str = f"encoder {int(encoder):,}" if encoder is not None else ""
        val_str = ""
        if value is not None:
            try:
                v = float(value)
                val_str = f"value={v:.2f}" if v <= 1 else f"metric={v:.1f}"
            except (TypeError, ValueError):
                pass

        system_prompt = (
            "You are an industrial QC assistant. Be CONCISE: at most 2 sentences. "
            "Lead with the most likely cause. If the data is ambiguous, say so plainly. "
            "Do not invent details that aren't in the context. Speak directly to the operator. "
            f"Respond in {language_name}. Technical identifiers like class names "
            "(TB, mean_L, blob_brightness, …), endpoint paths, numeric values, "
            "and timestamps stay in their original form — only the natural-language "
            "prose is translated."
        )
        # 3.23.1 — mode-specific user query so the same endpoint serves "explain
        # this dot", "explain this class right now", "explain this score",
        # and "explain this verdict" with the same context-gathering pipeline.
        if mode == "class":
            user_query = (
                f"Per-class card on Process tab: class '{metric}'.\n"
                f"Current capture state: {current_state}; pipeline: {current_pipeline}.\n"
                f"System: {sys_state}.\n\n"
                + ("\n\n".join(ctx_lines) if ctx_lines else "(no recent samples)")
                + "\n\nWhy is this class behaving this way right now? "
                  "Highlight any obvious drift in count, drift in confidence, or anomalous cameras. "
                  "(2 sentences max.)"
            )
        elif mode == "score":
            try:
                score = float(value) if value is not None else float("nan")
            except (TypeError, ValueError):
                score = float("nan")
            score_str = f"{score:.1f}" if score == score else "?"  # nan check

            # 3.24.1 — fetch the actual quality-score payload so the AI sees
            # the top defects, impact totals, throughput, etc. Without this
            # the model had nothing concrete to point at ("subcomponent
            # data not shown" was the exact complaint).
            score_block = ""
            try:
                from routers.timeline import _compute_quality_payload
                qs = _compute_quality_payload(shipment=shipment, window="24h")
                if qs and qs.get("score") is not None:
                    top = qs.get("top_defects") or []
                    top_lines = "\n".join(
                        f"  {d.get('class')} impact={d.get('impact')} "
                        f"count={d.get('count')} severity={d.get('severity')}"
                        for d in top[:5]
                    ) or "  (no defects with non-zero severity)"
                    score_block = (
                        f"\n\nShipment quality breakdown (window={qs.get('window')}):\n"
                        f"  verdict={qs.get('verdict')}  "
                        f"thresholds=release≥{qs.get('thresholds',{}).get('release')} "
                        f"reinspect≥{qs.get('thresholds',{}).get('reinspect')}\n"
                        f"  impact_total={qs.get('impact_total')} "
                        f"impact_per_unit={qs.get('impact_per_unit')} "
                        f"{qs.get('impact_per_unit_label','')}\n"
                        f"  total_detections={qs.get('total_detections')}\n"
                        f"  encoder_span={qs.get('encoder_span')} {qs.get('encoder_unit','')} "
                        f"frames={qs.get('frame_count')} "
                        f"throughput={qs.get('throughput')} {qs.get('throughput_label','')}\n"
                        f"  Top defects by impact:\n{top_lines}"
                    )
            except Exception as _e:
                logger.debug(f"why mode=score quality fetch failed: {_e}")

            user_query = (
                f"Shipment quality score is {score_str}/100 for shipment '{shipment}'.\n"
                f"Current capture state: {current_state}; pipeline: {current_pipeline}.\n"
                f"System: {sys_state}."
                + score_block
                + ("\n\n" + "\n\n".join(ctx_lines) if ctx_lines else "")
                + "\n\nWhy is the score at this level? What's the biggest contributor? "
                  "If the top-defects list is empty, point at throughput/frame count or normalization mode. (2 sentences max.)"
            )
        elif mode == "verdict":
            verdict = str(extra.get("verdict") or "UNKNOWN")
            user_query = (
                f"Shipment '{shipment}' has been graded {verdict}.\n"
                f"Current capture state: {current_state}; pipeline: {current_pipeline}.\n"
                f"System: {sys_state}.\n\n"
                + ("\n\n".join(ctx_lines) if ctx_lines else "(no surrounding context)")
                + f"\n\nWhy did this shipment land at {verdict}? Name the top driver. (2 sentences max.)"
            )
        else:
            user_query = (
                f"At {when_str}, on {cam_str}, the class '{metric}' fired ({val_str}).\n"
                f"{enc_str}\n"
                f"Shipment: {shipment}.\n"
                f"Current capture state: {current_state}; pipeline: {current_pipeline}.\n"
                f"System: {sys_state}.\n\n"
                + ("\n\n".join(ctx_lines) if ctx_lines else "(no surrounding context available)")
                + "\n\nWhy did this happen? (2 sentences max.)"
            )

        watcher_instance = request.app.state.watcher_instance
        # 3.24.0 — usage tracking. We log regardless of outcome so the per-site
        # bill captures failed attempts too (model timeouts, quota errors).
        import time as _t
        _t0 = _t.time()
        _status = "ok"
        _err = None
        answer = ""
        try:
            # 3.24.3 — Tools OFF for /api/why: we pre-load all the context
            # (DB queries, conf baselines, color/area stats, quality payload)
            # before calling the AI. Letting the model also invoke tools added
            # round-trips that blocked the event loop via the sync execute_tool()
            # chain, which is what made other endpoints feel frozen during a
            # Why? call.
            answer = await call_ai_model(
                provider, api_key, system_prompt, user_query, watcher_instance,
                base_url=base_url, model_id=model_id,
                tools_enabled=False,
            )
        except Exception as _e:
            _status = "error"
            _err = str(_e)
            raise
        finally:
            try:
                from services.ai_usage import log_usage
                # Per-model rate overrides live in audio_settings.ai.<name>.{rate_input_per_mtoken,
                # rate_output_per_mtoken} so resellers can charge their own rate cards.
                _ai_cfg = active_model
                rate_in  = float(_ai_cfg.get("rate_input_per_mtoken")  or 0) or None
                rate_out = float(_ai_cfg.get("rate_output_per_mtoken") or 0) or None
                kwargs = {}
                if rate_in:  kwargs["rate_in"]  = rate_in
                if rate_out: kwargs["rate_out"] = rate_out
                log_usage(
                    endpoint="/api/why",
                    mode=mode,
                    model_name=active_name,
                    provider=provider,
                    prompt_text=(system_prompt + "\n" + user_query),
                    answer_text=answer or "",
                    latency_ms=int((_t.time() - _t0) * 1000),
                    status=_status,
                    operator=request.headers.get("X-Operator") or None,
                    error=_err,
                    **kwargs,
                )
            except Exception:
                pass
        return JSONResponse(content={
            "answer": (answer or "").strip() or "(no answer)",
            "model":  active_name,
            "usage":  f"window=±{win_s}s · ctx_blocks={len(ctx_lines)}",
        })
    except Exception as e:
        logger.error(f"/api/why failed: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/suggest_severities")
async def suggest_severities(request: Request, payload: Dict[str, Any] = None):
    """3.25.2 — Ask the active AI model to suggest a 0-100 Severity value for
    every detected class on the Process tab.

    Severity drives the Shipment Quality Score: higher Severity = a fired
    detection of that class drops the score more. Operators can't guess
    these well without help — the AI gets:
      - the class name
      - its detection count over the last 7d
      - its confidence percentiles (p5/p50/p95) from /api/conf_baselines
      - whether it has area / color analytics on
      - its current Severity (if previously set)
      - an optional `business_context` string the operator can supply
        (e.g. "We make car parts, TB = broken pieces, mean_L = metric not defect")

    Returns a structured list `[{class, suggested_severity, current_severity,
    reason, tier}]`. The UI renders this as a table with per-row Apply
    buttons + an Apply All button.
    """
    payload = payload or {}
    business_context = str(payload.get("business_context") or "").strip()
    language = str(payload.get("language") or "en").lower()

    # Resolve the active model
    ai_data = _get_ai_models_from_redis()
    active_name = ai_data.get("active")
    if not active_name or active_name not in ai_data.get("models", {}):
        return JSONResponse(content={"error": "AI not configured."}, status_code=400)
    active_model = ai_data["models"][active_name]
    provider = active_model["provider"]
    api_key  = active_model["api_key"]
    base_url = (active_model.get("base_url") or "").strip()
    model_id = (active_model.get("model_id") or "").strip()

    # Gather per-class data
    try:
        from routers.timeline import _compute_conf_baselines
        baselines = _compute_conf_baselines() or {}
    except Exception:
        baselines = {}

    try:
        from config import load_service_config as _load_svc
        svc = _load_svc() or {}
        aud = (svc.get("audio_settings") or {})
    except Exception:
        aud = {}

    # 3.25.2 — split classes into "active" (worth AI's attention) and "silent"
    # (no recent data — safely auto-suggest 0 / NONE without an AI roundtrip).
    # This keeps the prompt small AND keeps DeepSeek/Kimi within their token
    # budgets even on sites with 200+ historical classes.
    all_classes = sorted(aud.keys())
    if not all_classes:
        return JSONResponse(content={"suggestions": [], "note": "No classes configured yet."})

    _MIN_N_FOR_AI = 5  # min detections in 7d for a class to be worth asking AI about
    active_classes = []
    silent_classes = []
    for name in all_classes:
        bl = baselines.get(name) or {}
        if (bl.get("n") or 0) >= _MIN_N_FOR_AI:
            active_classes.append(name)
        else:
            silent_classes.append(name)

    lines = []
    for name in active_classes:
        entry = aud.get(name) or {}
        bl = baselines.get(name) or {}
        n = bl.get("n", 0)
        p5  = (bl.get("p5")  or 0) * 100
        p50 = (bl.get("p50") or 0) * 100
        p95 = (bl.get("p95") or 0) * 100
        cur_sev = int(entry.get("severity") or 0)
        color_on = bool(entry.get("color_e"))
        area_on  = bool(entry.get("area"))
        is_shown = bool(entry.get("show"))
        lines.append(
            f"- {name}: n={n}  conf p5={p5:.0f} p50={p50:.0f} p95={p95:.0f}  "
            f"current_severity={cur_sev}  show={is_shown}  color_e={color_on}  area={area_on}"
        )
    data_block = "\n".join(lines) if lines else "(no active classes — every class is silent in the last 7 days)"

    # 3.25.2 — feed the AI a snapshot of recent shipment quality scores so it
    # can tune severities to produce a useful, calibrated score distribution.
    # Without this, the AI guesses severities in a vacuum and you get either
    # "every shipment scores 99" (under-weighted) or "every shipment scores 30"
    # (over-weighted) — both are useless for a RELEASE / REINSPECT decision.
    shipment_block = ""
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is not None:
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT DISTINCT SPLIT_PART(image_path, '/', 1) AS ship,
                           MIN(time) AS first_t
                    FROM inference_results
                    WHERE time > NOW() - INTERVAL '30 days'
                      AND image_path IS NOT NULL
                      AND image_path NOT LIKE 'no_shipment%'
                    GROUP BY 1
                    ORDER BY first_t DESC
                    LIMIT 30
                    """
                )
                recent_ships = [row[0] for row in cur.fetchall() if row[0]]
                cur.close()
            finally:
                try:
                    release_db_connection(conn)
                except Exception:
                    pass
            if recent_ships:
                try:
                    from routers.timeline import _compute_quality_payload
                    scores = []
                    for s in recent_ships:
                        try:
                            qp = _compute_quality_payload(shipment=s, window="24h")
                            sc = qp and qp.get("score")
                            if sc is not None:
                                scores.append({
                                    "ship": s, "score": round(float(sc), 1),
                                    "verdict": qp.get("verdict"),
                                    "top": [d.get("class") for d in (qp.get("top_defects") or [])][:3],
                                })
                        except Exception:
                            continue
                    if scores:
                        ss = sorted(s["score"] for s in scores)
                        def _pct(arr, p):
                            if not arr: return 0
                            k = max(0, min(len(arr)-1, int(round((len(arr)-1) * p))))
                            return arr[k]
                        p5, p50, p95 = _pct(ss, 0.05), _pct(ss, 0.50), _pct(ss, 0.95)
                        shipment_block = (
                            f"\n\nRecent shipment quality scores (last {len(scores)} shipments):\n"
                            f"  distribution  p5={p5}  p50={p50}  p95={p95}\n"
                            "  recent shipments (most recent first):\n"
                            + "\n".join(
                                f"    {s['ship']:>14s}  score={s['score']:>5}  {s.get('verdict') or '?'}  "
                                f"top={','.join(s['top']) if s['top'] else '-'}"
                                for s in scores[:15]
                            )
                            + ("\n    ..." if len(scores) > 15 else "")
                        )
                except Exception:
                    pass
    except Exception:
        pass

    _LANG_NAMES = {
        "en": "English", "fa": "Persian (فارسی)", "ar": "Arabic (العربية)",
        "de": "German", "tr": "Turkish", "ja": "Japanese", "es": "Spanish",
    }
    language_name = _LANG_NAMES.get(language, "English")

    system_prompt = (
        "You are an industrial quality-control consultant. You map detection "
        "classes to a 0–100 Severity score so the Shipment Quality system can "
        "weight defects correctly.\n\n"
        "Severity tiers:\n"
        "  0          — math channel / metric / not a defect at all\n"
        "  1–20  COSMETIC — minor visible issue, low impact\n"
        "  21–50 MODERATE — should be fixed but won't reject the shipment alone\n"
        "  51–80 SERIOUS  — multiple instances mean the shipment is at risk\n"
        "  81–100 CRITICAL — a single instance can reject the shipment\n\n"
        "Use ALL signals to decide:\n"
        "  * class NAME — words like 'broken', 'missing', 'defect', 'crack', 'contamination', "
        "'leak' → SERIOUS or CRITICAL\n"
        "  * NAMES that look like statistics (mean_L, std_L, fft_*, blob_*, sharpness_*, "
        "exposure_*, band_*, tophat_*, grad_*, row/col_*, *_period_px, *_anomaly, *_residual) "
        "→ 0 (these are metrics, not defects)\n"
        "  * very high detection counts (n > 100,000) for a non-metric class often suggest a "
        "common defect — usually MODERATE (21–50). Confidence band tells you whether the "
        "model is decisive: a tight p5–p95 band (<30 points) means a clean signal, deserves "
        "more weight than a wide noisy one.\n"
        "  * if current_severity is already set non-zero, do not stray far without strong reason.\n"
        "  * unknown short-code classes (TB, LL, jea, MD, …) usually map to specific defect "
        "types in textile/jean/PVB QC — assume MODERATE 30–50 unless evidence suggests otherwise.\n\n"
        "Return STRICT JSON ONLY (no prose, no markdown fences) — an array where each item is "
        '`{"class":"<name>","severity":<int 0-100>,"tier":"<NONE|COSMETIC|MODERATE|SERIOUS|CRITICAL>",'
        '"reason":"<one sentence in ' + language_name + '>"}`. '
        "Include EVERY class from the input. Do not invent classes that are not in the input."
    )

    user_query = (
        f"Business context (optional, may be empty):\n"
        f"{business_context or '(none provided — use class-name heuristics + baseline stats)'}\n\n"
        f"Active classes (n >= {_MIN_N_FOR_AI} detections in last 7d):\n{data_block}"
        + shipment_block
        + f"\n\nNow return the JSON array for ALL {len(active_classes)} active classes above. "
        + ("Aim for severities that would keep recent shipment p50 score between 80 and 92, "
           "with worst shipments dropping below 70. The current distribution is shown above — "
           "adjust each class's severity so the resulting score distribution stays in that range. "
           if shipment_block else "")
    )

    # Call AI without tools (one-shot, structured).
    watcher_instance = request.app.state.watcher_instance
    import time as _t
    _t0 = _t.time()
    try:
        raw = await call_ai_model(
            provider, api_key, system_prompt, user_query, watcher_instance,
            base_url=base_url, model_id=model_id, tools_enabled=False,
            max_tokens=16384,   # 3.25.2 — JSON array of ~50-200 classes can be long
        )
    except Exception as e:
        return JSONResponse(content={"error": f"AI call failed: {e}"}, status_code=502)

    # Detect the case where call_ai_model swallowed an upstream error and
    # returned it as a fake "answer" — don't try to parse that as JSON.
    if isinstance(raw, str) and raw.lstrip().lower().startswith(("ai request failed", "error code")):
        return JSONResponse(content={
            "suggestions": [], "model": active_name,
            "upstream_error": raw[:600],
            "hint": "The AI provider returned an error. Try again, or switch model in AI Configuration.",
        }, status_code=502)

    # Strip markdown code fences if any, then parse JSON.
    import json as _json, re as _re
    text = (raw or "").strip()
    text = _re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=_re.IGNORECASE).strip()
    suggestions = []
    parse_err = None
    cur_sevs = {n: int((aud.get(n) or {}).get("severity") or 0) for n in all_classes}
    try:
        arr = _json.loads(text)
        if isinstance(arr, list):
            for item in arr:
                if not isinstance(item, dict):
                    continue
                cls = str(item.get("class") or "").strip()
                if not cls or cls not in cur_sevs:
                    continue
                sev = max(0, min(100, int(item.get("severity") or 0)))
                suggestions.append({
                    "class": cls,
                    "suggested_severity": sev,
                    "current_severity": cur_sevs[cls],
                    "tier": str(item.get("tier") or "").upper(),
                    "reason": str(item.get("reason") or ""),
                })
        # 3.25.2 — auto-suggest 0 for silent classes (no AI roundtrip needed).
        suggested_set = {s["class"] for s in suggestions}
        for cls in silent_classes:
            if cls in suggested_set:
                continue
            suggestions.append({
                "class": cls,
                "suggested_severity": 0,
                "current_severity": cur_sevs.get(cls, 0),
                "tier": "NONE",
                "reason": "No detections in the last 7 days — auto-set to 0.",
            })
        # Stable order: highest suggested first, alphabetical tiebreak
        suggestions.sort(key=lambda s: (-s["suggested_severity"], s["class"]))
    except Exception as e:
        parse_err = f"AI did not return parseable JSON: {e}"

    # Log usage (same shape as /api/why)
    try:
        from services.ai_usage import log_usage
        log_usage(
            endpoint="/api/suggest_severities",
            mode="severity",
            model_name=active_name,
            provider=provider,
            prompt_text=(system_prompt + "\n" + user_query),
            answer_text=(raw or ""),
            latency_ms=int((_t.time() - _t0) * 1000),
            status="ok" if suggestions else "error",
            operator=request.headers.get("X-Operator") or None,
            error=parse_err,
        )
    except Exception:
        pass

    return JSONResponse(content={
        "suggestions": suggestions,
        "model": active_name,
        "parse_error": parse_err,
        "raw_preview": (raw or "")[:300] if parse_err else None,
    })


@router.post("/api/states/ai_recommend")
async def ai_recommend_state(request: Request, payload: Dict[str, Any] = None):
    """3.25.3 — Ask the active AI to pick the best capture state for the
    current line + product.

    Operator hits the "🤖 ?" button next to the Dashboard state picker.
    AI sees:
      - the list of defined capture states with their phases (which cameras
        fire, light mode, delay, steps)
      - the currently active state
      - the active inference pipeline + how many cameras are physically
        present
      - last-hour detection rate so it can sense whether the line is
        actually running
      - the operator's optional product description ("knit production",
        "PVB film, backlit", "jeans dye line", …)

    Returns:
      {
        "recommended_state": "<name>",
        "current_state":     "<name>",
        "reason":            "one short sentence",
        "confidence":        "high|medium|low",
        "alternatives":      [{"name":"...","reason":"..."}, ...]
      }
    """
    payload = payload or {}
    product_context = str(payload.get("product_context") or "").strip()
    language = str(payload.get("language") or "en").lower()

    # active AI
    ai_data = _get_ai_models_from_redis()
    active_name = ai_data.get("active")
    if not active_name or active_name not in ai_data.get("models", {}):
        return JSONResponse(content={"error": "AI not configured."}, status_code=400)
    am = ai_data["models"][active_name]
    provider = am["provider"]; api_key = am["api_key"]
    base_url = (am.get("base_url") or "").strip()
    model_id = (am.get("model_id") or "").strip()

    # gather state + pipeline + camera + detection context
    try:
        from config import load_service_config as _load_svc
        svc = _load_svc() or {}
    except Exception:
        svc = {}
    states = (svc.get("states") or {})
    current_state = svc.get("current_state_name") or "?"
    pipeline_name = (svc.get("pipeline_config") or {}).get("current_pipeline") or "?"
    cameras_cfg = svc.get("cameras") or {}
    n_cameras = len([k for k, v in cameras_cfg.items() if isinstance(v, dict)])

    # describe each state for the AI
    state_lines = []
    for name, body in (states or {}).items():
        if not isinstance(body, dict):
            continue
        phases = body.get("phases") or []
        bits = []
        for i, p in enumerate(phases):
            cams = p.get("cameras") or p.get("cams") or []
            bits.append(
                f"phase{i}(cams={cams}, light={p.get('light_mode','?')}, "
                f"delay={p.get('delay','?')}s, steps={p.get('steps','?')}, "
                f"analog={p.get('analog','?')})"
            )
        marker = " ← currently active" if name == current_state else ""
        state_lines.append(f"- {name!r}{marker}: {' '.join(bits) or '(no phases)'}")
    states_block = "\n".join(state_lines) if state_lines else "(no capture states defined)"

    # last-hour activity from inference_results so the AI knows whether the line is running
    activity_block = "(no recent activity data)"
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is not None:
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT (det->>'name') AS cls, COUNT(*) AS n,
                           COUNT(DISTINCT (det->>'_cam')) AS cams
                    FROM inference_results, LATERAL jsonb_array_elements(detections) det
                    WHERE time > NOW() - INTERVAL '1 hour'
                      AND (det->>'name') IS NOT NULL
                    GROUP BY 1 ORDER BY n DESC LIMIT 8
                    """
                )
                rows = cur.fetchall()
                if rows:
                    activity_block = (
                        "Last 1h top detected classes (name=count cams):\n"
                        + "\n".join(f"  {r[0]}={r[1]} cams={r[2]}" for r in rows)
                    )
                cur.close()
            finally:
                try: release_db_connection(conn)
                except Exception: pass
    except Exception:
        pass

    _LANG_NAMES = {
        "en": "English", "fa": "Persian (فارسی)", "ar": "Arabic",
        "de": "German", "tr": "Turkish", "ja": "Japanese", "es": "Spanish",
    }
    language_name = _LANG_NAMES.get(language, "English")

    system_prompt = (
        "You are a quality-control consultant choosing the best capture state "
        "for a production line. A capture state defines which cameras fire, "
        "with what lighting, delay between phases, and how many steps per cycle.\n\n"
        "Decision heuristics:\n"
        "  * If the line uses ALL cameras simultaneously, prefer a state whose "
        "phase 0 has all cams listed.\n"
        "  * For continuous high-speed production (knit, PVB film, conveyor "
        "fabric): prefer 'infinite' or 'infinite-max' (steps=-1).\n"
        "  * For step-by-step inspection (one piece at a time, e.g. jeans QC, "
        "discrete part inspection): prefer 'default' or named single-step states.\n"
        "  * Light-mode hints in state names matter (U_ON / B_ON / KC-Back / etc.) — "
        "they reflect the lighting rig that state was built for. Match the "
        "product description to the lighting.\n"
        "  * If the line is currently producing nothing (activity block shows no "
        "recent detections), suggest the operator's stated need or keep the current.\n\n"
        f"Respond in {language_name}. Return STRICT JSON ONLY (no prose, no "
        "markdown fences) with shape:\n"
        '  {"recommended_state": "<exact name from the list>",\n'
        '   "reason": "<one sentence>",\n'
        '   "confidence": "high|medium|low",\n'
        '   "alternatives": [{"name":"<other-state>","reason":"<one sentence>"}, ...]}\n'
        "Only choose from the state names provided."
    )

    user_query = (
        f"Product / line context (optional):\n{product_context or '(none provided)'}\n\n"
        f"Active inference pipeline: {pipeline_name}\n"
        f"Cameras configured: {n_cameras}\n"
        f"Currently active capture state: {current_state}\n\n"
        f"Available capture states:\n{states_block}\n\n"
        f"{activity_block}\n\n"
        f"Now pick the best state."
    )

    watcher_instance = request.app.state.watcher_instance
    import time as _t
    _t0 = _t.time()
    try:
        raw = await call_ai_model(
            provider, api_key, system_prompt, user_query, watcher_instance,
            base_url=base_url, model_id=model_id, tools_enabled=False, max_tokens=2048,
        )
    except Exception as e:
        return JSONResponse(content={"error": f"AI call failed: {e}"}, status_code=502)

    if isinstance(raw, str) and raw.lstrip().lower().startswith(("ai request failed", "error code")):
        return JSONResponse(content={
            "error": "upstream_ai_error", "upstream_error": raw[:600],
            "hint": "Try again or switch model in AI Configuration.",
        }, status_code=502)

    import json as _json, re as _re
    text = (raw or "").strip()
    text = _re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=_re.IGNORECASE).strip()
    try:
        d = _json.loads(text)
    except Exception as e:
        return JSONResponse(content={
            "error": f"AI did not return valid JSON: {e}",
            "raw_preview": text[:300],
        }, status_code=502)

    # Validate the suggested state name exists
    rec = str(d.get("recommended_state") or "").strip()
    if rec not in (states or {}):
        return JSONResponse(content={
            "error": f"AI suggested an unknown state {rec!r}",
            "available_states": list((states or {}).keys()),
            "raw": d,
        }, status_code=502)

    # Log usage
    try:
        from services.ai_usage import log_usage
        log_usage(
            endpoint="/api/states/ai_recommend",
            mode="state_pick",
            model_name=active_name,
            provider=provider,
            prompt_text=(system_prompt + "\n" + user_query),
            answer_text=text,
            latency_ms=int((_t.time() - _t0) * 1000),
            status="ok",
            operator=request.headers.get("X-Operator") or None,
        )
    except Exception:
        pass

    return JSONResponse(content={
        "recommended_state": rec,
        "current_state": current_state,
        "reason": str(d.get("reason") or ""),
        "confidence": str(d.get("confidence") or "").lower(),
        "alternatives": d.get("alternatives") or [],
        "model": active_name,
    })


@router.post("/api/apply_severities")
async def apply_severities(payload: Dict[str, Any]):
    """Bulk-apply suggested severity values to service_config.audio_settings.
    Payload: {"updates": [{"class": "...", "severity": N}, ...]}.
    Returns how many rows were updated."""
    updates = payload.get("updates") or []
    if not isinstance(updates, list) or not updates:
        return JSONResponse(content={"error": "updates list required"}, status_code=400)
    from config import load_service_config as _load_svc, save_service_config as _save_svc
    svc = _load_svc() or {}
    aud = svc.get("audio_settings") or {}
    n_applied = 0
    for u in updates:
        if not isinstance(u, dict):
            continue
        cls = str(u.get("class") or "").strip()
        if not cls or cls not in aud:
            continue
        try:
            sev = max(0, min(100, int(u.get("severity") or 0)))
        except (TypeError, ValueError):
            continue
        if not isinstance(aud[cls], dict):
            aud[cls] = {}
        aud[cls]["severity"] = sev
        n_applied += 1
    svc["audio_settings"] = aud
    _save_svc(svc)
    try:
        from services.draw_filters import invalidate_cache as _df_inv
        _df_inv()
    except Exception:
        pass
    return JSONResponse(content={"success": True, "applied": n_applied})


@router.post("/api/ai_query")
async def query_ai(request: Request, payload: Dict[str, Any]):
    """Query AI with production data from TimescaleDB and real-time metrics."""
    try:
        user_query = payload.get("query", "")
        if not user_query:
            return JSONResponse(content={"error": "Query is required"}, status_code=400)

        # Get active AI model from multi-model storage
        ai_data = _get_ai_models_from_redis()
        active_name = ai_data.get("active")
        if not active_name or active_name not in ai_data.get("models", {}):
            return JSONResponse(content={
                "error": "AI not configured. Please add and activate a model in the AI Configuration panel."
            }, status_code=400)

        active_model = ai_data["models"][active_name]
        model = active_model["provider"]
        api_key = active_model["api_key"]
        # 3.21.23 — optional overrides for OpenAI-compatible relays / custom model IDs
        base_url = (active_model.get("base_url") or "").strip()
        model_id = (active_model.get("model_id") or "").strip()

        r = Redis("redis", 6379, db=REDIS_DB)

        # Gather current system context from Redis
        def _r_get(key, default="N/A"):
            v = r.get(key)
            if v is None:
                return default
            return v.decode('utf-8') if isinstance(v, bytes) else v

        # Build comprehensive camera info
        watcher = request.app.state.watcher_instance
        camera_info = "No cameras"
        if watcher and watcher.cameras:
            cam_details = []
            for cid in sorted(watcher.cameras.keys()):
                cam = watcher.cameras[cid]
                has_frame = hasattr(cam, 'frame') and cam.frame is not None
                cam_details.append(f"Camera {cid}: {'active' if has_frame else 'no frame'}")
            camera_info = "; ".join(cam_details)

        # Build comprehensive AI prompt with full schema and context
        system_prompt = f"""You are a powerful AI assistant embedded in the MonitaQC industrial quality control system. You have full access to query the database, read real-time sensor data, check camera status, and analyze system health.

## Current System State
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Encoder: {_r_get('encoder', '0')} | OK: {_r_get('ok_counter', '0')} | NG: {_r_get('ng_counter', '0')}
- Shipment: {_r_get('shipment', 'none')} | Moving: {_r_get('is_moving', 'false')}
- Downtime: {_r_get('downtime_seconds', '0')}s
- Cameras: {camera_info}
- Inference: {'YOLO (' + YOLO_INFERENCE_URL + ')' if YOLO_INFERENCE_URL else 'Gradio HuggingFace'}

## Database Schema (TimescaleDB - PostgreSQL)
You MUST use these exact table and column names:

### Table: production_metrics (hypertable, partitioned by time)
| Column | Type | Description |
|--------|------|-------------|
| time | TIMESTAMPTZ | Timestamp (auto-set, use for time queries) |
| encoder_value | INTEGER | Conveyor encoder position |
| ok_counter | INTEGER | Cumulative OK product count |
| ng_counter | INTEGER | Cumulative NG (defective) product count |
| shipment | TEXT | Current shipment/batch identifier |
| is_moving | BOOLEAN | Whether conveyor is moving |
| downtime_seconds | INTEGER | Accumulated downtime |

### Table: inference_results (hypertable, partitioned by time)
| Column | Type | Description |
|--------|------|-------------|
| time | TIMESTAMPTZ | When inference ran |
| shipment | TEXT | Shipment/batch identifier |
| image_path | TEXT | Path to the inspected image |
| detections | JSONB | Array of detection objects (class, confidence, bbox) |
| detection_count | INTEGER | Number of objects detected |
| inference_time_ms | INTEGER | How long inference took in ms |
| model_used | TEXT | Which AI model was used |
| pipeline_name | TEXT | Pipeline that produced this result |
| module_id | TEXT | Inference module identifier |
| phase_id | INTEGER | Capture phase number |

### Common Query Patterns
- Recent defects: `SELECT time, detections, detection_count, inference_time_ms FROM inference_results ORDER BY time DESC LIMIT 20`
- Defect rate last hour: `SELECT COUNT(*) FILTER (WHERE detection_count > 0) as defective, COUNT(*) as total FROM inference_results WHERE time > NOW() - INTERVAL '1 hour'`
- Production summary: `SELECT shipment, MAX(ok_counter) as ok, MAX(ng_counter) as ng FROM production_metrics WHERE time > NOW() - INTERVAL '24 hours' GROUP BY shipment`
- Detections JSON example: each item in detections array has: {{"class": "defect_name", "confidence": 0.95, "bbox": [x1,y1,x2,y2]}}
- To parse JSONB: `SELECT jsonb_array_elements(detections)->>'class' as defect_class FROM inference_results WHERE detection_count > 0`

## Available Redis Keys (real-time data)
encoder, ok_counter, ng_counter, shipment, is_moving, downtime_seconds, inference_times (list), frame_intervals (list), capture_timestamps (list), detection_events (list of JSON)

## Tools Available
1. **query_database** - Execute any SQL query on TimescaleDB
2. **get_redis_data** - Read real-time values from Redis
3. **get_system_status** - Get full system status (cameras, inference, config)
4. **call_api_endpoint** - Call any internal API endpoint for data or actions

## Formatting Rules
- Write clear, natural language. Avoid unnecessary jargon.
- Use bullet points and numbered lists for readability.
- Include concrete numbers, percentages, and time references.
- Use emojis sparingly: ✅ ❌ 📊 ⚠️ 🔍
- When showing data, use tables for comparisons.
- Provide actionable insights and recommendations.
- Keep responses concise but complete."""

        # Call AI API based on model
        ai_response = await call_ai_model(model, api_key, system_prompt, user_query, watcher_instance=watcher, base_url=base_url, model_id=model_id)

        return JSONResponse(content={"response": ai_response, "model": model})

    except Exception as e:
        logger.error(f"Error querying AI: {e}")
        return JSONResponse(content={"error": f"AI query failed: {str(e)}"}, status_code=500)
