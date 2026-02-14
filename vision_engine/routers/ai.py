from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from config import (
    REDIS_HOST, REDIS_PORT, load_data_file, save_data_file,
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
        r = Redis("redis", 6379, db=0)
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
        r = Redis("redis", 6379, db=0)
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
        # Strip API keys for display (show only last 4 chars)
        safe_models = {}
        for name, cfg in data.get("models", {}).items():
            key = cfg.get("api_key", "")
            masked = f"***{key[-4:]}" if len(key) > 4 else "***"
            safe_models[name] = {"provider": cfg["provider"], "api_key_masked": masked}
        return JSONResponse(content={"models": safe_models, "active": data.get("active")})
    except Exception as e:
        logger.error(f"Error getting AI config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/ai_config")
async def save_ai_config(config: Dict[str, Any]):
    """Save/update an AI model configuration."""
    try:
        name = config.get("name", "").strip()
        provider = config.get("provider", "")
        api_key = config.get("api_key", "")

        if not name:
            return JSONResponse(content={"error": "Model name is required"}, status_code=400)
        if not provider:
            return JSONResponse(content={"error": "Provider is required"}, status_code=400)
        if not api_key:
            return JSONResponse(content={"error": "API key is required"}, status_code=400)

        data = _get_ai_models_from_redis()
        data["models"][name] = {"provider": provider, "api_key": api_key}

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
        r = Redis("redis", 6379, db=0)
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
        r = Redis("redis", 6379, db=0)
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
            "description": "Call any internal MonitaQC API endpoint. GET endpoints: /api/cameras, /api/inference, /api/pipelines, /api/pipelines/current, /api/states, /api/inference/stats, /api/system/metrics, /api/latest_detections, /api/timeline_count, /api/cameras/config, /api/models, /api/gradio/models. POST endpoints: /api/cameras/discover, /api/timeline_clear.",
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
            r = Redis("redis", 6379, db=0)
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

async def call_ai_model(model: str, api_key: str, system_prompt: str, user_query: str, watcher_instance=None) -> str:
    """Call the appropriate AI model API with tool support."""
    try:
        if model == "claude":
            # Anthropic Claude API with tool use
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            tools = get_ai_tools()
            messages = [{"role": "user", "content": user_query}]

            # Agentic loop - allow multiple tool calls
            max_iterations = 5
            for iteration in range(max_iterations):
                response = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=4096,
                    system=system_prompt,
                    messages=messages,
                    tools=tools
                )

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
            # OpenAI ChatGPT API with function calling
            import openai
            client = openai.OpenAI(api_key=api_key)

            # Convert tools to OpenAI format
            functions = []
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
            max_iterations = 5
            for iteration in range(max_iterations):
                response = client.chat.completions.create(
                    model="gpt-4o",  # Use gpt-4o for 128K context
                    messages=messages,
                    functions=functions,
                    max_tokens=4096
                )

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

        r = Redis("redis", 6379, db=0)

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
        ai_response = await call_ai_model(model, api_key, system_prompt, user_query, watcher_instance=watcher)

        return JSONResponse(content={"response": ai_response, "model": model})

    except Exception as e:
        logger.error(f"Error querying AI: {e}")
        return JSONResponse(content={"error": f"AI query failed: {str(e)}"}, status_code=500)
