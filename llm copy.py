import json
import os
import subprocess


ALLOWED_ACTIONS = {
    "system-info",
    "process-list",
    "process-start",
    "process-kill",
    "fs-list",
    "fs-mkdir",
    "fs-rm",
    "fs-mv",
    "fs-cp",
    "network-info",
    "workflow",
    "open-website",
    "open-app",
    "memory-warning",
    "devices",
    "external-storage",
    "connections",
    "install-app",
    "uninstall-app",
    "help",
    "exit",
}


SYSTEM_PROMPT = (
    "You are a command router for a local OS assistant. "
    "Return ONLY JSON with keys: action, params. "
    "Allowed actions: system-info, process-list, process-start, process-kill, "
    "fs-list, fs-mkdir, fs-rm, fs-mv, fs-cp, network-info, workflow, open-website, "
    "open-app, memory-warning, devices, external-storage, connections, "
    "install-app, uninstall-app, help, exit. "
    "Params should be a JSON object with the needed fields. "
    "Examples: "
    "{\"action\":\"open-website\",\"params\":{\"site\":\"chatgpt\"}} "
    "{\"action\":\"workflow\",\"params\":{\"target\":\"whatsapp in chrome\",\"action\":\"send hi\"}} "
    "{\"action\":\"fs-mv\",\"params\":{\"src\":\"a.txt\",\"dst\":\"b.txt\"}} "
    "{\"action\":\"process-kill\",\"params\":{\"pid\":\"123\"}}"
)


def _extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


DEFAULT_MODEL = "qwen2.5:7b"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "llm_config.json")


def _load_model():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                model = data.get("model")
                if model:
                    return model
        except (OSError, json.JSONDecodeError):
            pass
    return DEFAULT_MODEL


def llm_parse(user_text, model=None):
    model = model or _load_model()
    try:
        output = subprocess.check_output(
            ["ollama", "run", model, f"{SYSTEM_PROMPT}\nUser: {user_text}\nJSON:"],
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return {"action": "help", "params": {"error": "llm_unavailable"}}

    data = _extract_json(output)
    if not data:
        return {"action": "help", "params": {"error": "llm_bad_output"}}

    action = data.get("action")
    params = data.get("params", {})
    if action not in ALLOWED_ACTIONS:
        return {"action": "help", "params": {"error": "llm_unknown_action"}}
    if not isinstance(params, dict):
        params = {}
    return {"action": action, "params": params}
