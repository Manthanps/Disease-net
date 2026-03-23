from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, Optional, Tuple

import api


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for a disease prediction project. "
    "Use the available tools to inspect schemas, run tabular predictions, and explain results. "
    "If a user asks for a prediction, call the appropriate tool. "
    "If needed, ask for missing fields but prefer using defaults from the schema."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangChain terminal interface")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="LLM model name (default: env OPENAI_MODEL or gpt-4o-mini)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature (default: 0.2)",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt",
    )
    parser.add_argument(
        "--once",
        default="",
        help="Run a single query and exit",
    )
    return parser.parse_args()


def _ensure_langchain():
    try:
        from langchain_openai import ChatOpenAI  # noqa: F401
        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage  # noqa: F401
        from langchain_core.tools import tool  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "LangChain is not available. Install dependencies: "
            "`pip install langchain langchain-openai`.\n"
            f"Import error: {exc}"
        )


def _parse_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"value": payload}
    return {"value": payload}


def build_tools():
    _ensure_langchain()
    from langchain_core.tools import tool

    @tool
    def list_diseases() -> Dict[str, Any]:
        """List supported diseases and available modes."""
        return {
            "tabular": list(api.DATASETS_CONFIG.get("tabular", {}).keys()),
            "images": list(api.DATASETS_CONFIG.get("images", {}).keys()),
        }

    @tool
    def get_schema(disease: str) -> Dict[str, Any]:
        """Get tabular schema and defaults for a disease (blood, lung, heart)."""
        return api.schema_tabular(disease)

    @tool
    def predict_tabular(disease: str, payload: Any) -> Dict[str, Any]:
        """Run a tabular prediction for a disease using a JSON payload."""
        return api.predict_tabular(disease, _parse_payload(payload))

    @tool
    def explain_tabular(disease: str, payload: Any) -> Dict[str, Any]:
        """Run a tabular prediction with SHAP explanations."""
        return api.predict_tabular_explain(disease, _parse_payload(payload))

    return [list_diseases, get_schema, predict_tabular, explain_tabular]


def _iter_tool_calls(msg: Any) -> Iterable[Tuple[str, Dict[str, Any], Optional[str]]]:
    calls = getattr(msg, "tool_calls", None)
    if calls is None:
        calls = msg.additional_kwargs.get("tool_calls", []) if hasattr(msg, "additional_kwargs") else []
    for call in calls:
        if isinstance(call, dict):
            name = call.get("name")
            args = call.get("args") or call.get("arguments") or {}
            call_id = call.get("id")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"value": args}
        else:
            name = getattr(call, "name", None)
            args = getattr(call, "args", {})
            call_id = getattr(call, "id", None)
        if name:
            yield name, args, call_id


def run_repl(model: str, temperature: float, system_prompt: str, once: str = "") -> None:
    _ensure_langchain()
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

    tools = build_tools()
    tool_map = {t.name: t for t in tools}

    llm = ChatOpenAI(model=model, temperature=temperature)
    llm_tools = llm.bind_tools(tools)

    messages = [SystemMessage(content=system_prompt)]

    def _handle_user_input(user_text: str) -> None:
        messages.append(HumanMessage(content=user_text))
        ai_msg = llm_tools.invoke(messages)
        messages.append(ai_msg)

        tool_calls = list(_iter_tool_calls(ai_msg))
        if not tool_calls:
            print(ai_msg.content)
            return

        for name, args, call_id in tool_calls:
            tool = tool_map.get(name)
            if tool is None:
                messages.append(ToolMessage(content=f"Unknown tool: {name}", tool_call_id=call_id))
                continue
            result = tool.invoke(args)
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call_id))

        final_msg = llm_tools.invoke(messages)
        messages.append(final_msg)
        print(final_msg.content)

    if once:
        _handle_user_input(once)
        return

    print("LangChain terminal interface. Type 'exit' to quit. Type 'help' for tips.")
    while True:
        try:
            user_text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if user_text.lower() == "help":
            print("Example: 'Predict blood with age=45 and WBC=6.1' or 'Show schema for heart'.")
            continue
        _handle_user_input(user_text)


def main() -> None:
    args = parse_args()
    run_repl(args.model, args.temperature, args.system, args.once)


if __name__ == "__main__":
    main()
