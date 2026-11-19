"""Seed harness (v000): the unevolved starting point.

It is deliberately minimal, like the "basic shell tool" seeds in the paper:
the model answers in JSON, the harness runs the commands and shows the output.
Nothing checks the work, output is cut short, and "done" is taken on trust.
The RSI loop's job is to find out which of these weaknesses cost tasks and fix them.
"""
import json

SYSTEM = """You are a coding agent working in a Linux shell. The working directory is /work.
Reply with one JSON object and nothing else:
{"commands": ["<shell command>", ...], "task_complete": false}
Set "task_complete" to true when the task is finished."""

MAX_OUTPUT = 600


def parse_reply(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def run_task(task, policy, sandbox):
    messages = [{"role": "user", "content": "Task:\n" + task["instruction"]}]
    turns = 0
    while policy.calls_left > 0:
        reply = policy.ask(SYSTEM, messages)
        turns += 1
        messages.append({"role": "assistant", "content": reply})
        try:
            action = parse_reply(reply)
        except ValueError:
            messages.append({"role": "user", "content": "Could not parse your reply as JSON."})
            continue
        outputs = []
        for cmd in action.get("commands", []):
            result = sandbox.run(cmd, timeout=10)
            outputs.append("$ " + cmd + "\n" + result.stdout[:MAX_OUTPUT])
        if action.get("task_complete"):
            return {"stop": "model_declared_complete", "turns": turns}
        messages.append({"role": "user", "content": "\n".join(outputs) or "(no commands were run)"})
    return {"stop": "out_of_turns", "turns": turns}
