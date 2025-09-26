"""Naive ReAct client for AIOpsLab (extended with CLI and OpenRouter).

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). 
React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Code: https://github.com/ysymyth/ReAct
Paper: https://arxiv.org/abs/2210.03629
"""

import asyncio
import argparse
import json
import os
from pathlib import Path

import tiktoken
from aiopslab.orchestrator import Orchestrator
from aiopslab.orchestrator.problems.registry import ProblemRegistry
from clients.utils.llm import OpenRouterClient
from clients.utils.templates import DOCS

from dotenv import load_dotenv

# Load environment variables from the .env file
_ = load_dotenv()

RESP_INSTR = """DO NOT REPEAT ACTIONS! Respond with:
Thought: <your thought on the previous output>
Action: <your action towards mitigating>
"""

# ----------------------------
# Utility functions
# ----------------------------

def count_message_tokens(message, enc) -> int:
    # Each message format adds ~4 tokens of overhead
    tokens = 4  # <|start|>role/name + content + <|end|>
    tokens += len(enc.encode(message.get("content", "")))
    return tokens


def trim_history_to_token_limit(history, max_tokens: int = 120000, model: str = "gpt-4") -> list[dict]:
    enc = tiktoken.encoding_for_model(model)

    trimmed = []
    total_tokens = 0

    # Always include the last message
    last_msg = history[-1]
    last_msg_tokens = count_message_tokens(last_msg, enc)

    if last_msg_tokens > max_tokens:
        # If even the last message is too big, truncate its content
        truncated_content = enc.decode(enc.encode(last_msg["content"])[:max_tokens - 4])
        return [{"role": last_msg["role"], "content": truncated_content}]
    
    trimmed.insert(0, last_msg)
    total_tokens += last_msg_tokens

    # Add earlier messages in reverse until limit is reached
    for message in reversed(history[:-1]):
        message_tokens = count_message_tokens(message, enc)
        if total_tokens + message_tokens > max_tokens:
            break
        trimmed.insert(0, message)
        total_tokens += message_tokens

    return trimmed


def get_completed_problems(results_dir: Path, agent_name: str, model: str) -> set[str]:
    """Get set of completed problem IDs from existing result files."""
    completed = set()

    organized_dir = results_dir / agent_name / model.replace("/", "_")
    if organized_dir.exists():
        for result_file in organized_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    if 'problem_id' in data:
                        completed.add(data['problem_id'])
            except (json.JSONDecodeError, IOError):
                continue

    # Legacy flat structure
    for result_file in results_dir.glob("*.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                if ('problem_id' in data and
                    data.get('agent') == agent_name and
                    model.split('/')[-1] in str(result_file)):
                    completed.add(data['problem_id'])
        except (json.JSONDecodeError, IOError):
            continue

    return completed


def setup_results_directory(model: str, agent_name: str = "react") -> Path:
    """Setup organized results directory structure."""
    results_base = Path("aiopslab/data/results")
    model_safe = model.replace("/", "_")
    results_dir = results_base / agent_name / model_safe
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# ----------------------------
# Agent
# ----------------------------

class Agent:
    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.history: list[dict] = []
        self.llm = OpenRouterClient(model=model)
        self.model = model

    def init_context(self, problem_desc: str, instructions: str, apis: dict):
        """Initialize the context for the agent."""

        self.shell_api = self._filter_dict(apis, lambda k, _: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)
        self.telemetry_apis = self._filter_dict(
            apis, lambda k, _: "exec_shell" not in k and "submit" not in k
        )

        stringify_apis = lambda apis: "\n\n".join(
            [f"{k}\n{v}" for k, v in apis.items()]
        )

        self.system_message = DOCS.format(
            prob_desc=problem_desc,
            telemetry_apis=stringify_apis(self.telemetry_apis),
            shell_api=stringify_apis(self.shell_api),
            submit_api=stringify_apis(self.submit_api),
        )

        self.task_message = instructions

        self.history.append({"role": "system", "content": self.system_message})
        self.history.append({"role": "user", "content": self.task_message})

    async def get_action(self, input: str) -> str:
        """Wrapper to interface the agent with OpsBench."""
        self.history.append({"role": "user", "content": self._add_instr(input)})
        trimmed_history = trim_history_to_token_limit(self.history)
        try:
            response = self.llm.run(trimmed_history)
            print(f"===== Agent (OpenRouter - {self.model}) ====\n{response[0]}")

            self.history.append({"role": "assistant", "content": response[0]})
            return response[0]
        except Exception as e:
            print(f"ReAct API error: {e}")
            fallback_response = f"Error occurred while calling API: {e}"
            self.history.append({"role": "assistant", "content": fallback_response})
            return fallback_response

    def _filter_dict(self, dictionary, filter_func):
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}

    def _add_instr(self, input: str) -> str:
        return input + "\n\n" + RESP_INSTR


# ----------------------------
# Main entry
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ReAct agent on AIOpsLab problems')
    parser.add_argument('--skip-completed', action='store_true',
                       help='Skip problems that have already been completed')
    parser.add_argument('--problem-ids', nargs='+',
                       help='Run only specific problem IDs')
    parser.add_argument('--max-steps', type=int, default=30,
                       help='Maximum steps per problem (default: 30)')
    parser.add_argument('--model', type=str,
                       default=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                       help='OpenRouter model to use')

    args = parser.parse_args()

    model = args.model
    agent_name = "react"

    results_dir = setup_results_directory(model, agent_name)
    print(f"Results will be saved to: {results_dir}")

    problems = ProblemRegistry().PROBLEM_REGISTRY

    if args.problem_ids:
        def read_lines_without_newline(filepath: str) -> list[str]:
            """Read all lines from a file, stripping trailing newlines."""
            with open(filepath, "r", encoding="utf-8") as f:
                return [line.rstrip("\n") for line in f]
        problems = {pid: problems[pid] for pid in read_lines_without_newline(args.problem_ids[0]) if pid in problems}
        if not problems:
            print("No valid problem IDs found")
            exit(1)

    if args.skip_completed:
        completed_problems = get_completed_problems(
            Path("aiopslab/data/results"), agent_name, model
        )
        problems = {pid: prob for pid, prob in problems.items()
                   if pid not in completed_problems}

        print(f"Found {len(completed_problems)} completed problems")
        print(f"Running {len(problems)} remaining problems")

        if not problems:
            print("All problems have been completed!")
            exit(0)

    print(f"Running {len(problems)} problems with model: {model}")

    for pid in problems:
        print(f"\n=== Starting problem: {pid} ===")
        agent = Agent(model=model)
        orchestrator = Orchestrator(results_dir=results_dir)
        orchestrator.register_agent(agent, name=agent_name)

        try:
            problem_desc, instructs, apis = orchestrator.init_problem(pid)
            agent.init_context(problem_desc, instructs, apis)
            asyncio.run(orchestrator.start_problem(max_steps=args.max_steps))
        except Exception as e:
            print(f"Error while running problem {pid}: {e}")

