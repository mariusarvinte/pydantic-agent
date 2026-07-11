import asyncio
import json

from pathlib import Path
from pydantic_ai import Agent

model = "openrouter:nvidia/nemotron-3-nano-30b-a3b:free"

user_prompt = f"""
Read all files in the 'workspace' folder.
Identify which one of them is responsible for training a deep learning model.
Modify it **only once** to improve training speed without losing numerical precision.
"""

agent = Agent(model)


@agent.tool_plain
def get_files_in_folder(path: Path) -> str:
    files_in_folder = path.glob("*")

    result = "\n".join([str(file) for file in files_in_folder])
    return result


@agent.tool_plain
def read_file(path: Path) -> str:
    return path.read_text()


@agent.tool_plain
def write_file(path: Path, contents: str) -> str:
    """
    Writes 'contents' to the file at 'path'.
    Creates the parent directory of 'path' if required.
    """

    # Create parent directory if required
    path.parent.mkdir(exist_ok=True)

    # Write contents to file
    path.write_text(contents)
    return "Success!"


@agent.tool_plain
def edit_file(path: Path, current: str, new: str) -> str:
    """
    Edits the file at 'path' by replacing all occurences of 'current' with 'new'.
    """

    if not path.is_file():
        return "The file does not exist!"

    contents = path.read_text()
    if current not in contents:
        return "The file does not contain any occurence of 'current'!"

    path.write_text(contents.replace(current, new))
    return "Success!"


async def run_agent():
    captured_nodes = []
    async with agent.iter(user_prompt) as agent_run:
        async for node in agent_run:
            # Each node represents a step in the agent's execution
            captured_nodes.append(node)

    # Save agent trajectory to JSON
    json_bytes = agent_run.all_messages_json()
    json_data = json.loads(json_bytes)
    with open("agent_run.json", "w") as f:
        json.dump(json_data, f, indent=2)

    # Print final result to console
    if agent_run.result:
        print(agent_run.result.output)
    else:
        print("Agent did not finish its task!")


if __name__ == "__main__":
    asyncio.run(run_agent())
