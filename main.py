from pathlib import Path
from pydantic_ai import Agent

model = 'openrouter:nvidia/nemotron-3-nano-30b-a3b:free'

user_prompt = f"""
Read all files in the 'workspace' folder.
Identify which one of them is responsible for training a deep learning model.
Respond with its source code.
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

result = agent.run_sync(user_prompt)
print(result.output)