from pydantic_ai import Agent

model = 'openrouter:nvidia/nemotron-3-nano-30b-a3b:free'

instructions = 'Be concise, reply with one sentence.'
user_prompt = 'Where does "hello world" come from?'

agent = Agent(model, instructions=instructions)

result = agent.run_sync(user_prompt)
print(result.output)