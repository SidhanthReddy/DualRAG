from llm_adapter import LLMAdapter

llm = LLMAdapter(provider="mock")

out = llm.generate("Change navbar color")

print("RAW LLM OUTPUT:\n")
print(out)
