from global_rag_formatter import format_global_rag_for_prompt
from global_rag import GlobalRAG

rag = GlobalRAG()

results = rag.retrieve("sticky navbar tailwind", k=3)

prompt_block = format_global_rag_for_prompt(results)

print(prompt_block)
