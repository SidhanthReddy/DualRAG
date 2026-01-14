from orchestrator import Orchestrator
from state_rag_enums import ArtifactType

orch = Orchestrator()

prompt = orch.handle_request(
    user_request="Make the navbar sticky and reduce its height",
    scope=[ArtifactType.component],
)

print(prompt)
