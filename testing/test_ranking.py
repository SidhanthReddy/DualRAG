from state_rag_manager import StateRAGManager
from state_rag_enums import ArtifactType

manager = StateRAGManager()

results = manager.retrieve(
    scope=[ArtifactType.component],
    user_query="navbar sticky height",
)

print("Ranked artifacts:")
for a in results:
    print("-", a.file_path)
