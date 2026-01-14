from state_rag_manager import StateRAGManager
from state_rag_enums import ArtifactType

manager = StateRAGManager()

results = manager.retrieve(
    scope=[ArtifactType.component],
    limit=5
)

print("Retrieved artifacts:")
for a in results:
    print(f"- {a.file_path} (v{a.version}, {a.source})")
