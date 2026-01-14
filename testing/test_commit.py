from artifact import Artifact
from state_rag_enums import ArtifactType, ArtifactSource
from state_rag_manager import StateRAGManager

manager = StateRAGManager()

hero = Artifact(
    type=ArtifactType.component,
    name="Hero",
    file_path="components/Hero.tsx",
    content="<section>Hero</section>",
    language="tsx",
    source=ArtifactSource.ai_generated
)

manager.commit(hero)

print("Hero committed successfully")
