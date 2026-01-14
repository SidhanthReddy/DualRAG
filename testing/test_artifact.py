from artifact import Artifact
from state_rag_enums import ArtifactType, ArtifactSource

a = Artifact(
    type=ArtifactType.component,
    name="Navbar",
    file_path="components/Navbar.tsx",
    content="<nav>Test</nav>",
    language="tsx",
    source=ArtifactSource.user_modified
)

print(a)
