from typing import List

from state_rag_manager import StateRAGManager
from global_rag import GlobalRAG
from validator import Validator, ProposedArtifact
from artifact import Artifact
from state_rag_enums import ArtifactSource


class Orchestrator:
    def __init__(self):
        self.state_rag = StateRAGManager()
        self.global_rag = GlobalRAG()
        self.validator = Validator()

    def handle_request(
        self,
        user_request: str,
        allowed_paths: List[str],
    ):
        """
        End-to-end execution without LLM.
        LLM output is simulated for now.
        """

        # 1. Retrieve authoritative state
        active_artifacts = self.state_rag.retrieve(
            file_paths=allowed_paths
        )

        # 2. Retrieve advisory knowledge
        global_refs = self.global_rag.retrieve(
            query=user_request,
            k=3
        )

        # 3. Simulate LLM output (for now)
        proposed = self._mock_llm_output(allowed_paths)
        #raw = self.llm.generate(prompt)
        #proposed = parse_llm_output(raw)

        # 4. Validate proposed changes
        result = self.validator.validate(
            proposed=proposed,
            active_artifacts=active_artifacts,
            allowed_paths=allowed_paths,
        )

        if not result.ok:
            raise RuntimeError(
                f"Validation failed: {result.reason}"
            )

        # 5. Commit validated artifacts
        committed = []
        for p in result.artifacts:
            # Find existing active artifact (if any)
            old = next(
                (a for a in active_artifacts if a.file_path == p.file_path),
                None
            )

            # Preserve user authority if user explicitly allowed the change
            if old and old.source == ArtifactSource.user_modified:
                source = ArtifactSource.user_modified
            else:
                source = ArtifactSource.ai_modified

            artifact = Artifact(
                type=self._infer_type(p.file_path),
                name=p.file_path.split("/")[-1],
                file_path=p.file_path,
                content=p.content,
                language=p.language,
                source=source,
            )

            committed.append(
                self.state_rag.commit(artifact)
            )

        return committed

    # -------------------------
    # Helpers
    # -------------------------

    def _mock_llm_output(self, allowed_paths: List[str]) -> List[ProposedArtifact]:
        """
        Temporary stand-in for LLM output.
        """
        return [
            ProposedArtifact(
                file_path=allowed_paths[0],
                content="<div>Mock update</div>",
                language="tsx",
            )
        ]

    def _infer_type(self, file_path: str):
        if "components/" in file_path:
            return "component"
        if "app/" in file_path:
            return "page"
        return "config"
