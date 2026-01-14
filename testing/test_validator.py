from validator import Validator, ProposedArtifact
from artifact import Artifact
from state_rag_enums import ArtifactType, ArtifactSource

from datetime import datetime


def active_artifact(
    file_path: str,
    source: ArtifactSource,
):
    return Artifact(
        type=ArtifactType.component,
        name=file_path.split("/")[-1],
        file_path=file_path,
        content="<div />",
        language="tsx",
        version=1,
        is_active=True,
        source=source,
        dependencies=[],
        framework="react",
        styling="tailwind",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


def case(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception as e:
        print(f"[FAIL] {name}")
        raise


# =========================
# Test cases
# =========================

def test_allows_authorized_user_file_edit():
    validator = Validator()

    active = [
        active_artifact(
            "components/Navbar.tsx",
            ArtifactSource.user_modified
        )
    ]

    proposed = [
        ProposedArtifact(
            file_path="components/Navbar.tsx",
            content="<nav>updated</nav>",
            language="tsx"
        )
    ]

    result = validator.validate(
        proposed=proposed,
        active_artifacts=active,
        allowed_paths=["components/Navbar.tsx"]
    )

    assert result.ok


def test_rejects_unauthorized_user_file_edit():
    validator = Validator()

    active = [
        active_artifact(
            "components/Navbar.tsx",
            ArtifactSource.user_modified
        )
    ]

    proposed = [
        ProposedArtifact(
            file_path="components/Navbar.tsx",
            content="<nav>evil</nav>",
            language="tsx"
        )
    ]

    result = validator.validate(
        proposed=proposed,
        active_artifacts=active,
        allowed_paths=[]
    )

    assert not result.ok
    assert "Unauthorized modification" in result.reason


def test_rejects_out_of_scope_change():
    validator = Validator()

    active = [
        active_artifact(
            "components/Navbar.tsx",
            ArtifactSource.ai_generated
        )
    ]

    proposed = [
        ProposedArtifact(
            file_path="components/Navbar.tsx",
            content="<nav>oops</nav>",
            language="tsx"
        )
    ]

    result = validator.validate(
        proposed=proposed,
        active_artifacts=active,
        allowed_paths=["components/Hero.tsx"]
    )

    assert not result.ok
    assert "Out-of-scope" in result.reason


def test_rejects_empty_content():
    validator = Validator()

    proposed = [
        ProposedArtifact(
            file_path="components/Navbar.tsx",
            content="   ",
            language="tsx"
        )
    ]

    result = validator.validate(
        proposed=proposed,
        active_artifacts=[],
        allowed_paths=["components/Navbar.tsx"]
    )

    assert not result.ok
    assert "Empty content" in result.reason


def test_rejects_duplicate_outputs():
    validator = Validator()

    proposed = [
        ProposedArtifact(
            file_path="components/Navbar.tsx",
            content="<nav>1</nav>",
            language="tsx"
        ),
        ProposedArtifact(
            file_path="components/Navbar.tsx",
            content="<nav>2</nav>",
            language="tsx"
        ),
    ]

    result = validator.validate(
        proposed=proposed,
        active_artifacts=[],
        allowed_paths=["components/Navbar.tsx"]
    )

    assert not result.ok
    assert "Duplicate artifact" in result.reason


# =========================
# Run tests
# =========================

if __name__ == "__main__":
    print("=== VALIDATOR TESTS START ===")

    case("Authorized user-modified edit", test_allows_authorized_user_file_edit)
    case("Unauthorized user-modified edit", test_rejects_unauthorized_user_file_edit)
    case("Out-of-scope rejection", test_rejects_out_of_scope_change)
    case("Empty content rejection", test_rejects_empty_content)
    case("Duplicate output rejection", test_rejects_duplicate_outputs)

    print("=== VALIDATOR TESTS COMPLETE ===")
