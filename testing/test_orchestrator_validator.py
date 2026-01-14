from orchestrator import Orchestrator

orch = Orchestrator()

result = orch.handle_request(
    user_request="Change navbar color",
    allowed_paths=["components/Navbar.tsx"]
)

print("Committed artifacts:")
for a in result:
    print(f"- {a.file_path} v{a.version}")
