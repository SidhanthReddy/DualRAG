from llm_output_parser import parse_llm_output

raw = """
FILE: components/Navbar.tsx
<nav class='h-20 bg-blue-600 text-white'>
  Test Navbar
</nav>

FILE: app/page.tsx
<div>
  Home Page
</div>
"""

artifacts = parse_llm_output(raw)

print("Parsed artifacts:")
for a in artifacts:
    print(f"- {a.file_path} ({a.language})")
    print(a.content)
    print("-----")
