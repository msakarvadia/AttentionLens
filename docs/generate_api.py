"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

project_dir = "attention_lens"
reference_dir = "reference"

for path in sorted(Path(project_dir).rglob("**/*.py")):
    # module_path = path.relative_to(project_dir).with_suffix("")
    module_path = path.with_suffix("")
    doc_path = path.relative_to(project_dir).with_suffix(".md")
    full_doc_path = Path(reference_dir, doc_path)

    print(f"{module_path=}\n{doc_path=}\n{full_doc_path=}\n")

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    if len(parts) == 1:
        nav_parts = parts
    elif len(parts) == 2:
        nav_parts = (parts[1],)
    else:
        nav_parts = tuple([parts[1]] + [p.split(".")[-1] for p in parts[2:]])
    nav[nav_parts] = doc_path.as_posix()

    print(f"{parts=}")
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# {parts[-1]}\n\n::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open(f"{reference_dir}/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
