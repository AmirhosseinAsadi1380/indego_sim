"""Generate docs/PARAMETERS.md from the dataclasses (defaults + inline provenance comments) in config.py."""
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "indego_sim", "config.py")
OUT = os.path.join(HERE, "..", "docs", "PARAMETERS.md")

lines = open(SRC).read().splitlines()
tree = ast.parse("\n".join(lines))
md = ["# Parameters and provenance", "",
      "Generated from `indego_sim/config.py` by `tools/gen_parameters_md.py`. Tags: **[doc]** from Indego",
      "documentation / papers on the Indego, **[lit]** from the biomechanics / SCI literature, **[est]** an",
      "engineering estimate that you should measure, **[model]** a modelling choice (tuned for stability or",
      "behaviour, not a physical constant), **[calib]** set by a calibration in this package.", "",
      "Anything tagged [est] or [model] is a candidate for the calibration procedures in `CALIBRATION.md`.", ""]
for node in tree.body:
    if isinstance(node, ast.ClassDef):
        doc = ast.get_docstring(node) or ""
        md += [f"## `{node.name}`", ""]
        if doc:
            md += [doc.replace("\n", " "), ""]
        md += ["| parameter | default | notes |", "|---|---|---|"]
        for st in node.body:
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                name = st.target.id
                src = lines[st.lineno - 1]
                m = re.search(r"#\s*(.*)$", src)
                note = m.group(1) if m else ""
                # continuation comment lines
                k = st.end_lineno
                # (deeply indented = continuation of this field's note; field-level comments start a new topic)
                while k < len(lines) and re.match(r"^\s{10,}#\s", lines[k]) and not re.match(r"^\s+#\s*---", lines[k]):
                    note += " " + lines[k].strip().lstrip("#").strip()
                    k += 1
                val = ast.get_source_segment("\n".join(lines), st.value) if st.value is not None else ""
                val = val.replace("field(default_factory=", "").rstrip(")") if val.startswith("field(") else val
                val = val.replace("|", "\\|")
                if len(val) > 60:
                    val = val[:57] + "..."
                md.append(f"| `{name}` | `{val}` | {note.replace('|', '/')} |")
        md.append("")
open(OUT, "w", newline="\n").write("\n".join(md) + "\n")      # LF on every OS
print("wrote", OUT)
