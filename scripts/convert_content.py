#!/usr/bin/env python3
"""Convert Hugo content for Astro: TOML+++ -> YAML--- frontmatter, relref -> permalinks."""
import re
import sys
from pathlib import Path

SRC = Path("src/content")


def parse_toml(text: str) -> dict:
    """Minimal TOML parser for the flat key = value frontmatter used here."""
    data = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'(\w+)\s*=\s*(.*)$', line)
        if not m:
            sys.exit(f"Unsupported frontmatter line: {line!r}")
        key, raw = m.group(1), m.group(2).strip()
        if raw.startswith("["):
            inner = raw.strip("[]").strip()
            data[key] = [] if not inner else re.findall(r'"([^"]*)"', inner)
        elif raw in ("true", "false"):
            data[key] = raw == "true"
        elif raw.startswith('"'):
            data[key] = raw[1:-1]
        else:
            data[key] = raw
    return data


def slugify(title: str) -> str:
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def yaml_str(v: str) -> str:
    if re.search(r"[:#{}&*!|>'\"%@`]", v) or "\n" in v:
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return v


def to_yaml(data: dict) -> str:
    lines = []
    for key, value in data.items():
        if isinstance(value, list):
            if not value:
                lines.append(f"{key}: []")
            else:
                lines.append(f"{key}:")
                lines.extend(f"  - {yaml_str(str(v))}" for v in value)
        else:
            lines.append(f"{key}: {yaml_str(str(value))}")
    return "\n".join(lines) + "\n"


def convert_relrefs(body: str) -> str:
    # {{< relref "posts/foo.md" >}} or "posts/foo" or "awesome-ssm" -> /posts/foo/
    def repl(m):
        target = m.group(1)
        anchor = m.group(2) or ""
        target = target.removesuffix(".md")
        if target.startswith("posts/"):
            slug = slugify(Path(target).name)
            return f"/posts/{slug}/{anchor}"
        # top-level page e.g. "awesome-ssm", "/"
        slug = slugify(Path(target).name) if target.strip("/") else ""
        return f"/{slug}/"

    return re.sub(r'\{\{<\s*relref\s+"([^"]+)"(\s+"[^"]*")?\s*>\}\}', repl, body)


for md in sorted(SRC.rglob("*.md")):
    text = md.read_text(encoding="utf-8")
    m = re.match(r"\+\+\+ *\n(.*?)\n\+\+\+ *\n?", text, re.DOTALL)
    if not m:
        continue
    try:
        data = parse_toml(m.group(1))
    except Exception as e:
        sys.exit(f"TOML parse error in {md}: {e}")

    body = text[m.end():]
    data.pop("externalLink", None)
    data.pop("slug", None)  # all empty
    if not data.get("description"):
        data.pop("description", None)
    if not data.get("draft"):
        data.pop("draft", None)
    for key in ("authors", "tags", "categories", "series"):
        if key in data and not data[key]:
            data.pop(key)

    body = convert_relrefs(body)
    md.write_text(f"---\n{to_yaml(data)}---\n\n{body}", encoding="utf-8")
    print(f"converted {md}")
print("done")
