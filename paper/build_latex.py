"""Transform atom_hybrid.md → atom_hybrid_latex.md (LaTeX-ready).

Operations:
1. Move Abstract section content into YAML frontmatter
2. Strip explicit section numbers (LaTeX auto-numbers)
3. Shift heading levels: ## -> #, ### -> ##
4. Remove --- horizontal rules between sections (LaTeX has built-in spacing)
5. Replace Unicode math/Greek with LaTeX commands (pdflatex-safe)
6. Mark References as unnumbered, insert \\appendix before appendix sections
"""

# Unicode chars are kept as-is in markdown.
# LaTeX preamble (added by postprocess_tex.py) maps them via newunicodechar,
# which is more robust than pandoc's $...$ math handling.
import re
import sys

src = open('atom_hybrid.md').read()

# Split YAML frontmatter
m = re.match(r'^---\n(.*?)\n---\n(.*)$', src, re.DOTALL)
assert m, 'no YAML frontmatter found'
yaml_body, body = m.group(1), m.group(2)

# Extract the Abstract section text
m_abs = re.search(r'^## Abstract\s*\n+(.*?)(?=^##|\Z)', body, re.DOTALL | re.MULTILINE)
assert m_abs, 'Abstract section not found'
abstract_text = m_abs.group(1).strip()
# Strip trailing --- separator if present
abstract_text = re.sub(r'\n*---\s*$', '', abstract_text).strip()

# Remove the Abstract section from body
body = body[:m_abs.start()] + body[m_abs.end():]

# Strip explicit section numbers
body = re.sub(r'^## \d+\. ', '## ', body, flags=re.MULTILINE)
body = re.sub(r'^### \d+\.\d+ ', '### ', body, flags=re.MULTILINE)

# Strip explicit Appendix labels (will use LaTeX appendix env)
body = re.sub(r'^## Appendix [A-Z]: ', '## ', body, flags=re.MULTILINE)

# Shift heading levels (### -> ##, ## -> #) using sentinel approach
body = re.sub(r'^### ', '\x00\x00 ', body, flags=re.MULTILINE)  # ### -> sentinel
body = re.sub(r'^## ', '# ', body, flags=re.MULTILINE)             # ## -> #
body = re.sub(r'^\x00\x00 ', '## ', body, flags=re.MULTILINE)      # sentinel -> ##

# Remove standalone --- horizontal rules (but not the YAML closing ---)
body = re.sub(r'^---\s*$\n', '', body, flags=re.MULTILINE)

# Build new YAML with abstract included
yaml_indented = '\n'.join('  ' + line for line in abstract_text.split('\n'))
new_yaml = yaml_body.rstrip() + '\nabstract: |\n' + yaml_indented

result = '---\n' + new_yaml + '\n---\n\n' + body.lstrip()

with open('atom_hybrid_latex.md', 'w') as f:
    f.write(result)

print(f'Written: atom_hybrid_latex.md ({len(result)} bytes)')
