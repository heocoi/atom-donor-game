"""Post-process atom_hybrid.tex after pandoc conversion.

Operations:
1. Make References section unnumbered (\section* not \section)
2. Insert \appendix before Reproducibility (so appendices use A, B, C)
3. Inject newunicodechar declarations to handle remaining Unicode
   chars in code blocks (works for both pdflatex and xelatex)
"""

with open('atom_hybrid.tex') as f:
    tex = f.read()

# 1. Make References unnumbered
tex = tex.replace(
    r'\section{References}',
    r'\section*{References}\addcontentsline{toc}{section}{References}'
)

# 2. Insert \appendix before Reproducibility
tex = tex.replace(
    r'\section{Reproducibility}',
    '\\appendix\n\\section{Reproducibility}'
)

# 3. Fix Hyperparameters table column widths
# pandoc auto-computes widths that put Symbol column too narrow for long names like
# S2_CONFIDENCE_THRESHOLD. Override widths for the table that contains DONATE_COST.
import re
def fix_hyperparams_table(text):
    # Find the longtable that contains DONATE_COST and rewrite its column widths
    # Pattern: \begin{longtable}[]{@{} ... \real{0.2857} ... DONATE\_COST
    pattern = re.compile(
        r'(\\begin\{longtable\}\[\]\{@\{\}\s*\n'
        r'\s*>\{\\raggedright\\arraybackslash\}p\{\(\\linewidth - 4\\tabcolsep\) \* \\real\{)'
        r'0\.\d+'  # symbol column width
        r'(\}\}\s*\n'
        r'\s*>\{\\raggedright\\arraybackslash\}p\{\(\\linewidth - 4\\tabcolsep\) \* \\real\{)'
        r'0\.\d+'  # value column width
        r'(\}\}\s*\n'
        r'\s*>\{\\raggedright\\arraybackslash\}p\{\(\\linewidth - 4\\tabcolsep\) \* \\real\{)'
        r'0\.\d+'  # description column width
        r'(\}\}@\{\}\}\s*'
        r'\\toprule[\s\S]*?DONATE\\_COST)',
        re.MULTILINE
    )
    def repl(m):
        return m.group(1) + '0.42' + m.group(2) + '0.10' + m.group(3) + '0.48' + m.group(4)
    new = pattern.sub(repl, text, count=1)
    if new == text:
        print('  WARNING: hyperparams table not matched - column widths unchanged')
    return new
tex = fix_hyperparams_table(tex)

# 4. Inject Unicode handling preamble before \begin{document}
unicode_decl = r"""
% --- Custom Unicode handling (works for pdflatex + xelatex) ---
\IfFileExists{newunicodechar.sty}{%
  \usepackage{newunicodechar}
  \newunicodechar{σ}{\ensuremath{\sigma}}
  \newunicodechar{Σ}{\ensuremath{\Sigma}}
  \newunicodechar{δ}{\ensuremath{\delta}}
  \newunicodechar{α}{\ensuremath{\alpha}}
  \newunicodechar{τ}{\ensuremath{\tau}}
  \newunicodechar{≥}{\ensuremath{\geq}}
  \newunicodechar{≤}{\ensuremath{\leq}}
  \newunicodechar{∈}{\ensuremath{\in}}
  \newunicodechar{≈}{\ensuremath{\approx}}
  \newunicodechar{→}{\ensuremath{\rightarrow}}
  \newunicodechar{←}{\ensuremath{\leftarrow}}
  \newunicodechar{·}{\ensuremath{\cdot}}
  \newunicodechar{×}{\ensuremath{\times}}
  \newunicodechar{±}{\ensuremath{\pm}}
  \newunicodechar{ᵢ}{\textsubscript{i}}
  \newunicodechar{₀}{\textsubscript{0}}
  \newunicodechar{₁}{\textsubscript{1}}
  \newunicodechar{₂}{\textsubscript{2}}
  \newunicodechar{₃}{\textsubscript{3}}
}{}
% --- end custom Unicode ---

"""
tex = tex.replace(
    r'\begin{document}',
    unicode_decl + r'\begin{document}',
    1
)

with open('atom_hybrid.tex', 'w') as f:
    f.write(tex)

print('Post-processing complete')
print('  - References: \\section* (unnumbered)')
print('  - Appendix: \\appendix inserted before Reproducibility')
print('  - Unicode: newunicodechar declarations added')
