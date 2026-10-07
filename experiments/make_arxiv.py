"""Builds arxiv_package.zip: main.tex + every \\input / \\includegraphics it reaches + main.bbl, then test-compiles it in a clean directory."""
import os, re, shutil, subprocess, sys, zipfile, tempfile
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT)
need, todo = set(), ['main.tex']
while todo:
    f = todo.pop()
    if f in need: continue
    need.add(f); t = open(f).read()
    for m in re.findall(r'\\input\{([^}]+)\}', t): todo.append(m if m.endswith('.tex') else m + '.tex')
    for m in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', t): need.add(m)
need.add('main.bbl')
out = os.path.abspath('arxiv_package.zip')
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(need): z.write(f, f)
tmp = tempfile.mkdtemp()
with zipfile.ZipFile(out) as z: z.extractall(tmp)
for _ in range(2): r = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'main.tex'], cwd=tmp, capture_output=True)
log = open(os.path.join(tmp, 'main.log'), errors='replace').read()
print(len(need), 'files;', 'compiled:', os.path.exists(os.path.join(tmp, 'main.pdf')), '| undefined refs:', len(re.findall(r'undefined', log)))
