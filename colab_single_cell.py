import os, shutil, time, subprocess, sys
from IPython.display import display, HTML, FileLink

print("=" * 65)
print("  ByteBrad's | Redrob AI Candidate Ranker | India Runs 2026")
print("=" * 65)

# ── Helper: run shell commands and stream ALL output (stdout + stderr) ──
def run(cmd, cwd='/content/india_runs'):
    proc = subprocess.Popen(
        cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()
    return proc.returncode

# STEP 1: Clone repo
os.chdir('/content')
if not os.path.exists('/content/india_runs'):
    run('git clone https://github.com/PardheevKomakula/Hack2skill-x-redrob-The-Data-AI-Challenge.git india_runs', cwd='/content')
    print("✅ Repo cloned.")
else:
    print("✅ Repo already present.")
os.chdir('/content/india_runs')

# STEP 2: Fix dependency conflicts
# peft MUST be uninstalled first — peft 0.19.1 breaks transformers < 4.43
# Do NOT pin numpy — Colab pre-built packages require numpy 2.x
print("\n⏳ Fixing dependencies (this takes ~1 min)...")
run('pip uninstall -y peft sentence-transformers transformers tokenizers accelerate -q')
run('pip install sentence-transformers==3.0.1 transformers==4.44.2 tokenizers==0.19.1 pyyaml -q')
print("✅ Dependencies ready.")

# STEP 3: Auto-detect GPU and patch config
import yaml, torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🖥️  Device: {device.upper()} {'(T4 GPU — 🚀 precompute ~2 min)' if device == 'cuda' else '(CPU — precompute ~7 min)'}")
with open('config/ranking_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['model']['device'] = device
with open('config/ranking_config.yaml', 'w') as f:
    yaml.dump(cfg, f)
print(f"   Config patched → device: {device}")

# STEP 4: Move uploaded files
print("\n📁 Checking uploaded files...")
os.makedirs('data', exist_ok=True)
all_good = True
for fname in ['candidates.json', 'job_description.txt']:
    src = f'/content/{fname}'
    dst = f'/content/india_runs/{fname}'
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
        print(f"   ✅ Moved {fname}")
    elif os.path.exists(dst):
        print(f"   ✅ {fname} in place")
    else:
        print(f"   ⚠️  {fname} NOT FOUND — upload via 📁 panel then re-run")
        all_good = False

if not all_good:
    raise FileNotFoundError("Missing required files. See warnings above.")

# STEP 5: Precompute embeddings (cached — skips if already done this session)
print("\n🧠 Checking embeddings cache...")
if os.path.exists('data/candidate_embeddings.npy') and os.path.exists('data/candidate_ids.npy'):
    print("   ✅ Cache found — skipping (~2-7 min saved).")
else:
    print("   ⏳ Precomputing embeddings for 100K candidates...")
    t_pre = time.time()
    rc = run('python precompute_embeddings.py --candidates ./candidates.json')
    if rc != 0:
        raise RuntimeError(f"precompute_embeddings.py failed (exit code {rc}). See error above.")
    print(f"   ✅ Done in {time.time()-t_pre:.0f}s")

# STEP 6: Run full production ranking pipeline
print("\n🚀 Running Production Ranking Pipeline...")
print("-" * 65)
t0 = time.time()
rc = run('python rank.py --candidates ./candidates.json --jd ./job_description.txt --out ./submission.csv')
elapsed = time.time() - t0
print("-" * 65)
if rc != 0:
    raise RuntimeError(f"rank.py failed (exit code {rc}). See error output above.")
print(f"⏱  Pipeline completed in {elapsed:.1f}s")

# STEP 7: Verify and display results
print()
if not os.path.exists('submission.csv'):
    raise FileNotFoundError("submission.csv was not created. Check rank.py errors above.")

import pandas as pd
df = pd.read_csv('submission.csv')
print(f"✅ {len(df)} candidates ranked | Score: {df['score'].min():.4f} → {df['score'].max():.4f}\n")

print("🏆 TOP 10 RANKED CANDIDATES")
display(HTML(
    df.head(10)[['rank', 'candidate_id', 'score', 'reasoning']]
    .to_html(index=False, border=0)
    .replace('<table', '<table style="width:100%;border-collapse:collapse;font-size:13px"')
    .replace('<th>', '<th style="background:#1a1a2e;color:white;padding:8px;text-align:left">')
    .replace('<td>', '<td style="padding:7px;border-bottom:1px solid #ddd">')
))

print("\n📊 SCORE DISTRIBUTION (Top 100)")
for label, mask in [
    ("  Elite   0.80+",     df['score'] >= 0.80),
    ("  Strong  0.60–0.80", (df['score'] >= 0.60) & (df['score'] < 0.80)),
    ("  Good    0.40–0.60", (df['score'] >= 0.40) & (df['score'] < 0.60)),
    ("  Marginal <0.40",    df['score'] < 0.40),
]:
    count = int(mask.sum())
    print(f"{label}: {count:3d}  {'█' * count}")

print("\n📥 Download your submission:")
display(FileLink('submission.csv', result_html_prefix="👉 "))
print("\n" + "=" * 65)
print("  Pipeline complete. Good luck, ByteBrad's! 🏆")
print("=" * 65)
