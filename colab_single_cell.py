import os, shutil, time
from IPython.display import display, HTML, FileLink

print("=" * 65)
print("  ByteBrad's | Redrob AI Candidate Ranker | India Runs 2026")
print("=" * 65)

# STEP 1: Fresh clone (always get latest from GitHub)
os.chdir('/content')
if os.path.exists('/content/india_runs'):
    shutil.rmtree('/content/india_runs')
os.system('git clone https://github.com/PardheevKomakula/Hack2skill-x-redrob-The-Data-AI-Challenge.git india_runs')
print("✅ Repo cloned (latest).")
os.chdir('/content/india_runs')

# STEP 2: Fix dependencies
print("\n⏳ Installing dependencies...")
os.system('pip uninstall -y peft sentence-transformers transformers tokenizers accelerate -q 2>/dev/null')
os.system('pip install sentence-transformers==3.0.1 transformers==4.44.2 tokenizers==0.19.1 pyyaml -q')
print("✅ Dependencies ready.")

# STEP 3: Auto-detect GPU and patch config
import yaml, torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🖥️  Device: {device.upper()} {'(T4 GPU 🚀)' if device == 'cuda' else '(CPU)'}")
with open('config/ranking_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['model']['device'] = device
with open('config/ranking_config.yaml', 'w') as f:
    yaml.dump(cfg, f)

# STEP 4: Move uploaded files
print("\n📁 Checking uploaded files...")
os.makedirs('data', exist_ok=True)
for fname in ['candidates.json', 'job_description.txt']:
    src, dst = f'/content/{fname}', f'/content/india_runs/{fname}'
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
        print(f"   ✅ Moved {fname}")
    elif os.path.exists(dst):
        print(f"   ✅ {fname} in place")
    else:
        print(f"   ⚠️  {fname} NOT FOUND — upload via 📁 panel then re-run")

assert os.path.exists('candidates.json'), "candidates.json missing!"
assert os.path.exists('job_description.txt'), "job_description.txt missing!"

# STEP 5: Precompute embeddings (cached)
print("\n🧠 Precomputing embeddings...")
if os.path.exists('data/candidate_embeddings.npy') and os.path.exists('data/candidate_ids.npy'):
    print("   ✅ Cache found — skipping.")
else:
    t_pre = time.time()
    rc = os.system('python precompute_embeddings.py --candidates ./candidates.json')
    assert rc == 0, f"precompute failed (exit {rc})"
    print(f"   ✅ Done in {time.time()-t_pre:.0f}s")

# STEP 6: Run ranking
print("\n🚀 Running Production Ranking Pipeline...")
t0 = time.time()
rc = os.system('python rank.py --candidates ./candidates.json --jd ./job_description.txt --out ./submission.csv')
assert rc == 0, f"rank.py failed (exit {rc})"
print(f"⏱  Pipeline completed in {time.time()-t0:.1f}s")

# STEP 7: Display results
assert os.path.exists('submission.csv'), "submission.csv not created!"
import pandas as pd
df = pd.read_csv('submission.csv')
print(f"\n✅ {len(df)} candidates ranked | Score: {df['score'].min():.4f} → {df['score'].max():.4f}\n")

print("🏆 TOP 10 RANKED CANDIDATES")
display(HTML(
    df.head(10)[['rank', 'candidate_id', 'score', 'reasoning']]
    .to_html(index=False, border=0)
    .replace('<table', '<table style="width:100%;border-collapse:collapse;font-size:13px"')
    .replace('<th>', '<th style="background:#1a1a2e;color:white;padding:8px;text-align:left">')
    .replace('<td>', '<td style="padding:7px;border-bottom:1px solid #ddd">')
))

print("\n📊 SCORE DISTRIBUTION")
for label, mask in [
    ("  Elite   0.80+", df['score'] >= 0.80),
    ("  Strong  0.60–0.80", (df['score'] >= 0.60) & (df['score'] < 0.80)),
    ("  Good    0.40–0.60", (df['score'] >= 0.40) & (df['score'] < 0.60)),
    ("  Marginal <0.40", df['score'] < 0.40),
]:
    print(f"{label}: {int(mask.sum()):3d}  {'█' * int(mask.sum())}")

print("\n📥 Download:")
display(FileLink('submission.csv', result_html_prefix="👉 "))
print("\n" + "=" * 65)
print("  Pipeline complete. Good luck, ByteBrad's! 🏆")
print("=" * 65)
