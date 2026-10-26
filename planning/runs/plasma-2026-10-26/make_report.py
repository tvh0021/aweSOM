"""Assemble the self-contained HTML results page for the Metal SCE test and the plasma retraining."""
import base64, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "reports" / "2026-10-26-plasma-metal-results.html"

summ = json.load(open(HERE / "summary_thr025.json"))
agree = json.load(open(HERE / "agreement_thr025.json"))
sweep = json.load(open(HERE / "seed_sweep.json"))
binfo = json.load(open(HERE / "batch_info.json"))
timing = {}
for ln in open(HERE / "sce_timing.txt"):
    m = re.match(r"(\w+) wall_s=([\d.]+)", ln)
    if m:
        timing[m.group(1)] = float(m.group(2))

def img(name):
    b = base64.b64encode(open(HERE / name, "rb").read()).decode()
    return f"data:image/png;base64,{b}"

FEATS = ["j_par", "j_asym", "b_perp", "e_par"]
PAPER = [5.4, 1.7, 1.1, 2.5, 3.3, 1.5, 84.5]

def cluster_rows(tag):
    rows = sorted(summ[tag]["rows"], key=lambda r: -r["fraction"])
    out = []
    for r in rows:
        m = r["mean"]
        out.append("<tr><td>c%d</td><td>%.1f</td>%s</tr>" % (r["cluster"], r["fraction"] * 100, "".join(f"<td>{v:+.2f}</td>" for v in m)))
    return "\n".join(out)

def role(m):
    jp, ja, bp, ep = m
    if abs(ep) > 0.9:
        return "parallel E, " + ("positive" if ep > 0 else "negative")
    if max(abs(jp), abs(ja), abs(bp)) < 0.3:
        return "background"
    parts = []
    if abs(jp) > 0.5:
        parts.append("j_par " + ("+" if jp > 0 else "−"))
    if abs(ja) > 0.9:
        parts.append("j_asym " + ("+" if ja > 0 else "−"))
    if bp > 0.5:
        parts.append("high b_perp")
    return ", ".join(parts) or "mixed"

def cluster_rows_role(tag):
    rows = sorted(summ[tag]["rows"], key=lambda r: -r["fraction"])
    return "\n".join("<tr><td>c%d</td><td>%.1f</td>%s<td>%s</td></tr>" % (r["cluster"], r["fraction"] * 100, "".join(f"<td>{v:+.2f}</td>" for v in r["mean"]), role(r["mean"])) for r in rows)

fid, onl, bat = summ["fiducial"], summ["online"], summ["batch"]
bg = lambda t: max(r["fraction"] for r in summ[t]["rows"]) * 100

sweep_rows = []
for k, v in sweep.items():
    kind, rest = k.split("_", 1)
    sweep_rows.append(f"<tr><td>{kind}</td><td>{rest.replace('seed', 'seed ').replace('_ep', ', ')}{' epochs' if '_ep' in rest else ''}</td><td>{v['train_s']:.0f}</td><td>{v['n_clusters']}</td><td>{v['background']*100:.1f}</td><td>{v['sheetlike']*100:.1f}</td><td>{v['qe']:.4f}</td><td>{v['te']:.4f}</td></tr>")
sweep_rows = "\n".join(sweep_rows)

cpu_t, metal_t = timing.get("cpu"), timing.get("metal")
clean_row = f"<tr><td>Full ensemble, idle machine (this re-run)</td><td>28</td><td>{cpu_t:.0f}</td><td>{metal_t:.0f}</td><td>{cpu_t/metal_t:.2f}×</td></tr>" if cpu_t and metal_t else "<tr><td>Full ensemble, idle machine</td><td>28</td><td colspan=3>not recorded</td></tr>"

html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>aweSOM on Apple Metal and a retrain of the plasma-turbulence fiducial map</title>
  <style>
    :root {{
      --paper: #f6f4ee; --surface: #fffdfa; --ink: #1f2a35; --muted: #5d6974; --line: #cbd4da;
      --accent: #245f91; --teal: #27766e; --evidence: #eaf2f8; --uncertainty: #fff4d6; --limitation: #f6e9e7; --measure: 72ch;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; background: var(--paper); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 1rem; line-height: 1.65; }}
    a {{ color: var(--accent); text-underline-offset: 0.16em; }}
    :focus-visible {{ outline: 3px solid #d68d17; outline-offset: 3px; }}
    .skip-link {{ position: fixed; left: 1rem; top: -5rem; z-index: 10; padding: .65rem .9rem; background: var(--ink); color: white; }}
    .skip-link:focus {{ top: 1rem; }}
    header, main, footer {{ width: min(calc(100% - 2rem), 78rem); margin-inline: auto; }}
    header {{ padding: clamp(3rem, 9vw, 6rem) 0 2rem; border-bottom: 1px solid var(--line); }}
    .eyebrow {{ color: var(--teal); font-size: .78rem; font-weight: 750; letter-spacing: .1em; text-transform: uppercase; }}
    h1, h2, h3 {{ line-height: 1.15; text-wrap: balance; }}
    h1 {{ max-width: 26ch; margin: .4rem 0 1rem; font-size: clamp(1.9rem, 5vw, 3.2rem); letter-spacing: -.03em; }}
    h2 {{ margin-top: 3rem; font-size: clamp(1.45rem, 3vw, 2rem); }}
    h3 {{ margin-top: 2rem; font-size: 1.15rem; }}
    .lede {{ max-width: 62ch; color: var(--muted); font-size: clamp(1.05rem, 2vw, 1.2rem); }}
    .meta {{ display: flex; flex-wrap: wrap; gap: .5rem 1.5rem; margin-top: 1.5rem; color: var(--muted); font-size: .9rem; }}
    main {{ padding: 1rem 0 5rem; }}
    section {{ max-width: var(--measure); }}
    p, li {{ text-wrap: pretty; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr)); gap: 1rem; max-width: none; margin: 2rem 0; }}
    .card {{ padding: 1.1rem 1.2rem; background: var(--surface); border: 1px solid var(--line); border-radius: .6rem; }}
    .card > :first-child {{ margin-top: 0; }}
    .callout {{ margin: 1.5rem 0; padding: 1rem 1.15rem; border-left: .32rem solid var(--accent); border-radius: .25rem; }}
    .evidence {{ background: var(--evidence); }}
    .uncertainty {{ background: var(--uncertainty); border-left-color: #a56b00; }}
    .limitation {{ background: var(--limitation); border-left-color: #9a5048; }}
    .table-wrap {{ max-width: 100%; overflow-x: auto; border: 1px solid var(--line); border-radius: .4rem; margin: 1rem 0; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--surface); font-variant-numeric: tabular-nums; }}
    th, td {{ padding: .55rem .75rem; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: #edf1f3; }}
    caption {{ color: var(--muted); font-size: .9rem; text-align: left; padding: .5rem .8rem; caption-side: bottom; }}
    figure {{ margin: 1.5rem 0; max-width: none; }}
    figure img {{ width: 100%; height: auto; border: 1px solid var(--line); border-radius: .4rem; background: white; }}
    figcaption {{ color: var(--muted); font-size: .9rem; margin-top: .5rem; max-width: var(--measure); }}
    .wide {{ max-width: none; }}
    code, kbd {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .92em; }}
    pre {{ overflow-x: auto; padding: 1rem; background: #18232c; color: #eef4f7; border-radius: .4rem; }}
    pre code {{ font-size: .85em; }}
    details {{ margin: 1rem 0; padding: .8rem 1rem; background: var(--surface); border: 1px solid var(--line); border-radius: .4rem; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    .ref {{ padding-left: 1.5rem; text-indent: -1.5rem; margin: .6rem 0; }}
    footer {{ padding: 1.5rem 0 3rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .88rem; }}
    @media (max-width: 42rem) {{
      header, main, footer {{ width: min(calc(100% - 1.25rem), 78rem); }}
      header {{ padding-top: 2.5rem; }}
      th, td {{ padding: .45rem; }}
    }}
    @media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior: auto; }} }}
  </style>
</head>
<body>
  <a class="skip-link" href="#main">Skip to content</a>
  <header>
    <div class="eyebrow">Results note, autonomous session, 2026-10-26</div>
    <h1>aweSOM on Apple Metal, and a retrain of the plasma-turbulence fiducial map</h1>
    <p class="lede">The statistically combined ensemble (SCE) step runs on the MacBook GPU through jax-metal and reproduces the CPU result to float32 precision. Retraining the 2,097,152-cell plasma-turbulence data set with the online and the new batch trainer gives maps of the same quality as the published fiducial map, but the merged cluster set at the paper's threshold is not stable from seed to seed.</p>
    <div class="meta"><span>Checkout <code>/Users/tvh0021/git_repos/aweSOM-research</code>, branch <code>research/grmhd-scale</code></span><span>Hardware: Apple M1 Max, 64 GB, macOS 27.0.1</span><span>Scratch data: <code>planning/runs/plasma-2026-10-26/</code> (git-ignored)</span></div>
  </header>

  <main id="main">

    <section aria-labelledby="summary-heading">
      <h2 id="summary-heading">Summary</h2>
      <div class="summary-grid">
        <div class="card"><h3>Metal SCE</h3><p>Works out of the box with jax-metal 0.1.1 (jax 0.4.34). On the full 28-map ensemble the Gsum table agrees with the NumPy run to a maximum relative difference of 5.5 × 10<sup>−7</sup> with identical ranking; masks differ by at most 1.0 × 10<sup>−4</sup>. See <a href="#metal-heading">section 3</a>.</p></div>
        <div class="card"><h3>Paper fiducial reproduced</h3><p>The shipped pickle, re-clustered with a merge threshold of 0.25 as in Appendix A of Ha et al. (2024), gives the paper's seven clusters with volume fractions 5.4, 1.7, 1.1, 2.6, 3.3, 1.5 and 84.5 %. See <a href="#fiducial-heading">section 1</a>.</p></div>
        <div class="card"><h3>Retrains</h3><p>Online trainer: 31 s, quantization error {onl['qe']:.4f} versus {fid['qe']:.4f} for the fiducial. Batch trainer (40 epochs, CPU JAX): 156 s, quantization error {bat['qe']:.4f}, topographic error {bat['te']:.4f}. Both recover the background at 85.0 %, but merge into 4 and 5 clusters instead of 7. See <a href="#retrain-heading">section 2</a>.</p></div>
        <div class="card"><h3>Batch trainer on Metal</h3><p>Not possible: jax-metal has no float64 and no <code>jax.shard_map</code>. The trainer now raises a clear <code>ImportError</code> under that JAX instead of an <code>AttributeError</code>. See <a href="#limits-heading">section 5</a>.</p></div>
      </div>
      <aside class="callout uncertainty" aria-label="Uncertainty"><strong>Uncertainty.</strong> Timings in Table 6 marked "contended" were measured while another training job was running on the same CPU cores; the idle-machine row is the one to quote. Seed-to-seed cluster counts in Table 4 come from one run per seed, so the spread is indicative only.</aside>
    </section>

    <section aria-labelledby="fiducial-heading">
      <h2 id="fiducial-heading">1. The fiducial result in the paper, and what the shipped pickle contains</h2>
      <p>The README links the method paper, Ha et al. (2024), <a href="https://arxiv.org/abs/2410.01878">arXiv:2410.01878</a>. Its fiducial self-organizing map for the plasma-turbulence snapshot has 64 × 37 nodes, learning rate α<sub>0</sub> = 0.1, N = 128<sup>3</sup> training steps (one pass over every cell), uniform random initialization and a final neighbourhood radius of 8 (Section 3.1 and Appendix A). The U-matrix is partitioned into seven clusters by merging centroids whose boundary cost is below 0.25 of the cost between the two most distant centroids (Appendix A). Section 4.1 reports the volume-filling fractions as 5.4, 1.7, 1.1, 2.5, 3.3, 1.5 and 84.5 % for clusters 0 to 6, describes clusters 0 and 4 as isolated single current sheets, 1 and 2 as components of double current sheets, 6 as background, and gives a sheet-like total of about 11.5 %. The SCE over 36 realizations is thresholded at ΣGsum of 0.25, 0.5 and 0.8 (Section 4.2 and Figure 3). No quantization or topographic error is reported.</p>
      <p>The repository ships that fiducial map as <code>examples/plasma-turbulence/som_object.fiducial-5000-64-37-0.1-2097152-1u.pkl</code> with the four features <code>j_par</code>, <code>j_asym</code>, <code>b_perp</code>, <code>e_par</code>, standardized to zero mean and unit variance. Re-clustering its lattice with <code>assign_cluster_to_lattice(smoothing=None, merge_cost=0.25)</code> reproduces the paper's result exactly, including the cluster numbering (Table 1). Three details did not match the text:</p>
      <ul>
        <li><strong>Stored clusters differ from the paper.</strong> The <code>clusters</code> array inside the pickle has seven clusters with fractions 6.5, 1.7, 1.1, 1.5, 69.8, 17.9 and 1.5 %, which no threshold among 0.2, 0.25 and 0.3 reproduces. It was evidently produced with a different setting (or smoothing) than the published figure.</li>
        <li><strong>The example script uses 0.2.</strong> <code>examples/plasma-turbulence/run_plasma_som.py</code> hard-codes <code>merge_threshold = 0.2</code>, which yields nine clusters on this lattice, not the seven in the paper.</li>
        <li><strong>Initial weight range.</strong> Appendix B says initial weights are drawn uniformly in −1 ≤ w<sub>0</sub> ≤ 1; <code>Lattice</code> draws them in [0, 1), and the first history snapshot of the fiducial run confirms [0, 1). The paper also notes that all initializations converge to similar maps at this N, so this is a documentation discrepancy rather than a result-changing one.</li>
      </ul>
      <div class="table-wrap">
        <table id="tab-fiducial">
          <thead><tr><th scope="col">Cluster</th><th scope="col">Paper %</th><th scope="col">Pickle at 0.25 %</th><th scope="col">j_par</th><th scope="col">j_asym</th><th scope="col">b_perp</th><th scope="col">e_par</th><th scope="col">Reading</th></tr></thead>
          <tbody>
            {"".join("<tr><td>c%d</td><td>%.1f</td><td>%.1f</td>%s<td>%s</td></tr>" % (r["cluster"], PAPER[r["cluster"]], r["fraction"]*100, "".join(f"<td>{v:+.2f}</td>" for v in r["mean"]), role(r["mean"])) for r in fid["rows"])}
          </tbody>
          <caption>Table 1. Paper fractions (Ha et al. 2024, Section 4.1) against the shipped lattice re-clustered at merge threshold 0.25, with the mean standardized feature value of the cells in each cluster. The only difference is cluster 3 (2.5 versus 2.6 %), consistent with rounding. Quantization error of the fiducial lattice on all cells is {fid['qe']:.4f}; topographic error is {fid['te']:.4f}.</caption>
        </table>
      </div>
    </section>

    <section aria-labelledby="retrain-heading">
      <h2 id="retrain-heading">2. Retraining the same data</h2>
      <p>Both retrains used the pickle's own standardized <code>data_array</code> and the fiducial hyper-parameters: 64 × 37 nodes, α<sub>0</sub> = 0.1, 2,097,152 steps, uniform initialization, seed 42. The online run is <code>Lattice.train_lattice(history=False)</code> with the compiled step kernel from this branch. The batch run is <code>Lattice.train_lattice_batch(epochs=40)</code>, the JAX batch map added on this branch, executed on the CPU JAX backend (jax 0.12) with one device. Lattices were then clustered at 0.25 and every cell labelled with <code>Lattice.predict</code>.</p>
      <div class="table-wrap">
        <table id="tab-compare">
          <thead><tr><th scope="col">Map</th><th scope="col">Train time (s)</th><th scope="col">QE</th><th scope="col">TE</th><th scope="col">Clusters at 0.25</th><th scope="col">Background %</th><th scope="col">Sheet-like %</th><th scope="col">Jaccard vs fiducial (non-background)</th></tr></thead>
          <tbody>
            <tr><td>Fiducial (paper pickle)</td><td>not timed (pre-existing)</td><td>{fid['qe']:.4f}</td><td>{fid['te']:.4f}</td><td>{fid['n_clusters']}</td><td>{bg('fiducial'):.1f}</td><td>{100-bg('fiducial'):.1f}</td><td>1 (reference)</td></tr>
            <tr><td>Online retrain, seed 42</td><td>{onl['train_s']:.1f}</td><td>{onl['qe']:.4f}</td><td>{onl['te']:.4f}</td><td>{onl['n_clusters']}</td><td>{bg('online'):.1f}</td><td>{100-bg('online'):.1f}</td><td>{agree['online']['jaccard_nonbackground']:.3f}</td></tr>
            <tr><td>Batch retrain, 40 epochs, seed 42</td><td>{bat['train_s']:.1f}</td><td>{bat['qe']:.4f}</td><td>{bat['te']:.4f}</td><td>{bat['n_clusters']}</td><td>{bg('batch'):.1f}</td><td>{100-bg('batch'):.1f}</td><td>{agree['batch']['jaccard_nonbackground']:.3f}</td></tr>
          </tbody>
          <caption>Table 2. Quantization error (QE) is the mean Euclidean distance from each cell to its best-matching node over all 2,097,152 cells; topographic error (TE) is the fraction of cells whose first and second best nodes are not lattice neighbours. Jaccard is computed on the union of non-background cells, treating the largest cluster of each map as background. Precision and recall of non-background cells against the fiducial are {agree['online']['precision']:.2f} / {agree['online']['recall']:.2f} (online) and {agree['batch']['precision']:.2f} / {agree['batch']['recall']:.2f} (batch).</caption>
        </table>
      </div>
      <p>All three maps describe the data equally well: the online retrain lands within 0.6 % of the fiducial quantization error and the batch trainer is 6 % lower, with a slightly higher topographic error (the batch map is a touch less smooth). The background fraction comes out at 85.0 % in every case. What differs is how the U-matrix partitions into clusters at a fixed threshold of 0.25: the paper's seven-way split separates positive and negative <code>j_par</code> sheets, the two signs of <code>j_asym</code>, and both signs of <code>e_par</code>, whereas the online retrain merges the current-sheet families into one cluster per sign of <code>j_par</code> and keeps only positive <code>e_par</code>, and the batch map keeps negative <code>e_par</code> and three current-sheet clusters (Table 3). About 41 % (online) and 36 % (batch) of the non-background volume is shared with the fiducial. For scale, the 28 maps of the shipped ensemble (different sizes, learning rates and step counts) have between 5 and 9 clusters each.</p>
      <div class="table-wrap">
        <table id="tab-clusters">
          <thead><tr><th scope="col">Map</th><th scope="col">Cluster</th><th scope="col">%</th><th scope="col">j_par</th><th scope="col">j_asym</th><th scope="col">b_perp</th><th scope="col">e_par</th><th scope="col">Reading</th></tr></thead>
          <tbody>
            {"".join("<tr><td>Online</td>"+row[4:] for row in cluster_rows_role("online").split(chr(10)))}
            {"".join("<tr><td>Batch</td>"+row[4:] for row in cluster_rows_role("batch").split(chr(10)))}
          </tbody>
          <caption>Table 3. Clusters of the two retrained maps at merge threshold 0.25, ordered by volume fraction, with mean standardized feature values. "Reading" is a mechanical label from the signs of the means (|e_par| &gt; 0.9 → parallel-field cluster; |j_par| &gt; 0.5, |j_asym| &gt; 0.9, b_perp &gt; 0.5 → current-sheet components; all small → background).</caption>
        </table>
      </div>

      <figure id="fig-umat" class="wide">
        <img src="{img('fig_umatrices.png')}" alt="Three U-matrix heat maps of 64 by 37 nodes: fiducial lattice, online retrain and batch retrain. Red dots mark local minima used as centroids, white lines are cluster boundaries after merging at threshold 0.25.">
        <figcaption>Figure 1. U-matrices of the fiducial lattice (left), the online retrain (centre) and the batch retrain (right). Red dots are the centroid seeds; white contours are the surviving cluster boundaries after merging at 0.25. The fiducial and online maps share the high-cost wall near the left edge that isolates the strongest current-sheet nodes; the batch map places the equivalent wall at the right edge, as expected for a map trained from a different random state.</figcaption>
      </figure>
      <figure id="fig-slices" class="wide">
        <img src="{img('fig_slices.png')}" alt="Three 128 by 128 slices at z equals 64 showing coloured non-background cluster labels with red and blue contours of parallel current at plus and minus 1.5 sigma.">
        <figcaption>Figure 2. Non-background labels in the z = 64 slice of the 128<sup>3</sup> cube for the same three maps; colours are cluster indices (not matched across panels), red and blue contours mark j_par = ±1.5σ. All three pick out the elongated sheets; the batch map's large green cluster is its negative-e_par cluster (8.7 % of the volume), which has no counterpart in the fiducial partition.</figcaption>
      </figure>
      <figure id="fig-fractions" class="wide">
        <img src="{img('fig_fractions.png')}" alt="Three bar charts of per-cluster volume fraction on a log scale, coloured by the sign of mean parallel current, with the background bar in grey.">
        <figcaption>Figure 3. Volume fraction per cluster (log scale). Red: mean j_par above 0.3σ; blue: below −0.3σ; teal: |j_par| small but not background; grey: background.</figcaption>
      </figure>

      <h3 id="sweep-heading">Seed sweep</h3>
      <p>To see whether the four-versus-seven cluster difference is a property of the trainer or of the seed, four more online seeds and three more batch configurations were run (Table 4).</p>
      <div class="table-wrap">
        <table id="tab-sweep">
          <thead><tr><th scope="col">Trainer</th><th scope="col">Run</th><th scope="col">Time (s)</th><th scope="col">Clusters at 0.25</th><th scope="col">Background %</th><th scope="col">Sheet-like %</th><th scope="col">QE (1/8 subsample)</th><th scope="col">TE (1/8 subsample)</th></tr></thead>
          <tbody>
            {sweep_rows}
          </tbody>
          <caption>Table 4. Seed sweep with the fiducial hyper-parameters. QE and TE here are evaluated on every eighth cell to save time, so they are not directly comparable with Table 2 (the fiducial lattice scores {fid['qe']:.4f} on all cells). The batch seed-43 run at 40 epochs merged into six clusters of which the largest holds only 47 % of the cells: the background was split in two, so "background" as the largest cluster is not meaningful for that run.</caption>
        </table>
      </div>
      <p>Across seeds 42 to 46 the online trainer gives 4 to 6 clusters with background between 83 and 89 %; the batch trainer gives 5 to 7. Quantization error is stable to the third decimal within each trainer. So the number of merged clusters at a fixed 0.25 threshold is seed-dependent for both trainers, and the paper's seven clusters is one draw from that distribution, which is the reason the paper runs the SCE over 36 realizations rather than reading a single map. Eighty batch epochs lowered the quantization error only from 0.2991 to 0.2969 and doubled the time, so 40 epochs is a reasonable default for this data set.</p>
    </section>

    <section aria-labelledby="metal-heading">
      <h2 id="metal-heading">3. SCE on the MacBook GPU through Metal</h2>
      <p><code>aweSOM.sce</code> picks JAX when <code>jax.default_backend()</code> is <code>gpu</code> or <code>METAL</code>, otherwise NumPy. A separate virtual environment <code>.venv-metal</code> was built with Homebrew Python 3.12, <code>jax-metal 0.1.1</code> and the matching <code>jax</code>/<code>jaxlib 0.4.34</code>, <code>numpy 2.1.3</code> and <code>numba 0.68</code>, with aweSOM installed editable from the research checkout. JAX reports <code>Metal device set to: Apple M1 Max</code> and <code>default_backend() == "METAL"</code>, and the SCE log confirms the JAX path. The command in both environments was:</p>
      <pre><code>python -m aweSOM.sce --folder . --dims 128 128 128</code></pre>
      <p>Two inputs were used: a six-map subset (the first six maps in sorted order) and the full 28-map ensemble (193 clusters in total) stored as <code>labels.*.npy</code> in <code>examples/plasma-turbulence/som_results</code>.</p>
      <div class="table-wrap">
        <table id="tab-sce-agree">
          <thead><tr><th scope="col">Quantity</th><th scope="col">CPU (NumPy, float64)</th><th scope="col">Metal (JAX, float32)</th><th scope="col">Agreement</th></tr></thead>
          <tbody>
            <tr><td>Gsum entries, 28 maps</td><td>193</td><td>193</td><td>max relative difference 5.5 × 10<sup>−7</sup>; identical ranking within every map</td></tr>
            <tr><td>Mask files <code>mask-{{run}}-id{{cid}}.npy</code></td><td>193, float64</td><td>193, float32</td><td>max absolute difference 1.0 × 10<sup>−4</sup></td></tr>
            <tr><td>CPU run versus <code>som_results/SCE/multimap_mappings.txt</code> shipped in the repo</td><td colspan="3">bitwise identical (28 maps, 193 entries)</td></tr>
          </tbody>
          <caption>Table 5. Agreement of the Metal SCE run with the NumPy run and with the result already stored in the repository. The float32 differences are far below the 0.25 / 0.5 / 0.8 ΣGsum thresholds used in the paper, so cluster selection is unaffected.</caption>
        </table>
      </div>
      <div class="table-wrap">
        <table id="tab-sce-time">
          <thead><tr><th scope="col">Input</th><th scope="col">Maps</th><th scope="col">CPU wall (s)</th><th scope="col">Metal wall (s)</th><th scope="col">Speed-up</th></tr></thead>
          <tbody>
            <tr><td>Six-map subset, contended</td><td>6</td><td>19.2</td><td>34.3</td><td>0.56×</td></tr>
            <tr><td>Full ensemble, contended (CPU run overlapped a batch training job; Metal run overlapped the seed sweep)</td><td>28</td><td>415</td><td>188</td><td>2.2×</td></tr>
            {clean_row}
          </tbody>
          <caption>Table 6. Wall-clock time of the whole SCE command including interpreter start-up, JIT compilation and file I/O. The contended rows are kept for transparency only; the Metal time varied between 188 s and 230 s across the two full runs, so the speed-up on this laptop is roughly 1.8 to 2.2×. The work scales with the number of cluster pairs, 193 × 192 / 2 ≈ 18,500 pairwise mask comparisons on a 128<sup>3</sup> grid for the full ensemble, which is why the small subset does not amortize the Metal compile and transfer cost.</caption>
        </table>
      </div>
      <p>Two tests in <code>tests/sce_test.py</code> failed under the Metal environment and were test bugs, now fixed on the branch: <code>test_jax_cpu_available</code> patched the backend name but never re-imported the module, so it only passed on machines whose real backend was already CPU; <code>test_compute_SQ</code> asserted a Python <code>float</code> and a NumPy array where JAX returns device arrays. Both now reload the module under the patch and compare by value and shape.</p>
    </section>

    <section aria-labelledby="repro-heading">
      <h2 id="repro-heading">4. Where the artefacts are and how to redo this</h2>
      <p>Everything below lives in the research checkout. The <code>planning/runs/</code> directory is git-ignored because it holds about 8.5 GB of masks and labels; the scripts and JSON summaries listed below were force-added so the numbers on this page are reproducible from the repository.</p>
      <ul>
        <li><code>planning/runs/plasma-2026-10-26/train_plasma.py</code>: loads the pickle, trains online and batch, writes <code>{{tag}}_lattice/umat/clusters/labels/dist.npy</code>, <code>results.json</code>, <code>batch_info.json</code>.</li>
        <li><code>analyze_plasma.py</code>: re-clusters all three maps at 0.25, writes <code>summary_thr025.json</code>, <code>agreement_thr025.json</code> and the three figures above.</li>
        <li><code>seed_sweep.py</code> → <code>seed_sweep.json</code>.</li>
        <li><code>sce_full_cpu/</code>, <code>sce_full_metal/</code>, <code>sce_time_cpu/</code>, <code>sce_time_metal/</code>: SCE outputs; <code>sce_timing.txt</code>: idle-machine wall times; <code>sce_*.log</code>.</li>
        <li><code>make_report.py</code>: builds this page from those files.</li>
      </ul>
      <pre><code>cd /Users/tvh0021/git_repos/aweSOM-research
# Metal environment (one-off)
/opt/homebrew/bin/python3.12 -m venv .venv-metal
.venv-metal/bin/pip install "jax==0.4.34" "jaxlib==0.4.34" jax-metal "numpy&lt;2.2" numba scipy scikit-learn matplotlib h5py pytest
.venv-metal/bin/pip install --no-deps -e .
# SCE on the GPU
cd planning/runs/plasma-2026-10-26/sce_time_metal
../../../../.venv-metal/bin/python -m aweSOM.sce --folder . --dims 128 128 128
# retrain and analyse
cd .. &amp;&amp; ../../../.venv/bin/python train_plasma.py &amp;&amp; ../../../.venv/bin/python analyze_plasma.py</code></pre>
    </section>

    <section aria-labelledby="limits-heading">
      <h2 id="limits-heading">5. Limitations and open points</h2>
      <ul>
        <li><strong>Batch trainer cannot run on Metal.</strong> jax-metal 0.1.1 pins jax 0.4.34, which lacks <code>jax.enable_x64</code>, <code>jax.shard_map</code> and float64 support; the trainer is written for jax 0.12 with float64 accumulation. <code>aweSOM.batch</code> now raises an <code>ImportError</code> that says so (commit <code>926a9c0</code>). Parallel GPU training therefore still needs a CUDA or ROCm machine, as noted in the session report.</li>
        <li><strong>Online trainer is CPU only by design.</strong> It is sequential and runs in numba; Metal plays no part in it.</li>
        <li><strong>Only the number-of-clusters question was swept.</strong> No attempt was made to tune the merge threshold per map; the paper's SCE over many realizations is the intended remedy, and that step is what Metal now accelerates.</li>
        <li><strong>Metal float32.</strong> Gsum and masks are float32 on Metal. The differences measured here are six orders of magnitude below the thresholds in use, but a future SCE criterion that compares Gsum values close to a threshold should be aware of it.</li>
        <li><strong>Timing caveats.</strong> The "contended" rows in Table 6 overlapped other jobs; the idle row was measured sequentially with nothing else running.</li>
      </ul>
      <aside class="callout limitation" aria-label="Not changed"><strong>Nothing in the main checkout was modified.</strong> The pickle, the example scripts and the shipped SCE results in <code>/Users/tvh0021/git_repos/aweSOM</code> were only read. The documentation discrepancies in section 1 (stored clusters, threshold 0.2 in the example, initial weight range) are reported, not fixed, because each would change published example output.</aside>
    </section>

    <section aria-labelledby="refs-heading">
      <h2 id="refs-heading">References</h2>
      <p class="ref" id="ref-ha">Ha, T., Nättilä, J., Davelaar, J., &amp; Sironi, L. (2024). Machine-Learning Characterization of Intermittency in Relativistic Pair Plasma Turbulence: Single and Double Sheet Structures. <a href="https://arxiv.org/abs/2410.01878">arXiv:2410.01878</a>. Fiducial parameters: Section 3.1 and Appendix A; volume fractions: Section 4.1; initialization: Appendix B; SCE thresholds: Section 4.2.</p>
      <p class="ref" id="ref-jaxmetal">Apple (2024). Accelerated JAX on Mac, jax-metal plugin. <a href="https://developer.apple.com/metal/jax/">developer.apple.com/metal/jax</a>. Version 0.1.1 requires jax 0.4.34 and supports float32 only.</p>
      <p class="ref" id="ref-session">Session report for the same branch: <code>planning/reports/2026-10-26-grmhd-scale-report.html</code> (bug fixes, frozen model artifact, batch trainer acceptance).</p>
    </section>
  </main>
  <footer>Generated by <code>planning/runs/plasma-2026-10-26/make_report.py</code> on 2026-10-26. Figures are embedded as base64 PNG; the page has no network dependencies.</footer>
</body>
</html>
"""
OUT.write_text(html)
print("wrote", OUT, len(html) // 1024, "kB")
