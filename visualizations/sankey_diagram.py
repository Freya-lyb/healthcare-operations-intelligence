"""
Sankey Diagram: Patient Flow Through the System's Failure Chain
Social vulnerability → No PCP relationship → ED use → 30-day readmission

Node layout (left → right):
  0  All Patients
  1  SDOH Risk          2  Unscreened / No Risk
  3  SDOH + No PCP      4  SDOH + Has PCP
  5  No SDOH + No PCP   6  No SDOH + Has PCP
  7  ED Visit           8  Non-Emergency Care
  9  30-Day Readmission 10 No Readmission

Requires: plotly, pandas
Output:   sankey_patient_flow.png
"""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

BASE_DIR  = Path(__file__).parent
CLEAN_DIR = BASE_DIR / 'DATA_CLEANED'

SDOH_DOMAIN_COLS = [
    'sdoh_social_connections',
    'sdoh_depression',
    'sdoh_intimate_partner_violance',
    'sdoh_physical_activity',
    'sdoh_stress',
    'sdoh_housing_stability',
    'sdoh_alcohol_use',
    'sdoh_transportation_needs',
    'sdoh_utilities',
    'sdoh_financial_resource_strain',
    'sdoh_food_insecurity',
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_patient_features() -> pd.DataFrame:
    path = CLEAN_DIR / 'patient_features.csv'
    if not path.exists():
        raise FileNotFoundError(f"patient_features.csv not found in {CLEAN_DIR}")
    print("Loading patient_features.csv...")
    df = pd.read_csv(path, low_memory=False)

    # Normalise patient key name
    if 'PatientDurableKey' not in df.columns and 'DurableKey' in df.columns:
        df = df.rename(columns={'DurableKey': 'PatientDurableKey'})

    # SDOH any-risk flag: at least one domain scored positive
    sdoh_cols = [c for c in SDOH_DOMAIN_COLS if c in df.columns]
    if not sdoh_cols:
        raise ValueError("No SDOH domain columns found in patient_features.csv")
    df['sdoh_any'] = (df[sdoh_cols] > 0).any(axis=1)

    # ED user flag
    df['ed_user'] = df['ed_visits'] > 0

    print(f"  {len(df):,} patients loaded")
    print(f"  SDOH domains used: {len(sdoh_cols)}")
    return df


def load_pcp_status() -> pd.DataFrame | None:
    """Derive per-patient PCP relationship from encounters_cleaned.csv.
    Returns a DataFrame with [PatientDurableKey, has_pcp], or None if
    encounters_cleaned is unavailable."""
    enc_path = CLEAN_DIR / 'encounters_cleaned.csv'
    if not enc_path.exists():
        print("  encounters_cleaned.csv not found — will use ed_ratio proxy for PCP status.")
        return None

    print("Deriving PCP status from encounters_cleaned.csv (chunked)...")
    chunks = []
    for chunk in pd.read_csv(
        enc_path,
        usecols=['PatientDurableKey', 'is_pcp_encounter'],
        chunksize=500_000,
        low_memory=False,
    ):
        chunks.append(chunk)
    enc = pd.concat(chunks, ignore_index=True)

    has_pcp = (
        enc.groupby('PatientDurableKey')['is_pcp_encounter']
        .max()
        .astype(bool)
        .rename('has_pcp')
        .reset_index()
    )
    n = has_pcp['has_pcp'].sum()
    print(f"  {n:,} of {len(has_pcp):,} patients have ≥1 PCP encounter")
    return has_pcp


def attach_pcp(df: pd.DataFrame, has_pcp: pd.DataFrame | None) -> pd.DataFrame:
    if has_pcp is None:
        # Proxy: patients whose visits are largely non-ED and frequent
        # enough to suggest an ongoing PCP relationship.
        # Less precise than the encounter-derived flag — noted in output.
        df['has_pcp'] = (df['ed_ratio'] < 0.15) & (df['total_encounters'] > 2)
        print("  WARNING: PCP status derived from ed_ratio proxy (less accurate).")
    else:
        df = df.merge(has_pcp, on='PatientDurableKey', how='left')
        df['has_pcp'] = df['has_pcp'].fillna(False)
    return df


# ── Flow computation ───────────────────────────────────────────────────────────

def compute_flows(df: pd.DataFrame) -> dict:
    s = df['sdoh_any']
    p = df['has_pcp']
    e = df['ed_user']
    r = df['readmit_30d'].astype(bool)

    return {
        'total':           len(df),
        # Level 1 — SDOH
        'sdoh_yes':        int(s.sum()),
        'sdoh_no':         int((~s).sum()),
        # Level 2 — PCP within each SDOH group
        'sdoh_y_no_pcp':   int((s  & ~p).sum()),
        'sdoh_y_pcp':      int((s  &  p).sum()),
        'sdoh_n_no_pcp':   int((~s & ~p).sum()),
        'sdoh_n_pcp':      int((~s &  p).sum()),
        # Level 3 — ED use from each of the 4 groups
        'g_yn_ed':         int((s  & ~p &  e).sum()),   # SDOH yes, no PCP → ED
        'g_yn_no':         int((s  & ~p & ~e).sum()),
        'g_yp_ed':         int((s  &  p &  e).sum()),   # SDOH yes, has PCP → ED
        'g_yp_no':         int((s  &  p & ~e).sum()),
        'g_nn_ed':         int((~s & ~p &  e).sum()),   # SDOH no, no PCP → ED
        'g_nn_no':         int((~s & ~p & ~e).sum()),
        'g_np_ed':         int((~s &  p &  e).sum()),   # SDOH no, has PCP → ED
        'g_np_no':         int((~s &  p & ~e).sum()),
        # Level 4 — readmission from ED vs non-ED
        'ed_readmit':      int(( e &  r).sum()),
        'ed_no_readmit':   int(( e & ~r).sum()),
        'no_ed_readmit':   int((~e &  r).sum()),
        'no_ed_no_readmit':int((~e & ~r).sum()),
    }


# ── Sankey builder ─────────────────────────────────────────────────────────────

# Palette aligned to report's causal diagram:
#   red tones = risk / bad outcome
#   green tones = protective / good outcome
#   blue tones = neutral
_NODE_COLORS = [
    '#4472C4',  # 0  All Patients
    '#E05C30',  # 1  SDOH Risk
    '#5B9BD5',  # 2  Unscreened / No Risk
    '#C00000',  # 3  SDOH + No PCP  ← highest risk
    '#ED7D31',  # 4  SDOH + Has PCP
    '#F4B942',  # 5  No SDOH + No PCP
    '#70AD47',  # 6  No SDOH + Has PCP  ← most protected
    '#FF2B2B',  # 7  ED Visit
    '#4BACC6',  # 8  Non-Emergency Care
    '#7030A0',  # 9  30-Day Readmission
    '#375623',  # 10 No Readmission
]

_LINK_ALPHA = [
    'rgba(68,114,196,0.30)',   # from 0
    'rgba(224,92,48,0.35)',    # from 1
    'rgba(91,155,213,0.25)',   # from 2
    'rgba(192,0,0,0.40)',      # from 3 — dominant bad flow, more opaque
    'rgba(237,125,49,0.30)',   # from 4
    'rgba(244,185,66,0.30)',   # from 5
    'rgba(112,173,71,0.25)',   # from 6
    'rgba(255,43,43,0.35)',    # from 7
    'rgba(75,172,198,0.25)',   # from 8
]


def build_sankey(c: dict) -> go.Figure:
    def lbl(name, n):
        pct = f" ({n / c['total'] * 100:.1f}%)" if n < c['total'] else ""
        return f"{name}<br><b>{n:,}{pct}</b>"

    total_ed = c['g_yn_ed'] + c['g_yp_ed'] + c['g_nn_ed'] + c['g_np_ed']
    total_no_ed = c['g_yn_no'] + c['g_yp_no'] + c['g_nn_no'] + c['g_np_no']
    total_readmit = c['ed_readmit'] + c['no_ed_readmit']

    node_labels = [
        lbl("All Patients",          c['total']),           # 0
        lbl("SDOH Risk",             c['sdoh_yes']),        # 1
        lbl("Unscreened / No Risk",  c['sdoh_no']),         # 2
        lbl("SDOH + No PCP",         c['sdoh_y_no_pcp']),   # 3
        lbl("SDOH + Has PCP",        c['sdoh_y_pcp']),      # 4
        lbl("No SDOH + No PCP",      c['sdoh_n_no_pcp']),   # 5
        lbl("No SDOH + Has PCP",     c['sdoh_n_pcp']),      # 6
        lbl("ED Visit",              total_ed),             # 7
        lbl("Non-Emergency Care",    total_no_ed),          # 8
        lbl("30-Day Readmission",    total_readmit),        # 9
        lbl("No Readmission",        c['total'] - total_readmit),  # 10
    ]

    # (source_idx, target_idx, value, hover_label)
    links = [
        # Level 0 → 1
        (0, 1,  c['sdoh_yes'],       "Any SDOH Risk"),
        (0, 2,  c['sdoh_no'],        "Unscreened / No Risk"),
        # Level 1 → 2
        (1, 3,  c['sdoh_y_no_pcp'],  "SDOH — No PCP"),
        (1, 4,  c['sdoh_y_pcp'],     "SDOH — Has PCP"),
        (2, 5,  c['sdoh_n_no_pcp'],  "No SDOH — No PCP"),
        (2, 6,  c['sdoh_n_pcp'],     "No SDOH — Has PCP"),
        # Level 2 → 3  (ED vs non-ED)
        (3, 7,  c['g_yn_ed'],        "SDOH + No PCP → ED"),
        (3, 8,  c['g_yn_no'],        "SDOH + No PCP → Non-ED"),
        (4, 7,  c['g_yp_ed'],        "SDOH + PCP → ED"),
        (4, 8,  c['g_yp_no'],        "SDOH + PCP → Non-ED"),
        (5, 7,  c['g_nn_ed'],        "No SDOH + No PCP → ED"),
        (5, 8,  c['g_nn_no'],        "No SDOH + No PCP → Non-ED"),
        (6, 7,  c['g_np_ed'],        "No SDOH + PCP → ED"),
        (6, 8,  c['g_np_no'],        "No SDOH + PCP → Non-ED"),
        # Level 3 → 4  (readmission)
        (7, 9,  c['ed_readmit'],        "ED → 30-Day Readmission"),
        (7, 10, c['ed_no_readmit'],     "ED → No Readmission"),
        (8, 9,  c['no_ed_readmit'],     "Non-ED → 30-Day Readmission"),
        (8, 10, c['no_ed_no_readmit'],  "Non-ED → No Readmission"),
    ]

    sources     = [l[0] for l in links]
    targets     = [l[1] for l in links]
    values      = [l[2] for l in links]
    link_labels = [l[3] for l in links]
    link_colors = [_LINK_ALPHA[s] for s in sources]

    fig = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(
            pad=18,
            thickness=22,
            line=dict(color='white', width=0.5),
            label=node_labels,
            color=_NODE_COLORS,
            hovertemplate='%{label}<extra></extra>',
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color=link_colors,
            hovertemplate='%{label}<br>Patients: <b>%{value:,}</b><extra></extra>',
        ),
    ))

    fig.update_layout(
        title=dict(
            text=(
                'Patient Flow: Social Vulnerability → No Primary Care → '
                'Emergency Use → Readmission'
            ),
            font=dict(size=15),
        ),
        font=dict(family='Arial', size=12, color='#2C2C2C'),
        height=740,
        width=1350,
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df      = load_patient_features()
    has_pcp = load_pcp_status()
    df      = attach_pcp(df, has_pcp)

    print("\nComputing patient flows...")
    counts = compute_flows(df)

    total_ed = counts['g_yn_ed'] + counts['g_yp_ed'] + counts['g_nn_ed'] + counts['g_np_ed']
    total_readmit = counts['ed_readmit'] + counts['no_ed_readmit']
    print(f"\n── Flow Summary ─────────────────────────────")
    print(f"  Total patients:       {counts['total']:>10,}")
    print(f"  SDOH risk:            {counts['sdoh_yes']:>10,}  ({counts['sdoh_yes']/counts['total']*100:.1f}%)")
    print(f"    └─ No PCP:          {counts['sdoh_y_no_pcp']:>10,}  ({counts['sdoh_y_no_pcp']/counts['total']*100:.1f}%)")
    print(f"  ED users:             {total_ed:>10,}  ({total_ed/counts['total']*100:.1f}%)")
    print(f"  30-day readmissions:  {total_readmit:>10,}  ({total_readmit/counts['total']*100:.1f}%)")

    print("\nBuilding Sankey diagram...")
    fig = build_sankey(counts)

    out = BASE_DIR / 'sankey_patient_flow.png'
    fig.write_image(str(out), scale=2)
    print(f"Saved: {out.name}")
