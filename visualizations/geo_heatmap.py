"""
Geographic Choropleth: Healthcare Utilization by Census Block Group
Upgraded from bubble scatter to polygon fill using Census Bureau boundaries.
Output: 3 static PNG maps (encounters/capita, ED rate, night visit rate)

Requires: plotly, pandas, numpy, requests
Optional: geopandas (faster boundary download; falls back to TIGER REST API)
"""

from pathlib import Path
import json, requests, zipfile, io, tempfile
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

BASE_DIR  = Path(__file__).parent
ROOT_DIR  = BASE_DIR.parent          # repo root (one level up from visuals/)
DATA_DIR  = ROOT_DIR / 'DATA'
CLEAN_DIR = ROOT_DIR / 'DATA_CLEANED'
GEO_CACHE = ROOT_DIR / 'kansas_bg.geojson'

KANSAS_CENTER = {'lat': 38.50, 'lon': -98.35}


# ── Block-group polygon boundaries ────────────────────────────────────────────

def get_kansas_geojson() -> dict:
    """Return GeoJSON FeatureCollection for Kansas census block groups (2020).
    Downloads from Census Bureau on first run, then reads from local cache."""
    if GEO_CACHE.exists():
        print("  Loading cached block-group boundaries...")
        with open(GEO_CACHE) as f:
            return json.load(f)

    geojson = _download_via_geopandas() or _download_via_tiger_api()

    with open(GEO_CACHE, 'w') as f:
        json.dump(geojson, f)
    print(f"  Cached to {GEO_CACHE.name}")
    return geojson


def _download_via_geopandas() -> dict | None:
    try:
        import geopandas as gpd
        url = 'https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_20_bg_500k.zip'
        print("  Downloading Kansas block-group shapefile (Census Bureau)...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(tmp)
            shp = next(Path(tmp).glob('*.shp'))
            gdf = gpd.read_file(shp).to_crs('EPSG:4326')
        gdf['GEOID'] = gdf['GEOID'].astype(str).str.zfill(12)
        return json.loads(gdf[['geometry', 'GEOID']].to_json())
    except Exception as e:
        print(f"  geopandas unavailable ({e}); falling back to TIGER REST API...")
        return None


def _download_via_tiger_api() -> dict:
    """Paginate Census TIGER REST API — no geopandas required."""
    url = ('https://tigerweb.geo.census.gov/arcgis/rest/services/'
           'TIGERweb/tigerWMS_Census2020/MapServer/10/query')
    features, offset = [], 0
    print("  Fetching Kansas block groups from TIGER REST API (paginated)...")
    while True:
        r = requests.get(url, params={
            'where':             "STATE='20'",
            'outFields':         'GEOID',
            'returnGeometry':    'true',
            'geometryPrecision': 4,
            'f':                 'geojson',
            'resultRecordCount': 1000,
            'resultOffset':      offset,
        }, timeout=90)
        batch = r.json().get('features', [])
        if not batch:
            break
        for feat in batch:
            feat['properties']['GEOID'] = str(feat['properties']['GEOID']).zfill(12)
        features.extend(batch)
        offset += len(batch)
        print(f"    {len(features)} block groups fetched...")
        if len(batch) < 1000:
            break
    print(f"  Done — {len(features)} block groups total.")
    return {'type': 'FeatureCollection', 'features': features}


# ── Per-block-group statistics ─────────────────────────────────────────────────

def compute_geo_stats() -> pd.DataFrame:
    # Census reference (population)
    geo = pd.read_csv(DATA_DIR / 'tigercensuscodes.csv', low_memory=False)
    geo['GEOID'] = geo['GEOID'].astype(str).str.zfill(12)

    # Patient → census block mapping
    pat = pd.read_csv(
        DATA_DIR / 'patients.csv',
        usecols=['DurableKey', 'CensusBlockGroupFipsCode'],
        low_memory=False,
    )
    pat = pat.dropna(subset=['CensusBlockGroupFipsCode'])
    pat = pat[~pat['CensusBlockGroupFipsCode'].isin(['*Unspecified', '*Unknown'])]
    pat['GEOID'] = (pat['CensusBlockGroupFipsCode']
                    .astype(str)
                    .str.replace('.0', '', regex=False)
                    .str.zfill(12))

    # Encounters — use cleaned file if available, raw otherwise
    enc_path = (CLEAN_DIR / 'encounters_cleaned.csv'
                if (CLEAN_DIR / 'encounters_cleaned.csv').exists()
                else DATA_DIR / 'encounters.csv')
    print(f"  Loading encounters from {enc_path.name} (chunked)...")
    cols = ['PatientDurableKey', 'AdmitHour', 'AdmitYear', 'AdmitMonth', 'AdmitDay',
            'IsEdVisit', 'EncounterKey']
    chunks = []
    for chunk in pd.read_csv(enc_path, usecols=cols, chunksize=500_000, low_memory=False):
        chunks.append(chunk[chunk['AdmitYear'].between(2022, 2025)])
    enc = pd.concat(chunks, ignore_index=True)
    del chunks

    # Derive day-of-week (0=Mon … 6=Sun) so we can flag Saturday/Sunday
    admit_dt = pd.to_datetime(
        enc[['AdmitYear', 'AdmitMonth', 'AdmitDay']].rename(
            columns={'AdmitYear': 'year', 'AdmitMonth': 'month', 'AdmitDay': 'day'}
        ),
        errors='coerce',
    )
    enc['is_weekend'] = admit_dt.dt.dayofweek >= 5  # Sat=5, Sun=6

    enc_geo = enc.merge(
        pat[['DurableKey', 'GEOID']],
        left_on='PatientDurableKey', right_on='DurableKey', how='inner',
    )

    stats = enc_geo.groupby('GEOID').agg(
        total_encounters=('EncounterKey', 'count'),
        ed_visits=('IsEdVisit', 'sum'),
        night_visits=('AdmitHour', lambda x: ((x >= 22) | (x <= 6)).sum()),
        weekend_visits=('is_weekend', 'sum'),
        unique_patients=('PatientDurableKey', 'nunique'),
    ).reset_index()

    stats = stats.merge(geo[['GEOID', 'PopulationValue']], on='GEOID', how='inner')
    pop = stats['PopulationValue'].clip(lower=1)
    tot = stats['total_encounters'].clip(lower=1)
    stats['enc_per_capita']  = stats['total_encounters'] / pop
    stats['ed_rate']         = stats['ed_visits']        / tot * 100
    stats['night_rate']      = stats['night_visits']     / tot * 100
    stats['weekend_rate']    = stats['weekend_visits']   / tot * 100

    # Cap at 99th percentile to avoid outlier-driven color collapse
    for col in ['enc_per_capita', 'ed_rate', 'night_rate', 'weekend_rate']:
        stats[col] = stats[col].clip(upper=stats[col].quantile(0.99))

    return stats


# ── Choropleth renderer ────────────────────────────────────────────────────────

def make_choropleth(geojson, stats, *, metric, title, color_scale, label, filename):
    fig = px.choropleth_mapbox(
        stats,
        geojson=geojson,
        locations='GEOID',
        featureidkey='properties.GEOID',
        color=metric,
        color_continuous_scale=color_scale,
        mapbox_style='carto-positron',
        zoom=6,
        center=KANSAS_CENTER,
        opacity=0.75,
        labels={
            metric:            label,
            'total_encounters': 'Total Encounters',
            'unique_patients':  'Unique Patients',
            'PopulationValue':  'Population',
        },
        title=title,
        hover_data={
            'GEOID':            False,
            metric:             ':.2f',
            'total_encounters': ':,',
            'unique_patients':  ':,',
            'PopulationValue':  ':,',
        },
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=55, b=0),
        height=720,
        width=1100,
        title_font_size=15,
    )
    fig.update_coloraxes(colorbar_thickness=15, colorbar_len=0.55)

    out = BASE_DIR / filename
    fig.write_image(str(out), scale=2)
    print(f"  Saved: {filename}")


def make_night_weekend_choropleth(geojson, stats, *, filename):
    """Continuous bivariate choropleth: night rate → R channel, weekend rate → B channel.

    Both metrics are normalised to [0, 1].  The RGB colour per polygon is:
      R = 255 - round(200 * weekend_norm)   (blue ink removes red)
      G = 255 - round(210 * night_norm) - round(130 * weekend_norm)
      B = 255 - round(200 * night_norm)     (red ink removes blue)
    Corner colours: low/low = near-white, high-night/low-wknd = red,
    low-night/high-wknd = blue, high/high = deep purple.

    Uses px.choropleth_mapbox with each unique hex as its own colour category
    so Plotly renders the exact per-feature colour without discretising through z.
    """

    stats = stats.copy()

    def norm(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9)

    nn = norm(stats['night_rate'])    # night  → redder
    wn = norm(stats['weekend_rate'])  # weekend → bluer

    def to_hex(n, w):
        r = int(max(0, 255 - 200 * w))
        g = int(max(0, 255 - 210 * n - 130 * w))
        b = int(max(0, 255 - 200 * n))
        return f'#{r:02x}{g:02x}{b:02x}'

    # Discretize each axis to K steps → z = ni*K + wi (K² total colour classes).
    # K=25 gives 625 colorscale stops — smooth-looking without memory issues.
    K = 25
    ni_arr = (nn * (K - 1)).round().astype(int).clip(0, K - 1)
    wi_arr = (wn * (K - 1)).round().astype(int).clip(0, K - 1)
    stats['_z'] = ni_arr * K + wi_arr

    colorscale = []
    total = K * K
    for ni in range(K):
        for wi in range(K):
            idx = ni * K + wi
            col = to_hex(ni / (K - 1), wi / (K - 1))
            colorscale.append([idx / total,       col])
            colorscale.append([(idx + 1) / total, col])

    import plotly.graph_objects as go
    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=stats['GEOID'],
        z=stats['_z'],
        featureidkey='properties.GEOID',
        colorscale=colorscale,
        zmin=0,
        zmax=total,
        marker=dict(opacity=0.60, line_width=0),
        showscale=False,
        hovertemplate=(
            'Night Rate: <b>%{customdata[0]:.1f}%</b><br>'
            'Weekend Rate: <b>%{customdata[1]:.1f}%</b>'
            '<extra></extra>'
        ),
        customdata=stats[['night_rate', 'weekend_rate']].values,
    ))

    # ── Gradient legend square (paper coordinates) ─────────────────────────────
    # Render a 10×10 grid of colour swatches sampling the full continuous space.
    N = 10
    lx0, ly0 = 0.80, 0.02
    cell_w = 0.018
    cell_h = 0.018

    shapes, annotations = [], []
    for ni in range(N):
        for wi in range(N):
            n_val = ni / (N - 1)
            w_val = wi / (N - 1)
            shapes.append(dict(
                type='rect', xref='paper', yref='paper',
                x0=lx0 + wi * cell_w,       y0=ly0 + ni * cell_h,
                x1=lx0 + (wi+1) * cell_w,   y1=ly0 + (ni+1) * cell_h,
                fillcolor=to_hex(n_val, w_val),
                line_width=0,
            ))

    # Corner labels
    for text, x, y in [
        ('← Low weekend  High →', lx0 + N * cell_w / 2, ly0 - 0.022),
        ('Low<br>night', lx0 - 0.015, ly0 + 0.5 * cell_h),
        ('High<br>night', lx0 - 0.015, ly0 + (N - 0.5) * cell_h),
        ('<b>Visit Pattern</b>', lx0 + N * cell_w / 2, ly0 + N * cell_h + 0.018),
    ]:
        annotations.append(dict(
            xref='paper', yref='paper', x=x, y=y,
            text=text, showarrow=False, font_size=8, align='center',
        ))

    fig.update_layout(
        mapbox=dict(style='open-street-map', zoom=6, center=KANSAS_CENTER),
        title=dict(
            text=(
                'Night & Weekend Visit Concentration by Census Block Group<br>'
                '<sup>Colour encodes both dimensions simultaneously — '
                'red = high night rate · blue = high weekend rate · '
                'purple = high both</sup>'
            ),
            font_size=13,
        ),
        margin=dict(l=0, r=0, t=65, b=0),
        height=720,
        width=1150,
        paper_bgcolor='white',
        shapes=shapes,
        annotations=annotations,
    )

    out = BASE_DIR / filename
    fig.write_image(str(out), scale=2)
    print(f"  Saved: {filename}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Step 1 — Loading block-group polygon boundaries...")
    geojson = get_kansas_geojson()
    print(f"  {len(geojson['features'])} block groups loaded\n")

    print("Step 2 — Computing per-block-group statistics...")
    stats = compute_geo_stats()
    print(f"  {len(stats)} block groups with encounter data\n")

    print("Step 3 — Rendering choropleth maps...")

    make_choropleth(geojson, stats,
        metric='enc_per_capita',
        label='Encounters / Capita',
        color_scale='RdYlGn_r',
        title='Healthcare Demand by Census Block Group (2022–2025)',
        filename='geo_utilization_map.png',
    )

    make_choropleth(geojson, stats,
        metric='ed_rate',
        label='ED Visit Rate (%)',
        color_scale='OrRd',
        title='Emergency Department Visit Rate by Census Block Group',
        filename='geo_ed_rate_map.png',
    )

    # Night + weekend combined — gradient (Plasma) for night, opacity wash for weekend
    make_night_weekend_choropleth(geojson, stats,
        filename='geo_night_weekend_map.png',
    )

    print("\nDone:")
    print("  geo_utilization_map.png    — overall healthcare demand")
    print("  geo_ed_rate_map.png        — ED hotspots")
    print("  geo_night_weekend_map.png  — night rate (color) + weekend rate (blue wash)")
