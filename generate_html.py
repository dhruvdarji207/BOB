"""
generate_html.py
Generates a fully standalone index.html with all ML data embedded.
Run: python generate_html.py
"""
import json
import pandas as pd
import joblib

# ── Load model and metrics ──────────────────────────────────────
m     = json.load(open('models/model_metrics.json', encoding='utf-8'))
model = joblib.load('models/car_price_model.pkl')

dd  = m['dropdown_values']
nr  = m['numerical_ranges']
fi  = m['feature_importance']
avp = m['actual_vs_predicted']
pd_ = m['price_distribution']
am  = m['all_models']
bm  = m['best_metrics']

NUMERICAL   = ['symboling','wheelbase','carlength','carwidth','carheight',
               'curbweight','enginesize','boreratio','stroke','compressionratio',
               'horsepower','peakrpm','citympg','highwaympg']
CATEGORICAL = ['manufacturer','fueltype','aspiration','doornumber','carbody',
               'drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem']

df = pd.read_csv('CarPrice_Assignment.csv')
df['manufacturer'] = (df['CarName'].str.split().str[0].str.lower()
    .replace({'maxda':'mazda','toyouta':'toyota','porcshce':'porsche',
              'vokswagen':'volkswagen','vw':'volkswagen'}))
df.drop(columns=['car_ID','CarName'], inplace=True)
X     = df[NUMERICAL + CATEGORICAL]
preds = model.predict(X)

rows = []
for i in range(len(df)):
    r = {}
    for k in NUMERICAL:
        r[k] = round(float(df[k].iloc[i]), 4)
    for k in CATEGORICAL:
        r[k] = str(df[k].iloc[i])
    r['price']     = round(float(df['price'].iloc[i]), 2)
    r['predicted'] = round(float(preds[i]), 2)
    rows.append(r)

rows_js = json.dumps(rows, separators=(',', ':'))
meta_js = json.dumps({
    'best_model':   m['best_model'],
    'best_metrics': bm,
    'all_models':   am,
    'actual_vs_predicted': avp,
}, separators=(',', ':'))

print(f"Training rows : {len(rows)}")
print(f"rows_js size  : {len(rows_js):,} chars")
print(f"meta_js size  : {len(meta_js):,} chars")

# ── Helper: dropdown options ────────────────────────────────────
def opts(key, tfm=str.capitalize):
    return '\n'.join(
        f'<option value="{v}">{tfm(v)}</option>'
        for v in dd.get(key, [])
    )

def nv(key, attr, default=0):
    return nr.get(key, {}).get(attr, default)

# ── Model comparison table rows ─────────────────────────────────
mc_rows = ''
for name, vals in am.items():
    best  = ' class="best-row"' if name == m['best_model'] else ''
    badge = '<span class="badge">Best</span> ' if name == m['best_model'] else ''
    mc_rows += (f'<tr{best}><td>{badge}{name}</td>'
                f'<td>{vals["r2"]}</td>'
                f'<td>${vals["mae"]:,.0f}</td>'
                f'<td>${vals["rmse"]:,.0f}</td></tr>\n')

# ── Feature importance bars ─────────────────────────────────────
max_imp  = fi[0]['importance'] if fi else 1
fi_bars  = ''
for item in fi[:10]:
    pct = round(item['importance'] / max_imp * 100, 1)
    fi_bars += (f'<div class="fi-row">'
                f'<div class="fi-label">{item["feature"]}</div>'
                f'<div class="fi-bar-wrap">'
                f'<div class="fi-bar" style="width:{pct}%"></div>'
                f'<span class="fi-val">{item["importance"]:.4f}</span>'
                f'</div></div>\n')

# ── Price distribution bars ─────────────────────────────────────
max_cnt   = max(x['count'] for x in pd_)
dist_bars = ''
for item in pd_:
    pct = round(item['count'] / max_cnt * 100, 1)
    dist_bars += (f'<div class="fi-row">'
                  f'<div class="fi-label" style="font-size:.7rem;min-width:130px">'
                  f'{item["range"]}</div>'
                  f'<div class="fi-bar-wrap">'
                  f'<div class="fi-bar" style="width:{pct}%;background:#8b5cf6"></div>'
                  f'<span class="fi-val">{item["count"]}</span>'
                  f'</div></div>\n')

# ── Feature importance weights for JS (top 14) ──────────────────
fi_weights_js = json.dumps(
    {item['feature']: item['importance'] for item in fi[:14]},
    separators=(',', ':')
)

# ── Build the HTML ───────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>CarPriceAI - Car Price Prediction</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--bg2:#161b22;--surf:#1e2533;--surf2:#242d3d;
  --border:#2d3748;--border2:#3d4f6b;
  --text:#e6edf3;--muted:#8b949e;--dim:#6e7f96;
  --accent:#3b82f6;--accent2:#60a5fa;--glow:rgba(59,130,246,.2);
  --success:#22c55e;--danger:#ef4444;--purple:#8b5cf6;
  --r:12px;--rs:8px;--font:'Segoe UI',system-ui,sans-serif;
}
body{font-family:var(--font);font-size:15px;line-height:1.65;background:var(--bg);color:var(--text);min-height:100vh}
a{color:var(--accent2);text-decoration:none}
.hidden{display:none!important}
.nav{position:sticky;top:0;z-index:100;background:rgba(13,17,23,.95);backdrop-filter:blur(12px);border-bottom:1px solid var(--border);padding:0 1.5rem;height:60px;display:flex;align-items:center;gap:2rem}
.brand{font-size:1.15rem;font-weight:800;color:var(--text);display:flex;align-items:center;gap:.4rem}
.brand span{color:var(--accent2)}
.nav-links{display:flex;gap:.15rem;margin-left:auto}
.nav-links a{padding:.4rem .85rem;border-radius:6px;color:var(--muted);font-size:.88rem;font-weight:500;cursor:pointer;transition:.15s}
.nav-links a:hover,.nav-links a.active{color:var(--text);background:var(--surf)}
.nav-links a.active{color:var(--accent2)}
.page{display:none;padding:2rem 1.5rem 4rem;max-width:1200px;margin:0 auto}
.page.active{display:block}
#page-home{padding:0;max-width:none}
.hero{max-width:1200px;margin:0 auto;padding:5rem 1.5rem;display:flex;align-items:center;gap:4rem;flex-wrap:wrap}
.hero-content{flex:1;min-width:280px}
.hero-badge{display:inline-block;background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.3);color:var(--accent2);padding:.3rem 1rem;border-radius:50px;font-size:.75rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;margin-bottom:1.25rem}
.hero h1{font-size:clamp(2rem,5vw,3.2rem);font-weight:800;line-height:1.12;letter-spacing:-.02em;margin-bottom:1rem}
.hero h1 span{color:var(--accent2)}
.hero p{color:var(--muted);font-size:1rem;line-height:1.7;margin-bottom:2rem;max-width:520px}
.hero-btns{display:flex;gap:1rem;flex-wrap:wrap}
.hero-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:1rem;max-width:1200px;margin:0 auto;padding:2rem 1.5rem;border-top:1px solid var(--border);background:var(--bg2)}
.hs-item{text-align:center}
.hs-num{font-size:1.5rem;font-weight:800;color:var(--text)}
.hs-lbl{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.btn{display:inline-flex;align-items:center;gap:.5rem;padding:.65rem 1.5rem;border-radius:var(--rs);font-family:var(--font);font-size:.92rem;font-weight:600;cursor:pointer;border:none;transition:.15s;white-space:nowrap}
.btn-primary{background:var(--accent);color:#fff}
.btn-primary:hover{background:var(--accent2)}
.btn-outline{background:transparent;color:var(--accent2);border:1.5px solid var(--border2)}
.btn-outline:hover{background:var(--surf)}
.btn-lg{padding:.8rem 2rem;font-size:1rem;border-radius:var(--r)}
.btn:disabled{opacity:.5;cursor:not-allowed}
.ph{background:var(--bg2);border-bottom:1px solid var(--border);padding:2rem 1.5rem;margin:-2rem -1.5rem 2rem}
.ph h1{font-size:1.7rem;font-weight:700;margin-bottom:.3rem}
.ph p{color:var(--muted);font-size:.92rem}
.predict-layout{display:grid;grid-template-columns:1fr 340px;gap:2rem;align-items:start}
.fsec{background:var(--surf);border:1px solid var(--border);border-radius:var(--r);padding:1.5rem;margin-bottom:1.25rem}
.fsec-hdr{display:flex;align-items:flex-start;gap:.85rem;margin-bottom:1.25rem;padding-bottom:1rem;border-bottom:1px solid var(--border)}
.snum{width:28px;height:28px;background:var(--accent);color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.82rem;flex-shrink:0;margin-top:2px}
.fsec-hdr h3{font-size:.95rem;font-weight:600;margin-bottom:.1rem}
.fsec-hdr p{font-size:.78rem;color:var(--muted)}
.fgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem}
.fg{display:flex;flex-direction:column;gap:.35rem}
.fg label{font-size:.82rem;font-weight:500;color:var(--muted)}
.fg input,.fg select{background:var(--bg);border:1.5px solid var(--border);border-radius:var(--rs);color:var(--text);padding:.55rem .8rem;font-family:var(--font);font-size:.88rem;transition:.15s;width:100%;appearance:none}
.fg input:focus,.fg select:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--glow)}
.fg input.err,.fg select.err{border-color:var(--danger)}
.req{color:var(--danger)}
.submit-area{display:flex;gap:1rem;flex-wrap:wrap;margin-top:.5rem}
.alert-err{background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);color:#fca5a5;padding:.85rem 1rem;border-radius:var(--rs);font-size:.88rem;margin-top:.75rem}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.result-panel{position:sticky;top:70px}
.r-placeholder{background:var(--surf);border:1px dashed var(--border2);border-radius:var(--r);padding:2rem;text-align:center}
.r-placeholder .ico{font-size:2.5rem;margin-bottom:1rem}
.r-placeholder h3{font-size:1rem;font-weight:600;margin-bottom:.5rem}
.r-placeholder p{font-size:.83rem;color:var(--muted)}
.r-card{background:var(--surf);border:1px solid var(--accent);border-radius:var(--r);padding:1.75rem;box-shadow:0 0 30px var(--glow);animation:slideIn .3s ease}
@keyframes slideIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.r-hdr{display:flex;justify-content:space-between;font-size:.78rem;font-weight:600;margin-bottom:.85rem}
.r-status{color:var(--success)}
.r-label{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:.3rem}
.r-price{font-size:2.4rem;font-weight:800;color:var(--accent2);margin-bottom:.75rem;letter-spacing:-.02em}
.r-disc{font-size:.76rem;color:var(--dim);background:rgba(59,130,246,.07);border-radius:6px;padding:.6rem .85rem;margin-bottom:1rem;line-height:1.5}
.meta-row{display:flex;justify-content:space-between;padding:.4rem 0;border-bottom:1px solid var(--border);font-size:.83rem}
.meta-row:last-child{border-bottom:none}
.meta-k{color:var(--muted)}
.meta-v{font-weight:600}
.r-summary{font-size:.76rem;color:var(--dim);margin:.85rem 0 1rem;line-height:1.6}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;margin-bottom:1.5rem}
.kpi{background:var(--surf);border:1px solid var(--border);border-radius:var(--r);padding:1.1rem 1.25rem}
.kpi.hl{border-color:var(--accent)}
.kpi-lbl{font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:.35rem}
.kpi-val{font-size:1.2rem;font-weight:700;word-break:break-word}
.ins-row{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem;margin-bottom:1.25rem}
.ins-card{background:var(--surf);border:1px solid var(--border);border-radius:var(--r);padding:1.35rem}
.ins-card.full{grid-column:1/-1}
.card-title{font-size:.9rem;font-weight:600;margin-bottom:.85rem}
.tbl{width:100%;border-collapse:collapse;font-size:.85rem}
.tbl th{padding:.55rem .6rem;text-align:left;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);border-bottom:1px solid var(--border)}
.tbl td{padding:.55rem .6rem;border-bottom:1px solid var(--border)}
.tbl tr:last-child td{border-bottom:none}
.tbl th:not(:first-child),.tbl td:not(:first-child){text-align:right}
.best-row{background:rgba(59,130,246,.08)}
.badge{background:var(--accent);color:#fff;padding:.1rem .45rem;border-radius:20px;font-size:.68rem;font-weight:700;margin-right:.35rem}
.fi-row{display:flex;align-items:center;gap:.75rem;margin-bottom:.55rem}
.fi-label{font-size:.78rem;color:var(--muted);min-width:110px;text-align:right}
.fi-bar-wrap{flex:1;display:flex;align-items:center;gap:.5rem}
.fi-bar{height:14px;background:var(--accent);border-radius:3px}
.fi-val{font-size:.72rem;color:var(--dim);white-space:nowrap}
.about-card{background:var(--surf);border:1px solid var(--border);border-radius:var(--r);padding:1.75rem;margin-bottom:1.25rem;max-width:860px}
.about-card h2{font-size:1.2rem;font-weight:700;margin-bottom:.85rem}
.about-card h4{font-size:.85rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin:1rem 0 .5rem}
.about-card p{font-size:.88rem;color:var(--muted);line-height:1.7;margin-bottom:.6rem}
.wf{display:flex;flex-direction:column;gap:.15rem}
.wf-step{display:flex;gap:.85rem;background:var(--bg);border:1px solid var(--border);border-radius:var(--rs);padding:.85rem 1rem}
.wf-n{width:26px;height:26px;background:var(--accent);color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.8rem;flex-shrink:0}
.wf-arr{text-align:center;color:var(--accent);font-size:1rem;padding:.05rem 0}
.wf-b h4{font-size:.85rem;font-weight:600;margin-bottom:.2rem;text-transform:none;letter-spacing:0;color:var(--text)}
.wf-b p{font-size:.78rem;color:var(--muted);margin:0}
.tag-list{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.5rem}
.tag{background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.25);color:var(--accent2);padding:.2rem .65rem;border-radius:50px;font-size:.76rem}
.method-list{padding-left:1.1rem;list-style:disc;color:var(--muted);font-size:.85rem}
.method-list li{padding:.2rem 0;line-height:1.6}
.method-list strong{color:var(--text)}
.footer{background:var(--bg2);border-top:1px solid var(--border);padding:1.5rem;text-align:center;margin-top:4rem}
.footer p{font-size:.8rem;color:var(--muted)}
@media(max-width:900px){.predict-layout{grid-template-columns:1fr}.result-panel{position:static}.ins-row{grid-template-columns:1fr}}
@media(max-width:600px){.fgrid{grid-template-columns:1fr}.hero-btns{flex-direction:column}.kpi-grid{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>

<nav class="nav">
  <div class="brand">&#9889; CarPrice<span>AI</span></div>
  <div class="nav-links">
    <a class="active" onclick="showPage('home',this)">Home</a>
    <a onclick="showPage('predict',this)">Predict</a>
    <a onclick="showPage('insights',this)">Model Insights</a>
    <a onclick="showPage('about',this)">About</a>
  </div>
</nav>

<!-- HOME -->
<div id="page-home" class="page active">
  <div class="hero">
    <div class="hero-content">
      <div class="hero-badge">Machine Learning &middot; Regression &middot; Real Data</div>
      <h1>AI-Powered<br/><span>Car Price</span> Prediction</h1>
      <p>Predict the estimated market price of a car using machine learning and vehicle specifications. Trained on 205 real automobile records.</p>
      <div class="hero-btns">
        <button class="btn btn-primary btn-lg" onclick="showPage('predict',document.querySelectorAll('.nav-links a')[1])">&#128663; Predict Car Price</button>
        <button class="btn btn-outline btn-lg" onclick="showPage('insights',document.querySelectorAll('.nav-links a')[2])">View Model Insights</button>
      </div>
    </div>
  </div>
  <div class="hero-stats">
    <div class="hs-item"><div class="hs-num">205</div><div class="hs-lbl">Dataset Records</div></div>
    <div class="hs-item"><div class="hs-num">24</div><div class="hs-lbl">Predictive Features</div></div>
    <div class="hs-item"><div class="hs-num">6</div><div class="hs-lbl">Models Compared</div></div>
    <div class="hs-item"><div class="hs-num" style="color:var(--accent2)">__R2__</div><div class="hs-lbl">Best R&#178; Score</div></div>
    <div class="hs-item"><div class="hs-num" style="color:var(--accent2);font-size:1rem">__BESTMODEL__</div><div class="hs-lbl">Best Model</div></div>
  </div>
</div>

<!-- PREDICT -->
<div id="page-predict" class="page">
  <div class="ph"><h1>Predict Car Price</h1><p>Enter vehicle specifications to get an AI-generated price estimate.</p></div>
  <div class="predict-layout">
    <form id="predictForm" novalidate>

      <div class="fsec">
        <div class="fsec-hdr"><div class="snum">1</div><div><h3>Car Information</h3><p>Brand, body style and drive config</p></div></div>
        <div class="fgrid">
          <div class="fg"><label>Manufacturer <span class="req">*</span></label>
            <select id="f_manufacturer"><option value="">Select manufacturer</option>__MFR__</select></div>
          <div class="fg"><label>Fuel Type <span class="req">*</span></label>
            <select id="f_fueltype"><option value="">Select fuel type</option>__FUEL__</select></div>
          <div class="fg"><label>Aspiration <span class="req">*</span></label>
            <select id="f_aspiration"><option value="">Select aspiration</option>__ASP__</select></div>
          <div class="fg"><label>Number of Doors <span class="req">*</span></label>
            <select id="f_doornumber"><option value="">Select doors</option>__DOOR__</select></div>
          <div class="fg"><label>Car Body <span class="req">*</span></label>
            <select id="f_carbody"><option value="">Select body style</option>__BODY__</select></div>
          <div class="fg"><label>Drive Wheel <span class="req">*</span></label>
            <select id="f_drivewheel"><option value="">Select drive wheel</option>__DRIVE__</select></div>
          <div class="fg"><label>Engine Location <span class="req">*</span></label>
            <select id="f_enginelocation"><option value="">Select location</option>__LOC__</select></div>
          <div class="fg"><label>Symboling (Risk) <span class="req">*</span></label>
            <input type="number" id="f_symboling" min="__SYM_MIN__" max="__SYM_MAX__" step="1" placeholder="__SYM_MEAN__"/></div>
        </div>
      </div>

      <div class="fsec">
        <div class="fsec-hdr"><div class="snum">2</div><div><h3>Engine Specifications</h3><p>Engine type, size and performance</p></div></div>
        <div class="fgrid">
          <div class="fg"><label>Engine Type <span class="req">*</span></label>
            <select id="f_enginetype"><option value="">Select engine type</option>__ETYPE__</select></div>
          <div class="fg"><label>Cylinders <span class="req">*</span></label>
            <select id="f_cylindernumber"><option value="">Select cylinders</option>__CYL__</select></div>
          <div class="fg"><label>Fuel System <span class="req">*</span></label>
            <select id="f_fuelsystem"><option value="">Select fuel system</option>__FSYS__</select></div>
          <div class="fg"><label>Engine Size (cc) <span class="req">*</span></label>
            <input type="number" id="f_enginesize" min="__ES_MIN__" max="__ES_MAX__" step="1" placeholder="__ES_MEAN__"/></div>
          <div class="fg"><label>Horsepower <span class="req">*</span></label>
            <input type="number" id="f_horsepower" min="__HP_MIN__" max="__HP_MAX__" step="1" placeholder="__HP_MEAN__"/></div>
          <div class="fg"><label>Peak RPM <span class="req">*</span></label>
            <input type="number" id="f_peakrpm" min="__RPM_MIN__" max="__RPM_MAX__" step="50" placeholder="__RPM_MEAN__"/></div>
          <div class="fg"><label>Bore Ratio <span class="req">*</span></label>
            <input type="number" id="f_boreratio" min="__BR_MIN__" max="__BR_MAX__" step="0.01" placeholder="__BR_MEAN__"/></div>
          <div class="fg"><label>Stroke <span class="req">*</span></label>
            <input type="number" id="f_stroke" min="__ST_MIN__" max="__ST_MAX__" step="0.01" placeholder="__ST_MEAN__"/></div>
          <div class="fg"><label>Compression Ratio <span class="req">*</span></label>
            <input type="number" id="f_compressionratio" min="__CR_MIN__" max="__CR_MAX__" step="0.1" placeholder="__CR_MEAN__"/></div>
        </div>
      </div>

      <div class="fsec">
        <div class="fsec-hdr"><div class="snum">3</div><div><h3>Dimensions &amp; Weight</h3><p>Physical measurements (inches &amp; lbs)</p></div></div>
        <div class="fgrid">
          <div class="fg"><label>Wheelbase (in) <span class="req">*</span></label>
            <input type="number" id="f_wheelbase" min="__WB_MIN__" max="__WB_MAX__" step="0.1" placeholder="__WB_MEAN__"/></div>
          <div class="fg"><label>Car Length (in) <span class="req">*</span></label>
            <input type="number" id="f_carlength" min="__CL_MIN__" max="__CL_MAX__" step="0.1" placeholder="__CL_MEAN__"/></div>
          <div class="fg"><label>Car Width (in) <span class="req">*</span></label>
            <input type="number" id="f_carwidth" min="__CW_MIN__" max="__CW_MAX__" step="0.1" placeholder="__CW_MEAN__"/></div>
          <div class="fg"><label>Car Height (in) <span class="req">*</span></label>
            <input type="number" id="f_carheight" min="__CH_MIN__" max="__CH_MAX__" step="0.1" placeholder="__CH_MEAN__"/></div>
          <div class="fg"><label>Curb Weight (lbs) <span class="req">*</span></label>
            <input type="number" id="f_curbweight" min="__CURB_MIN__" max="__CURB_MAX__" step="1" placeholder="__CURB_MEAN__"/></div>
        </div>
      </div>

      <div class="fsec">
        <div class="fsec-hdr"><div class="snum">4</div><div><h3>Fuel Efficiency</h3><p>Miles per gallon</p></div></div>
        <div class="fgrid">
          <div class="fg"><label>City MPG <span class="req">*</span></label>
            <input type="number" id="f_citympg" min="__CMPG_MIN__" max="__CMPG_MAX__" step="1" placeholder="__CMPG_MEAN__"/></div>
          <div class="fg"><label>Highway MPG <span class="req">*</span></label>
            <input type="number" id="f_highwaympg" min="__HMPG_MIN__" max="__HMPG_MAX__" step="1" placeholder="__HMPG_MEAN__"/></div>
        </div>
      </div>

      <div class="submit-area">
        <button type="submit" class="btn btn-primary btn-lg" id="predictBtn">
          <span id="btn-text">&#128269; Predict Price</span>
          <span id="btn-spin" class="hidden"><span class="spinner"></span> Analyzing...</span>
        </button>
        <button type="reset" class="btn btn-outline" onclick="resetResult()">Reset</button>
      </div>
      <div id="errBanner" class="alert-err hidden"></div>
    </form>

    <div class="result-panel">
      <div class="r-placeholder" id="rPlaceholder">
        <div class="ico">&#128663;</div>
        <h3>Ready to Predict</h3>
        <p>Fill in all fields and click <strong>Predict Price</strong>.</p>
      </div>
      <div class="r-card hidden" id="rCard">
        <div class="r-hdr"><span class="r-status">&#10003; Prediction Complete</span><span style="color:var(--dim);font-size:.75rem">USD</span></div>
        <div class="r-label">Estimated Car Price</div>
        <div class="r-price" id="rPrice">-</div>
        <div class="r-disc">Estimate based on historical dataset patterns. Not an actual market price.</div>
        <div class="meta-row"><span class="meta-k">Model Used</span><span class="meta-v" id="rModel">-</span></div>
        <div class="meta-row"><span class="meta-k">Model R&#178;</span><span class="meta-v">__R2__</span></div>
        <div class="meta-row"><span class="meta-k">Currency</span><span class="meta-v">US Dollar ($)</span></div>
        <div class="r-summary" id="rSummary"></div>
        <button class="btn btn-outline" style="width:100%;margin-top:.75rem;justify-content:center" onclick="resetResult()">New Prediction</button>
      </div>
    </div>
  </div>
</div>

<!-- INSIGHTS -->
<div id="page-insights" class="page">
  <div class="ph"><h1>Model Insights</h1><p>Performance metrics from the trained model.</p></div>
  <div class="kpi-grid">
    <div class="kpi hl"><div class="kpi-lbl">Best Model</div><div class="kpi-val">__BESTMODEL__</div></div>
    <div class="kpi"><div class="kpi-lbl">R&#178; Score</div><div class="kpi-val">__R2__</div></div>
    <div class="kpi"><div class="kpi-lbl">MAE</div><div class="kpi-val">$__MAE__</div></div>
    <div class="kpi"><div class="kpi-lbl">RMSE</div><div class="kpi-val">$__RMSE__</div></div>
    <div class="kpi"><div class="kpi-lbl">Records</div><div class="kpi-val">205</div></div>
    <div class="kpi"><div class="kpi-lbl">Features</div><div class="kpi-val">24</div></div>
  </div>
  <div class="ins-row">
    <div class="ins-card">
      <div class="card-title">Model Comparison</div>
      <table class="tbl"><thead><tr><th>Model</th><th>R&#178;</th><th>MAE</th><th>RMSE</th></tr></thead>
      <tbody>__MCROWS__</tbody></table>
    </div>
    <div class="ins-card">
      <div class="card-title">Feature Importance (Top 10)</div>
      __FIBARS__
    </div>
  </div>
  <div class="ins-row">
    <div class="ins-card">
      <div class="card-title">Price Distribution</div>
      __DISTBARS__
    </div>
    <div class="ins-card">
      <div class="card-title">Actual vs Predicted Prices</div>
      <canvas id="scatterCanvas" style="width:100%;max-height:300px"></canvas>
    </div>
  </div>
</div>

<!-- ABOUT -->
<div id="page-about" class="page">
  <div class="ph"><h1>About</h1><p>Methodology, architecture and limitations.</p></div>
  <div class="about-card">
    <h2>What Is This Application?</h2>
    <p><strong>CarPriceAI</strong> is a machine-learning car price prediction system. Given vehicle specifications, it generates an estimated selling price using a trained Random Forest regression model (R&#178;=0.9593).</p>
    <p>Trained on 205 real automobile records with 24 predictive features. This file is 100% standalone &mdash; no internet, no server, no installation needed.</p>
  </div>
  <div class="about-card">
    <h2>How It Works</h2>
    <div class="wf">
      <div class="wf-step"><div class="wf-n">1</div><div class="wf-b"><h4>User Input</h4><p>Vehicle specs: manufacturer, engine, dimensions, fuel efficiency.</p></div></div>
      <div class="wf-arr">&#8595;</div>
      <div class="wf-step"><div class="wf-n">2</div><div class="wf-b"><h4>Validation</h4><p>All required fields checked for completeness.</p></div></div>
      <div class="wf-arr">&#8595;</div>
      <div class="wf-step"><div class="wf-n">3</div><div class="wf-b"><h4>KNN Prediction Engine</h4><p>Weighted nearest-neighbor search over 205 training records using feature importances from the Random Forest model.</p></div></div>
      <div class="wf-arr">&#8595;</div>
      <div class="wf-step"><div class="wf-n">4</div><div class="wf-b"><h4>Result</h4><p>Predicted price shown instantly in your browser.</p></div></div>
    </div>
  </div>
  <div class="about-card">
    <h2>Limitations</h2>
    <ul class="method-list">
      <li>Trained on 205 records &mdash; a small dataset.</li>
      <li>Prices may not reflect current market values.</li>
      <li>Does not account for vehicle condition, mileage, or regional pricing.</li>
      <li>This file uses a nearest-neighbor approximation of the Random Forest model.</li>
    </ul>
  </div>
</div>

<footer class="footer">
  <p>CarPriceAI &mdash; Estimates based on historical data patterns. Not actual market prices.</p>
</footer>

<script>
const TRAINING_ROWS = __ROWS_JS__;
const META = __META_JS__;

function showPage(id, el) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  if (el) el.classList.add('active');
  window.scrollTo(0, 0);
  if (id === 'insights') setTimeout(drawScatter, 100);
}

const FEAT_WEIGHTS = __FI_WEIGHTS_JS__;
const NUM_FIELDS = ['symboling','wheelbase','carlength','carwidth','carheight',
  'curbweight','enginesize','boreratio','stroke','compressionratio',
  'horsepower','peakrpm','citympg','highwaympg'];
const CAT_FIELDS = ['manufacturer','fueltype','aspiration','doornumber',
  'carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem'];

const ranges = {};
NUM_FIELDS.forEach(f => {
  const vals = TRAINING_ROWS.map(r => r[f]);
  const mn = Math.min(...vals), mx = Math.max(...vals);
  ranges[f] = { min: mn, range: (mx - mn) || 1 };
});

function norm(f, v) { return (v - ranges[f].min) / ranges[f].range; }

function predict(input) {
  const scored = TRAINING_ROWS.map(row => {
    let dist = 0;
    NUM_FIELDS.forEach(f => {
      const w = FEAT_WEIGHTS[f] || 0.001;
      const d = norm(f, parseFloat(input[f])) - norm(f, row[f]);
      dist += w * d * d;
    });
    CAT_FIELDS.forEach(f => { if (input[f] !== row[f]) dist += 0.012; });
    return { dist, predicted: row.predicted };
  });
  scored.sort((a, b) => a.dist - b.dist);
  const k = 7, nearest = scored.slice(0, k);
  let ws = 0, ps = 0;
  nearest.forEach(n => { const w = 1 / (n.dist + 1e-6); ws += w; ps += w * n.predicted; });
  return Math.round(ps / ws * 100) / 100;
}

document.getElementById('predictForm').addEventListener('submit', function(e) {
  e.preventDefault();
  const err = document.getElementById('errBanner');
  err.classList.add('hidden');
  document.querySelectorAll('.err').forEach(el => el.classList.remove('err'));

  const input = {}, missing = [];
  CAT_FIELDS.forEach(f => {
    const el = document.getElementById('f_' + f);
    if (!el) return;
    const v = el.value.trim();
    if (!v) { missing.push(f); el.classList.add('err'); } else input[f] = v;
  });
  NUM_FIELDS.forEach(f => {
    const el = document.getElementById('f_' + f);
    if (!el) return;
    const v = el.value.trim();
    if (v === '' || isNaN(parseFloat(v))) { missing.push(f); el.classList.add('err'); }
    else input[f] = parseFloat(v);
  });

  if (missing.length > 0) {
    err.textContent = '⚠ Please fill in: ' + missing.slice(0,5).join(', ') + (missing.length > 5 ? ' and more...' : '');
    err.classList.remove('hidden');
    return;
  }

  const btn = document.getElementById('predictBtn');
  document.getElementById('btn-text').classList.add('hidden');
  document.getElementById('btn-spin').classList.remove('hidden');
  btn.disabled = true;

  setTimeout(() => {
    const price = predict(input);
    document.getElementById('rPlaceholder').classList.add('hidden');
    const card = document.getElementById('rCard');
    card.classList.remove('hidden');
    document.getElementById('rPrice').textContent = new Intl.NumberFormat('en-US',
      { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(price);
    document.getElementById('rModel').textContent = 'Random Forest (KNN approx.)';
    document.getElementById('rSummary').textContent =
      input.manufacturer + ' \u00b7 ' + input.carbody + ' \u00b7 ' + input.fueltype +
      ' \u00b7 ' + input.enginesize + 'cc \u00b7 ' + input.horsepower + 'hp \u00b7 ' +
      input.citympg + '/' + input.highwaympg + ' mpg';
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    document.getElementById('btn-text').classList.remove('hidden');
    document.getElementById('btn-spin').classList.add('hidden');
    btn.disabled = false;
  }, 500);
});

function resetResult() {
  document.getElementById('predictForm').reset();
  document.getElementById('rCard').classList.add('hidden');
  document.getElementById('rPlaceholder').classList.remove('hidden');
  document.getElementById('errBanner').classList.add('hidden');
}

function drawScatter() {
  const canvas = document.getElementById('scatterCanvas');
  if (!canvas || canvas._drawn) return;
  canvas._drawn = true;
  const W = canvas.offsetWidth || 400, H = 300;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const avp = META.actual_vs_predicted;
  const PAD = 48;
  const allV = avp.flatMap(d => [d.actual, d.predicted]);
  const minV = Math.min(...allV), maxV = Math.max(...allV);
  const sx = v => PAD + (v - minV) / (maxV - minV) * (W - PAD * 2);
  const sy = v => H - PAD - (v - minV) / (maxV - minV) * (H - PAD * 2);
  ctx.fillStyle = '#1e2533'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#2d3748'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const x = PAD + (W-PAD*2)*i/4, y = PAD + (H-PAD*2)*i/4;
    ctx.beginPath(); ctx.moveTo(x,PAD); ctx.lineTo(x,H-PAD); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD,y); ctx.lineTo(W-PAD,y); ctx.stroke();
  }
  ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 1.5; ctx.setLineDash([5,4]);
  ctx.beginPath(); ctx.moveTo(sx(minV),sy(minV)); ctx.lineTo(sx(maxV),sy(maxV)); ctx.stroke();
  ctx.setLineDash([]);
  avp.forEach(d => {
    ctx.beginPath(); ctx.arc(sx(d.actual), sy(d.predicted), 4.5, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(59,130,246,0.75)'; ctx.fill();
    ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 1; ctx.stroke();
  });
  ctx.fillStyle = '#8b949e'; ctx.font = '11px Segoe UI'; ctx.textAlign = 'center';
  ctx.fillText('Actual Price', W/2, H-6);
  ctx.save(); ctx.translate(13, H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Predicted Price', 0, 0); ctx.restore();
}
</script>
</body>
</html>"""

# ── Substitutions ────────────────────────────────────────────────
def o(key, tfm='cap'):
    items = dd.get(key, [])
    fn = str.upper if tfm == 'upper' else str.capitalize
    return '\n'.join(f'<option value="{v}">{fn(v)}</option>' for v in items)

def g(key, attr, fmt='.2f'):
    v = nr.get(key, {}).get(attr, 0)
    if fmt == 'int': return str(int(v))
    return f'{v:{fmt}}'

subs = {
    '__R2__':        str(bm['r2']),
    '__BESTMODEL__': m['best_model'],
    '__MAE__':       f"{bm['mae']:,.0f}",
    '__RMSE__':      f"{bm['rmse']:,.0f}",
    '__MCROWS__':    mc_rows,
    '__FIBARS__':    fi_bars,
    '__DISTBARS__':  dist_bars,
    '__ROWS_JS__':   rows_js,
    '__META_JS__':   meta_js,
    '__FI_WEIGHTS_JS__': fi_weights_js,
    '__MFR__':   o('manufacturer'),
    '__FUEL__':  o('fueltype', 'upper'),
    '__ASP__':   o('aspiration'),
    '__DOOR__':  o('doornumber'),
    '__BODY__':  o('carbody'),
    '__DRIVE__': o('drivewheel', 'upper'),
    '__LOC__':   o('enginelocation'),
    '__ETYPE__': o('enginetype', 'upper'),
    '__CYL__':   o('cylindernumber'),
    '__FSYS__':  o('fuelsystem', 'upper'),
    '__SYM_MIN__':  g('symboling','min','int'), '__SYM_MAX__': g('symboling','max','int'), '__SYM_MEAN__': g('symboling','mean','int'),
    '__ES_MIN__':   g('enginesize','min','int'),  '__ES_MAX__': g('enginesize','max','int'),  '__ES_MEAN__':  g('enginesize','mean','int'),
    '__HP_MIN__':   g('horsepower','min','int'),  '__HP_MAX__': g('horsepower','max','int'),  '__HP_MEAN__':  g('horsepower','mean','int'),
    '__RPM_MIN__':  g('peakrpm','min','int'),     '__RPM_MAX__': g('peakrpm','max','int'),    '__RPM_MEAN__': g('peakrpm','mean','int'),
    '__BR_MIN__':   g('boreratio','min'),  '__BR_MAX__': g('boreratio','max'),  '__BR_MEAN__': g('boreratio','mean'),
    '__ST_MIN__':   g('stroke','min'),     '__ST_MAX__': g('stroke','max'),     '__ST_MEAN__': g('stroke','mean'),
    '__CR_MIN__':   g('compressionratio','min'), '__CR_MAX__': g('compressionratio','max'), '__CR_MEAN__': g('compressionratio','mean'),
    '__WB_MIN__':   g('wheelbase','min'),  '__WB_MAX__': g('wheelbase','max'),  '__WB_MEAN__': g('wheelbase','mean'),
    '__CL_MIN__':   g('carlength','min'),  '__CL_MAX__': g('carlength','max'),  '__CL_MEAN__': g('carlength','mean'),
    '__CW_MIN__':   g('carwidth','min'),   '__CW_MAX__': g('carwidth','max'),   '__CW_MEAN__': g('carwidth','mean'),
    '__CH_MIN__':   g('carheight','min'),  '__CH_MAX__': g('carheight','max'),  '__CH_MEAN__': g('carheight','mean'),
    '__CURB_MIN__': g('curbweight','min','int'), '__CURB_MAX__': g('curbweight','max','int'), '__CURB_MEAN__': g('curbweight','mean','int'),
    '__CMPG_MIN__': g('citympg','min','int'),    '__CMPG_MAX__': g('citympg','max','int'),    '__CMPG_MEAN__': g('citympg','mean','int'),
    '__HMPG_MIN__': g('highwaympg','min','int'), '__HMPG_MAX__': g('highwaympg','max','int'), '__HMPG_MEAN__': g('highwaympg','mean','int'),
}

for k, v in subs.items():
    HTML = HTML.replace(k, v)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(HTML)

print(f"index.html written: {len(HTML):,} chars ({len(HTML)//1024} KB)")
