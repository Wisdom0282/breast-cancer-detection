"""
Web app for the breast cancer classifier (Flask). Vercel looks for `app` in app.py.

Local:   python app.py   -> http://127.0.0.1:5000
API:     POST /predict   {"features": [30 numbers in WDBC column order]}
FOR EDUCATIONAL USE ONLY. Not a medical device.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
BUNDLE = joblib.load(BASE / "models" / "breast_cancer_model.joblib")
MODEL = BUNDLE["model"]
FEATURES = BUNDLE["feature_names"]
THRESHOLD = BUNDLE["threshold"]
SAMPLES = json.loads((BASE / "samples.json").read_text())["samples"]

app = Flask(__name__)


def predict_row(values):
    if not isinstance(values, (list, tuple)) or len(values) != len(FEATURES):
        raise ValueError(f"Expected a list of {len(FEATURES)} numbers.")
    x = np.array([[float(v) for v in values]])
    if not np.isfinite(x).all():
        raise ValueError("All values must be finite numbers.")
    p = float(MODEL.predict_proba(x)[0, 1])
    return {"p_malignant": round(p, 4),
            "prediction": "Malignant" if p >= THRESHOLD else "Benign"}


@app.get("/")
def index():
    return PAGE.replace("__FEATURES__", json.dumps(FEATURES)).replace("__SAMPLES__", json.dumps(SAMPLES))


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(predict_row(data.get("features")))
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.get("/health")
def health():
    return jsonify({"status": "ok", "n_features": len(FEATURES)})


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Breast Cancer Classifier</title>
<style>
 :root{--bg:#f7f7f5;--fg:#1c1c1c;--muted:#6b6b6b;--card:#fff;--line:#e3e3df;--accent:#2f5d8a;--bad:#b3261e;--ok:#1e7a46}
 @media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--muted:#9a9a9a;--card:#1f1f1f;--line:#333;--accent:#7fb0e0;--bad:#f2766d;--ok:#5fcf8f}}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
 main{max-width:920px;margin:0 auto;padding:24px 16px 48px}
 h1{font-size:1.5rem;margin:0 0 4px} .sub{color:var(--muted);margin:0 0 16px}
 .warn{border:1px solid var(--line);background:var(--card);padding:10px 12px;border-radius:8px;font-size:.9rem;color:var(--muted);margin-bottom:16px}
 .bar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
 button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
 button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
 label{display:block;font-size:.78rem;color:var(--muted)} input{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);font:inherit}
 fieldset{border:1px solid var(--line);border-radius:10px;margin:0 0 14px;padding:12px;background:var(--card)} legend{padding:0 6px;font-weight:600}
 #result{margin-top:16px;padding:16px;border-radius:10px;border:1px solid var(--line);background:var(--card);display:none}
 .big{font-size:1.4rem;font-weight:700} .err{color:var(--bad)}
</style></head><body><main>
<h1>Breast Cancer Classifier</h1>
<p class="sub">SVM trained on the Wisconsin Diagnostic dataset (30 FNA image features).</p>
<div class="warn">Educational demo only. Not a medical device and not for diagnosis.</div>
<div class="bar">
 <button type="button" id="sb">Load benign example</button>
 <button type="button" id="sm">Load malignant example</button>
 <button type="button" id="clr">Clear</button>
 <button type="button" class="primary" id="go">Predict</button>
</div>
<form id="f" onsubmit="return false"></form>
<div id="result"></div>
<script>
const F=__FEATURES__, S=__SAMPLES__;
const groups={"Mean values":F.filter(n=>n.startsWith("mean ")),"Standard error":F.filter(n=>n.endsWith(" error")),"Worst values":F.filter(n=>n.startsWith("worst "))};
const form=document.getElementById("f");
for(const [g,names] of Object.entries(groups)){
  const fs=document.createElement("fieldset"),lg=document.createElement("legend");lg.textContent=g;fs.appendChild(lg);
  const grid=document.createElement("div");grid.className="grid";
  names.forEach(n=>{const d=document.createElement("div");d.innerHTML=`<label>${n}</label><input type="number" step="any" data-f="${n}">`;grid.appendChild(d)});
  fs.appendChild(grid);form.appendChild(fs);
}
const inputs=()=>Object.fromEntries([...document.querySelectorAll("input[data-f]")].map(i=>[i.dataset.f,i]));
function fill(arr){const m=inputs();F.forEach((n,i)=>m[n].value=arr?arr[i]:"")}
document.getElementById("sb").onclick=()=>fill(S.benign);
document.getElementById("sm").onclick=()=>fill(S.malignant);
document.getElementById("clr").onclick=()=>{fill(null);document.getElementById("result").style.display="none"};
document.getElementById("go").onclick=async()=>{
  const m=inputs(),r=document.getElementById("result");r.style.display="block";
  const vals=F.map(n=>m[n].value===""?NaN:Number(m[n].value));
  if(vals.some(Number.isNaN)){r.innerHTML='<span class="err">Fill in all 30 values, or load an example.</span>';return}
  try{
    const res=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({features:vals})});
    const d=await res.json();
    if(d.error){r.innerHTML=`<span class="err">${d.error}</span>`;return}
    const mal=d.prediction==="Malignant";
    r.innerHTML=`<div class="big" style="color:var(${mal?'--bad':'--ok'})">${d.prediction}</div><div>Probability of malignancy: ${(d.p_malignant*100).toFixed(1)}%</div>`;
  }catch(e){r.innerHTML='<span class="err">Request failed.</span>'}
};
</script></main></body></html>"""


if __name__ == "__main__":
    app.run(debug=True)
