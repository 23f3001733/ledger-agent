import os, json, requests, pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# ---------- settings (come from Render's Environment Variables) ----------
LEDGER_URL = os.environ["LEDGER_URL"]          # your long URL WITHOUT "&path=..."
client = OpenAI(
    api_key=os.environ["LLM_KEY"],
    base_url=os.environ.get("LLM_BASE", "https://api.openai.com/v1"),
)
MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")

# ---------- 1. load all data once, at startup ----------
def fetch_all(path):
    rows = []
    url = f"{LEDGER_URL}&path={path}"
    while url:
        r = requests.get(url, timeout=10).json()
        rows += r["items"]                      # TODO: the real key holding the list
        url = r.get("next")                     # TODO: the real "next page" key
        # If next is just a path (e.g. "/orders?page=2"), build it:
        # url = f"{LEDGER_URL}&path={r['next']}" if r.get("next") else None
    return rows

orders = pd.DataFrame(fetch_all("/orders"))     # TODO: real path
refunds = pd.DataFrame(fetch_all("/refunds"))   # TODO: real path (or delete if none)

# TODO: clean the data, based on what you saw. Typical examples:
# orders["date"] = pd.to_datetime(orders["date"])
# orders["amount_usd"] = orders["amount_cents"] / 100
# orders = orders.drop_duplicates("order_id")

# ---------- 2. LLM turns the question into a "recipe" ----------
SYSTEM = """You convert questions about an order ledger into JSON. Output ONLY JSON:
{"metric": "revenue" | "order_count" | "refund_total" | "refund_count" | "top_product" | "customer_count",
 "region": string or null,
 "product": string or null,
 "start": "YYYY-MM-DD" or null,
 "end": "YYYY-MM-DD" or null}
Dates are inclusive. "March 2026" -> start 2026-03-01, end 2026-03-31.
Regions allowed: North, South, East, West.   # TODO: your real values
"""

def make_plan(question):
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": question}],
    )
    return json.loads(resp.choices[0].message.content)

# ---------- 3. plain code does the maths ----------
def apply_filters(df, p):
    if p.get("region"):
        df = df[df["region"].str.lower() == p["region"].lower()]     # TODO: column name
    if p.get("product"):
        df = df[df["product"].str.lower() == p["product"].lower()]   # TODO: column name
    if p.get("start"):
        df = df[df["date"] >= pd.Timestamp(p["start"])]
    if p.get("end"):
        df = df[df["date"] < pd.Timestamp(p["end"]) + pd.Timedelta(days=1)]
    return df

def run(p):
    m = p["metric"]
    if m == "revenue":
        return round(float(apply_filters(orders, p)["amount_usd"].sum()), 2)
    if m == "order_count":
        return int(len(apply_filters(orders, p)))
    if m == "refund_total":
        return round(float(apply_filters(refunds, p)["amount_usd"].sum()), 2)
    if m == "refund_count":
        return int(len(apply_filters(refunds, p)))
    if m == "top_product":
        return apply_filters(orders, p).groupby("product")["amount_usd"].sum().idxmax()
    if m == "customer_count":
        return int(apply_filters(orders, p)["customer_id"].nunique())  # TODO: column name
    raise ValueError(f"unknown metric {m}")

# ---------- 4. the web endpoint ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/")
def answer(body: dict):
    try:
        plan = make_plan(body["question"])
        return {"answer": run(plan)}
    except Exception as e:
        return {"answer": None, "error": str(e)}
