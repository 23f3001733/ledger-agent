import os, re, json, time, datetime, threading
import requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# ---------- settings (from Render environment variables) ----------
ORDERS_URL = os.environ["ORDERS_URL"]
RATES_URL = os.environ.get("RATES_URL", "")
MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
client = OpenAI(
    api_key=os.environ["LLM_KEY"],
    base_url=os.environ.get("LLM_BASE") or "https://api.openai.com/v1",
    timeout=10, max_retries=1,
)
TZ = "Asia/Kolkata"
DEFAULT_RATES = {"USD": 1.0, "EUR": 1.05, "INR": 0.01111}

DATA, INFO = None, {}
READY = threading.Event()


def get_json(url, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1 + i)


def page_url(p):
    if re.search(r"([?&])page=\d+", ORDERS_URL):
        return re.sub(r"([?&])page=\d+", r"\g<1>page=%d" % p, ORDERS_URL)
    return ORDERS_URL + "&page=%d" % p


# ---------- 1. load + clean data ----------
def build():
    global DATA, INFO
    rows, n_pages = None, 0
    export_url = os.environ.get("EXPORT_URL", "")
    if export_url:
        try:
            txt = requests.get(export_url, timeout=30).text
            rows = [json.loads(l) for l in txt.splitlines() if l.strip()]
            n_pages = -1
        except Exception:
            rows = None
    if not rows:
        first = get_json(page_url(1))
        rows = list(first["orders"])
        n_pages = int(first.get("pages", 1))
        if n_pages > 1:
            with ThreadPoolExecutor(8) as ex:
                for d in ex.map(lambda p: get_json(page_url(p)), range(2, n_pages + 1)):
                    rows += d["orders"]
    df = pd.DataFrame(rows)
    raw_rows = len(df)

    # latest updated_at wins (compare as real moments, not text)
    df["_u"] = pd.to_datetime(df["updated_at"], utc=True, format="ISO8601")
    df = df.sort_values("_u", kind="stable").drop_duplicates("id", keep="last")

    # rates: USD value of one unit of each currency
    rates = dict(DEFAULT_RATES)
    if RATES_URL:
        try:
            r = get_json(RATES_URL)
            rates = {k.upper(): float(v) for k, v in r["usd_per_unit"].items()}
        except Exception:
            pass

    out = pd.DataFrame(index=df.index)
    out["id"] = df["id"]
    out["date"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601").dt.tz_convert(TZ)
    out["status"] = df["status"].astype(str).str.strip().str.lower()
    out["region"] = df["region"].astype(str).str.strip()
    out["product"] = df["product"].astype(str).str.strip()
    out["customer"] = df["customer"].astype(str).str.strip()
    out["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
    cur = df["currency"].astype(str).str.strip().str.upper()
    mult = cur.map(rates)
    out["amount_usd"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0) * mult.fillna(0)

    DATA = out
    INFO = {
        "raw_rows": raw_rows,
        "rows_after_dedupe": len(out),
        "pages": n_pages,
        "rates": rates,
        "unmapped_currency_rows": int(mult.isna().sum()),
        "amount_vs_qty_x_price_mismatches": int(
            ((pd.to_numeric(df["amount"], errors="coerce")
              - pd.to_numeric(df["qty"], errors="coerce") * pd.to_numeric(df["unit_price"], errors="coerce")).abs() > 0.011).sum()),
        "statuses": sorted(out["status"].unique().tolist()),
        "regions": sorted(out["region"].unique().tolist()),
        "products": sorted(out["product"].unique().tolist()),
        "n_customers": int(out["customer"].nunique()),
        "date_range_IST": [str(out["date"].min()), str(out["date"].max())],
    }


def load():
    global INFO
    try:
        build()
    except Exception as e:
        INFO = {"startup_error": str(e)}
    finally:
        READY.set()


threading.Thread(target=load, daemon=True).start()


# ---------- 2. LLM: question -> small JSON plan ----------
def system_prompt():
    today = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).date()
    return f"""You convert a question about an order ledger into JSON. Output ONLY this JSON object:
{{"metric": "revenue|orders|units|aov|distinct_customers|distinct_products|refund_total|refund_count",
 "status": null|"paid"|"refunded"|"void"|"all",
 "group_by": null|"product"|"customer"|"region"|"month",
 "rank": null|"max"|"min",
 "top_n": null|integer,
 "region": null|string, "product": null|string, "customer": null|string,
 "start": null|"YYYY-MM-DD", "end": null|"YYYY-MM-DD"}}

Rules:
- Dates are inclusive. "March 2026" => start 2026-03-01, end 2026-03-31. "Q1 2026" => 2026-01-01..2026-03-31. Today is {today}.
- revenue / sales / income / turnover => metric revenue (status null).
- refunds / money returned / refunded amount => metric refund_total. "how many refunds / refunded orders" => refund_count.
- number of orders => orders. units / quantity sold => units. average order value => aov.
- best seller / top product / product with most sales => metric revenue, group_by product, rank max, top_n 1. If they say "units" or "quantity" use metric units.
- top customer / biggest spender => metric revenue, group_by customer, rank max, top_n 1.
- "how many customers" => distinct_customers. "how many different products" => distinct_products.
- "top 3 ..." => top_n 3. Lowest / least / worst => rank min.
- Use the exact spelling from the lists below when a name is mentioned. Customer ids look like C1011.
Regions: {", ".join(INFO.get("regions", []))}
Products: {", ".join(INFO.get("products", []))}"""


def normalize(p):
    m = str(p.get("metric") or "").lower()
    if "refund" in m:
        m = "refund_count" if ("count" in m or "number" in m or "num" in m) else "refund_total"
    elif "customer" in m:
        m = "distinct_customers"
    elif "product" in m:
        m = "distinct_products"
    elif "aov" in m or "average" in m or "avg" in m:
        m = "aov"
    elif "unit" in m or "qty" in m or "quantity" in m:
        m = "units"
    elif "order" in m or "count" in m:
        m = "orders"
    else:
        m = "revenue"
    p["metric"] = m
    if m.startswith("distinct"):
        p["group_by"] = None
        p["rank"] = None
        p["top_n"] = None
    return p


def make_plan(q):
    last = None
    for _ in range(2):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system_prompt()},
                          {"role": "user", "content": q}])
            return normalize(json.loads(r.choices[0].message.content))
        except Exception as e:
            last = e
            time.sleep(0.5)
    raise last


# ---------- 3. plain code does the maths ----------
COUNT_METRICS = ("orders", "units", "distinct_customers", "distinct_products")


def agg(df, m):
    if m == "revenue":            return df["amount_usd"].sum()
    if m == "orders":             return len(df)
    if m == "units":              return df["qty"].sum()
    if m == "aov":                return df["amount_usd"].mean() if len(df) else 0.0
    if m == "distinct_customers": return df["customer"].nunique()
    if m == "distinct_products":  return df["product"].nunique()
    raise ValueError("unknown metric " + str(m))


def clean(v, m):
    return int(round(float(v))) if m in COUNT_METRICS else round(float(v), 2)


def run(p):
    d = DATA
    m = p.get("metric")
    st = (p.get("status") or "").lower()
    if m == "refund_total":
        m, st = "revenue", (st or "refunded")
    elif m == "refund_count":
        m, st = "orders", (st or "refunded")
    if not st:
        st = "paid"
    if st != "all":
        d = d[d["status"] == st]
    for col in ("region", "product", "customer"):
        if p.get(col):
            d = d[d[col].str.lower() == str(p[col]).strip().lower()]
    if p.get("start"):
        d = d[d["date"] >= pd.Timestamp(p["start"], tz=TZ)]
    if p.get("end"):
        d = d[d["date"] < pd.Timestamp(p["end"], tz=TZ) + pd.Timedelta(days=1)]

    g = p.get("group_by")
    if not g:
        return clean(agg(d, m), m)
    key = d["date"].dt.strftime("%Y-%m") if g == "month" else d[g]
    s = pd.Series({k: agg(grp, m) for k, grp in d.groupby(key)}, dtype=float)
    rank = p.get("rank") or ("max" if p.get("top_n") else None)
    if rank:
        s = s.sort_values(ascending=(rank == "min"), kind="stable")
        n = int(p.get("top_n") or 1)
        names = [str(x) for x in s.index[:n]]
        return names[0] if n == 1 else names
    return {str(k): clean(v, m) for k, v in s.items()}


# ---------- 4. web app ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def health():
    return {"status": "ok", "rows": INFO.get("rows_after_dedupe"), "ready": READY.is_set()}


@app.get("/debug")
def debug():
    return INFO


@app.get("/reload")
def reload():
    build()
    return {"ok": True, "rows": INFO.get("rows_after_dedupe")}


@app.post("/")
@app.post("/ask")
def answer(body: dict):
    try:
        READY.wait(60)
        if DATA is None:
            build()
        return {"answer": run(make_plan(body["question"]))}
    except Exception as e:
        return {"answer": None, "error": str(e)}
