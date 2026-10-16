import os
import re
import sqlite3
import pandas as pd
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# Environment Variables
ORDERS_URL = os.environ.get("ORDERS_URL", "")
RATES_URL = os.environ.get("RATES_URL", "")
LLM_KEY = os.environ.get("LLM_KEY", "")
LLM_BASE = os.environ.get("LLM_BASE", "https://aipipe.org/openai/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")

DB_PATH = "ledger.db"

client = OpenAI(api_key=LLM_KEY, base_url=LLM_BASE)

SYSTEM_PROMPT = """
You are an expert SQLite query generator. Given a natural language question about an order ledger, output ONLY a single valid SQL query.

Database Schema (Table: `orders`):
- order_id (TEXT): Unique ID.
- status (TEXT): 'paid', 'refunded', 'pending'.
- region (TEXT): 'Central', 'North', 'South', 'East', 'West'.
- created_date (TEXT): Date in 'YYYY-MM-DD' format (Asia/Kolkata timezone).
- created_year_month (TEXT): Date in 'YYYY-MM' format e.g., '2026-02'.
- amount (REAL): Monetary amount in original currency.
- currency (TEXT): Currency code.
- product_name (TEXT): Name of the product.
- customer_id (TEXT): Customer identifier.
- usd_amount (REAL): Amount converted to USD.

Rules for Query Generation:
1. ALWAYS wrap aggregate results with COALESCE to prevent nulls: SELECT COALESCE(SUM(usd_amount), 0.0) or SELECT COALESCE(COUNT(*), 0).
2. For "how many different customers" or "unique customers", ALWAYS use: SELECT COALESCE(COUNT(DISTINCT customer_id), 0).
3. Filter ALL strings case-insensitively using LOWER() and LIKE to avoid exact-match errors:
   - LOWER(status) = 'paid'
   - LOWER(region) = 'central'
   - LOWER(product_name) LIKE '%rice cooker%'
4. Revenue queries strictly require LOWER(status) = 'paid'.
5. For month/year questions (e.g., "February 2026"), filter using created_year_month = '2026-02'.
6. Return ONLY the raw SQL query enclosed inside ```sql ... ``` code blocks.
"""

def sync_database():
    print("Starting database sync...")
    
    # 1. Fetch Rates Safely
    rate_cols = ["currency", "rate_to_usd"]
    df_rates = pd.DataFrame(columns=rate_cols)
    if RATES_URL:
        try:
            res = requests.get(RATES_URL, timeout=15)
            if res.status_code == 200:
                rates_data = res.json().get("rates", {})
                if isinstance(rates_data, dict) and rates_data:
                    df_rates = pd.DataFrame([{"currency": k, "rate_to_usd": float(v)} for k, v in rates_data.items()])
                elif isinstance(rates_data, list) and rates_data:
                    df_rates = pd.DataFrame(rates_data)
        except Exception as e:
            print("Error fetching rates:", e)

    # Force rate columns to exist to prevent SQL crash
    for col in rate_cols:
        if col not in df_rates.columns:
            df_rates[col] = pd.Series(dtype="float64" if col == "rate_to_usd" else "object")

    # 2. Fetch Orders Safely
    all_orders = []
    if ORDERS_URL:
        base_url = ORDERS_URL.split("&page=")[0]
        page = 1
        while True:
            try:
                res = requests.get(f"{base_url}&page={page}", timeout=15)
                if res.status_code != 200:
                    break
                data = res.json()
                orders = data.get("orders", data.get("data", []))
                if not orders:
                    break
                all_orders.extend(orders)
                if not (data.get("has_next", False) or page < data.get("total_pages", 1)):
                    break
                page += 1
            except Exception as e:
                print("Error fetching orders:", e)
                break

    df_orders = pd.DataFrame(all_orders)
    
    # Define expected columns so empty data doesn't crash SQLite
    order_cols = [
        "order_id", "status", "region", "created_at", "updated_at",
        "amount", "currency", "product_name", "customer_id", 
        "created_date", "created_year_month", "usd_amount", "updated_at_dt"
    ]

    if df_orders.empty:
        df_orders = pd.DataFrame(columns=order_cols)
    else:
        # Standardize ID
        if "order_id" not in df_orders.columns and "id" in df_orders.columns:
            df_orders["order_id"] = df_orders["id"]

        # Parse timezone securely and deduplicate
        df_orders["updated_at_dt"] = pd.to_datetime(df_orders.get("updated_at"), utc=True, errors="coerce")
        df_orders = df_orders.sort_values("updated_at_dt").groupby("order_id").last().reset_index()

        # Convert to Asia/Kolkata Timezone
        created_dt = pd.to_datetime(df_orders.get("created_at"), utc=True, errors="coerce").dt.tz_convert("Asia/Kolkata")
        df_orders["created_date"] = created_dt.dt.strftime("%Y-%m-%d")
        df_orders["created_year_month"] = created_dt.dt.strftime("%Y-%m")

        # Map USD values
        rate_map = df_rates.set_index("currency")["rate_to_usd"].to_dict() if not df_rates.empty else {}
        
        def calc_usd(row):
            amt = float(row.get("amount", 0.0) or 0.0)
            rate = float(rate_map.get(row.get("currency"), 1.0))
            return amt * rate
            
        df_orders["usd_amount"] = df_orders.apply(calc_usd, axis=1)
        df_orders["updated_at_dt"] = df_orders["updated_at_dt"].astype(str)

        # Ensure all columns exist before SQL write
        for col in order_cols:
            if col not in df_orders.columns:
                df_orders[col] = None

    # 3. Persist to SQLite
    try:
        conn = sqlite3.connect(DB_PATH)
        df_orders.to_sql("orders", conn, if_exists="replace", index=False)
        df_rates.to_sql("rates", conn, if_exists="replace", index=False)
        conn.close()
        print("Database sync completed successfully.")
    except Exception as e:
        print("Failed to save to SQLite:", e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    sync_database()
    yield

app = FastAPI(lifespan=lifespan)

class QuestionRequest(BaseModel):
    question: str

def execute_sql_query(query: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        conn.close()
        
        if result is None or result[0] is None:
            return 0
        val = result[0]
        return round(float(val), 2) if isinstance(val, (float, int)) and not isinstance(val, bool) else val
    except Exception as e:
        print("SQL Error:", e, "\nQuery was:", query)
        return 0

@app.post("/")
async def answer_question(payload: QuestionRequest):
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload.question}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        match = re.search(r"```sql\s*(.*?)\s*```", content, re.DOTALL)
        sql_query = match.group(1).strip() if match else content.strip()
        
        answer = execute_sql_query(sql_query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
