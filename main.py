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
ORDERS_URL = os.environ.get("ORDERS_URL")
RATES_URL = os.environ.get("RATES_URL")
LLM_KEY = os.environ.get("LLM_KEY")
LLM_BASE = os.environ.get("LLM_BASE", "https://aipipe.org/openai/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")

DB_PATH = "ledger.db"

client = OpenAI(api_key=LLM_KEY, base_url=LLM_BASE)

SYSTEM_PROMPT = """
You are an expert SQLite query generator. Given a natural language question about an order ledger, output ONLY a single valid SQL query.

Database Schema (Table: `orders`):
- order_id (TEXT): Unique ID of the order.
- status (TEXT): Order status e.g., 'paid', 'refunded', 'pending'. Always use LOWER(status) when comparing.
- region (TEXT): Region name e.g., 'Central', 'North', 'South', 'East', 'West'. Always use LOWER(region) when comparing.
- created_at (TEXT): Raw timestamp string.
- created_date (TEXT): Date in 'YYYY-MM-DD' format (Asia/Kolkata timezone).
- created_year_month (TEXT): Date in 'YYYY-MM' format e.g., '2026-02' (Asia/Kolkata timezone).
- amount (REAL): Monetary amount in original currency.
- currency (TEXT): Currency code e.g., 'USD', 'INR', 'EUR'.
- product_name (TEXT): Name of the product.
- customer_id (TEXT): Customer identifier.
- usd_amount (REAL): Monetary amount pre-converted to USD.

Database Schema (Table: `rates`):
- currency (TEXT): Currency code.
- rate_to_usd (REAL): Conversion multiplier to USD.

Rules for Query Generation:
1. ALWAYS wrap aggregate results with COALESCE to prevent null returns:
   - Use `SELECT COALESCE(COUNT(*), 0)` instead of `SELECT COUNT(*)`.
   - Use `SELECT COALESCE(SUM(usd_amount), 0.0)` for revenue / monetary questions.
2. Filter status and region case-insensitively using LOWER():
   - e.g., `LOWER(status) = 'refunded'` AND `LOWER(region) = 'central'`
3. Revenue queries strictly require `LOWER(status) = 'paid'`.
4. For month/year questions (e.g., "February 2026"), filter using `created_year_month = '2026-02'`.
5. Return ONLY the raw SQL query enclosed inside ```sql ... ``` code blocks.
"""

def fetch_all_orders():
    all_orders = []
    page = 1
    base_url = ORDERS_URL.split("&page=")[0] if ORDERS_URL else ""
    
    while True:
        url = f"{base_url}&page={page}"
        try:
            res = requests.get(url, timeout=15)
            if res.status_code != 200:
                break
            data = res.json()
            orders = data.get("orders", data.get("data", []))
            if not orders:
                break
            all_orders.extend(orders)
            
            has_next = data.get("has_next", False) or page < data.get("total_pages", 1)
            if not has_next:
                break
            page += 1
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
        
    return pd.DataFrame(all_orders)

def sync_database():
    if not ORDERS_URL or not RATES_URL:
        print("Missing ORDERS_URL or RATES_URL environment variables.")
        return

    print("Fetching rates...")
    df_rates = pd.DataFrame(columns=["currency", "rate_to_usd"]) # Pre-initialize columns to prevent SQL syntax crash
    try:
        rates_res = requests.get(RATES_URL, timeout=15).json()
        rates_data = rates_res.get("rates", {})
        
        if isinstance(rates_data, dict) and rates_data:
            df_rates = pd.DataFrame([{"currency": k, "rate_to_usd": v} for k, v in rates_data.items()])
        elif isinstance(rates_data, list) and rates_data:
            df_rates = pd.DataFrame(rates_data)
    except Exception as e:
        print(f"Failed to fetch rates: {e}")

    # Safety check: Force columns to exist even if API returned empty
    for col in ["currency", "rate_to_usd"]:
        if col not in df_rates.columns:
            df_rates[col] = pd.Series(dtype='float64' if col == 'rate_to_usd' else 'object')

    print("Fetching orders...")
    df_orders = fetch_all_orders()
    
    # Pre-initialize order schema to prevent SQL syntax crash if API is completely empty
    expected_order_cols = [
        "order_id", "status", "region", "created_at", "updated_at",
        "amount", "currency", "product_name", "customer_id", 
        "created_date", "created_year_month", "usd_amount", "updated_at_dt"
    ]

    if df_orders.empty or len(df_orders.columns) == 0:
        print("No orders fetched. Creating empty schema database.")
        df_orders = pd.DataFrame(columns=expected_order_cols)
    else:
        # Normalize ID field
        if "order_id" not in df_orders.columns and "id" in df_orders.columns:
            df_orders["order_id"] = df_orders["id"]

        # Parse timezones with utc=True safely
        df_orders["updated_at_dt"] = pd.to_datetime(df_orders["updated_at"], utc=True, errors="coerce")
        df_orders = df_orders.sort_values("updated_at_dt").groupby("order_id").last().reset_index()

        created_dt = pd.to_datetime(df_orders["created_at"], utc=True, errors="coerce").dt.tz_convert("Asia/Kolkata")
        df_orders["created_date"] = created_dt.dt.strftime("%Y-%m-%d")
        df_orders["created_year_month"] = created_dt.dt.strftime("%Y-%m")

        # Map USD amounts
        rate_map = df_rates.set_index("currency")["rate_to_usd"].to_dict() if not df_rates.empty else {}
        df_orders["usd_amount"] = df_orders.apply(
            lambda r: float(r.get("amount", 0)) * float(rate_map.get(r.get("currency"), 1.0)), axis=1
        )

        df_orders["updated_at_dt"] = df_orders["updated_at_dt"].astype(str)
        
        # Ensure any missing columns exist as nulls
        for col in expected_order_cols:
            if col not in df_orders.columns:
                df_orders[col] = None

    # Persist into SQLite safely
    try:
        conn = sqlite3.connect(DB_PATH)
        df_orders.to_sql("orders", conn, if_exists="replace", index=False)
        df_rates.to_sql("rates", conn, if_exists="replace", index=False)
        conn.close()
        print("Database sync completed successfully.")
    except Exception as e:
