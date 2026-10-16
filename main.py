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

    df_orders = pd.DataFrame(all
