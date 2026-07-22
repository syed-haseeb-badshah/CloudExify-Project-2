# generate_data.py
# Creates the two synthetic datasets used in the Data Science projects.
# I built these to look like real Pakistani retail sales so the analysis
# has meaningful patterns (seasonal trends, regional differences, and
# customer groups). A fixed random seed keeps the numbers reproducible.

import numpy as np
import pandas as pd
from pathlib import Path

# Both CSVs are written into sample_data/ so the repo stays tidy.
OUT = Path(__file__).parent / "sample_data"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# Dataset 1: sales_data.csv  (for Project 1 - Sales Analysis)
# ----------------------------------------------------------------------

products = {
    "Electronics": [
        ("Smartphone", 45000), ("Laptop", 120000), ("LED TV 43in", 65000),
        ("Bluetooth Speaker", 6500), ("Power Bank", 3500),
    ],
    "Grocery": [
        ("Basmati Rice 5kg", 2200), ("Cooking Oil 5L", 2800),
        ("Tea 950g", 1350), ("Sugar 10kg", 1400), ("Wheat Flour 20kg", 2600),
    ],
    "Fashion": [
        ("Lawn Suit", 4500), ("Kurta", 2500), ("Jeans", 3200),
        ("Sneakers", 5500),
    ],
    "Home": [
        ("Bed Sheet Set", 3800), ("Electric Kettle", 3200),
        ("Wall Clock", 1500), ("Cookware Set", 8500),
    ],
}

regions = {
    "Punjab": ["Lahore", "Faisalabad", "Rawalpindi", "Multan"],
    "Sindh": ["Karachi", "Hyderabad", "Sukkur"],
    "KPK": ["Peshawar", "Abbottabad", "Mardan"],
    "Islamabad": ["Islamabad"],
    "Balochistan": ["Quetta", "Gwadar"],
}
# Bigger markets get more sales, so weight the regions.
region_names = list(regions.keys())
region_weights = [0.42, 0.30, 0.14, 0.09, 0.05]

n_rows = 2600
rows = []
dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
# Give festival months (Ramadan/Eid around Mar-Apr, and year-end) a boost.
month_boost = {3: 1.6, 4: 1.7, 11: 1.3, 12: 1.4}

for i in range(n_rows):
    category = rng.choice(list(products.keys()))
    product, base_price = products[category][
        rng.integers(len(products[category]))]
    region = rng.choice(region_names, p=region_weights)
    city = rng.choice(regions[region])
    date = dates[rng.integers(len(dates))]

    boost = month_boost.get(date.month, 1.0)
    # Units depend a bit on the festival boost.
    units = max(1, int(rng.poisson(2 * boost)))
    # Price wobbles a little around the base price.
    unit_price = round(base_price * rng.normal(1.0, 0.05), 0)
    amount = units * unit_price

    rows.append({
        "OrderID": 100000 + i,
        "Date": date.strftime("%Y-%m-%d"),
        "Product": product,
        "Category": category,
        "Region": region,
        "City": city,
        "Units": units,
        "UnitPrice": unit_price,
        "Amount": amount,
    })

sales = pd.DataFrame(rows)

# Add a few real-world messes on purpose so the cleaning step matters:
# 1) some missing values in Amount and Region
missing_idx = rng.choice(sales.index, size=40, replace=False)
sales.loc[missing_idx[:20], "Amount"] = np.nan
sales.loc[missing_idx[20:], "Region"] = np.nan
# 2) a handful of exact duplicate rows
dupes = sales.sample(15, random_state=1)
sales = pd.concat([sales, dupes], ignore_index=True)
# 3) shuffle so duplicates/missing aren't all at the bottom
sales = sales.sample(frac=1, random_state=7).reset_index(drop=True)

sales.to_csv(OUT / "sales_data.csv", index=False)
print(f"sales_data.csv: {len(sales)} rows "
      f"({sales['Amount'].isna().sum()} missing amounts, "
      f"{sales.duplicated().sum()} duplicate rows)")


# ----------------------------------------------------------------------
# Dataset 2: customer_transactions.csv  (for Project 2 - Segmentation)
# ----------------------------------------------------------------------
# I plant four hidden customer types so K-Means has real structure to
# find: VIPs, loyal regulars, occasional buyers, and lapsed customers.

profiles = [
    # (label, n_customers, purchases_range, spend_per_txn, recency_range_days)
    ("vip",       80,  (15, 40), (8000, 20000), (1, 25)),
    ("regular",  160,  (6, 15),  (3000, 8000),  (5, 60)),
    ("occasional",240, (2, 6),   (1500, 4000),  (20, 150)),
    ("lapsed",   120,  (1, 4),   (1000, 3000),  (150, 330)),
]

ref_date = pd.Timestamp("2026-01-01")  # "today" for recency
txn_rows = []
customer_id = 5000
transaction_id = 900000

for label, n_cust, purch_range, spend_range, recency_range in profiles:
    for _ in range(n_cust):
        customer_id += 1
        n_purchases = rng.integers(purch_range[0], purch_range[1] + 1)
        last_gap = rng.integers(recency_range[0], recency_range[1] + 1)
        last_date = ref_date - pd.Timedelta(days=int(last_gap))
        # Spread the purchases over the year before the last one.
        for _ in range(n_purchases):
            transaction_id += 1
            back = rng.integers(0, 330)
            date = last_date - pd.Timedelta(days=int(back))
            amount = round(rng.uniform(*spend_range), 0)
            txn_rows.append({
                "TransactionID": transaction_id,
                "CustomerID": customer_id,
                "Date": date.strftime("%Y-%m-%d"),
                "Amount": amount,
            })

transactions = pd.DataFrame(txn_rows).sample(
    frac=1, random_state=3).reset_index(drop=True)
transactions.to_csv(OUT / "customer_transactions.csv", index=False)
print(f"customer_transactions.csv: {len(transactions)} transactions, "
      f"{transactions['CustomerID'].nunique()} customers")
