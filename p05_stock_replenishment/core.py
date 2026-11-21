# %% [markdown]
# ## Project 05 · Stock replenishment agent
# **Business problem:** Annapurna Mart (a fictional grocery store in Pune) orders stock by gut feel every Monday.
# Popular items run out before the next delivery while slow items sit on the shelf, and the weekly
# purchasing budget is never quite enough for everything.
#
# **Division of labour:**
# * **Python** does the maths: average demand, safety stock, reorder points, cost. The numbers are reproducible.
# * The **model** makes the judgement call when the budget is too small for everything: what to cut and why.
# * **Code** checks the model's plan against hard rules (budget, minimum order quantity, shelf life).
# * Small orders are approved automatically **by a written policy**; big ones wait for the store manager.
#   That is the autonomy dial from class, set by value.

# %%
import csv
import json
import math
import os
import re
import sys  # nb:skip
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # nb:skip
from agentkit import Tool, Toolbox, extract_json, run_bounded_agent  # nb:skip

WEEKLY_BUDGET_INR = 150000
REVIEW_DAYS = 7
AUTO_APPROVE_BELOW_INR = 5000
Z = {"A": 1.65, "B": 1.28, "C": 1.04}   # service levels 95% / 90% / 85%


def load_data(data_dir="data"):
    return {"sales": pd.read_csv(os.path.join(data_dir, "sales_history.csv"), parse_dates=["date"]),
            "inventory": pd.read_csv(os.path.join(data_dir, "inventory.csv")),
            "data_dir": data_dir}


# %% [markdown]
# ### Step 1 · The maths (no model involved)
# For each item: average daily sales over the last 28 days, the safety stock needed for the service level,
# and the quantity that brings stock up to "enough until the delivery after next".
# Quantities are rounded up to the supplier's minimum order quantity (MOQ).

# %%
def compute_plan_inputs(data, window_days=28):
    sales = data["sales"]
    recent = sales[sales["date"] > sales["date"].max() - pd.Timedelta(days=window_days)]
    stats = recent.groupby("sku")["units_sold"].agg(["mean", "std"])
    rows = []
    for _, item in data["inventory"].iterrows():
        avg, sd = stats.loc[item.sku, "mean"], stats.loc[item.sku, "std"]
        lt = int(item.lead_time_days)
        safety = Z[item.abc_class] * sd * math.sqrt(lt)
        reorder_point = avg * lt + safety
        target = avg * (lt + REVIEW_DAYS) + safety
        position = int(item.on_hand + item.on_order)
        raw_qty = max(0.0, target - position)
        moq = int(item.min_order_qty)
        qty = int(math.ceil(raw_qty / moq) * moq) if raw_qty > 0 else 0
        days_cover = item.on_hand / avg if avg else float("inf")
        rows.append({
            "sku": item.sku, "name": item["name"], "class": item.abc_class, "supplier": item.supplier,
            "avg_daily": round(avg, 1), "on_hand": int(item.on_hand), "on_order": int(item.on_order),
            "lead_time_days": lt, "days_of_cover": round(days_cover, 1),
            "stockout_before_delivery": bool(item.on_hand + item.on_order < avg * lt),
            "below_reorder_point": bool(position < reorder_point),
            "recommended_qty": qty, "moq": moq, "unit_cost_inr": float(item.unit_cost_inr),
            "cost_inr": round(qty * float(item.unit_cost_inr)),
            "max_qty_shelf_life": int(avg * int(item.shelf_life_days)),
        })
    return pd.DataFrame(rows)


# %% [markdown]
# ### Step 2 · The model prioritises when money is short
# The full recommendation usually costs more than the weekly budget. The agent can look at the numbers with
# read-only tools and must return a plan. It cannot change stock, costs or the budget.

# %%
def make_toolbox(table):
    by_sku = {r["sku"]: r for r in table.to_dict("records")}

    def cost(text):
        total, lines = 0, []
        for sku, qty in re.findall(r"(SKU-\d+)\s*[:x×=]\s*(\d+)", str(text)):
            if sku in by_sku:
                c = int(qty) * by_sku[sku]["unit_cost_inr"]
                total += c
                lines.append(f"{sku} {qty} × ₹{by_sku[sku]['unit_cost_inr']:.0f} = ₹{c:,.0f}")
        return {"lines": lines, "total_inr": total, "budget_inr": WEEKLY_BUDGET_INR,
                "within_budget": total <= WEEKLY_BUDGET_INR}

    return Toolbox([
        Tool("stock_report", lambda _: table[["sku", "name", "class", "days_of_cover", "lead_time_days",
                                              "stockout_before_delivery", "recommended_qty", "moq",
                                              "cost_inr"]].to_dict("records"),
             "Stock position and the maths-based recommendation for every item.", "any text"),
        Tool("item_detail", lambda s: by_sku.get(str(s).strip().upper(), {"error": "Unknown SKU"}),
             "All numbers for one item.", "a SKU such as SKU-104"),
        Tool("plan_cost", cost, "Cost of a draft plan, checked against the budget.",
             "pairs like 'SKU-101: 20, SKU-104: 30'"),
    ])


def agent_plan(llm, table, max_steps=5):
    need = int(table["cost_inr"].sum())
    return run_bounded_agent(
        llm,
        goal=(f"Propose this week's purchase plan. The full recommendation costs ₹{need:,} but the budget is "
              f"₹{WEEKLY_BUDGET_INR:,}. Protect items that will run out before the next delivery and class A "
              "items first. Quantities must be multiples of each item's MOQ. Check your total with plan_cost. "
              'Finish with JSON: {"lines": [{"sku": "...", "qty": number, "reason": "..."}], "summary": "..."}'),
        toolbox=make_toolbox(table), max_steps=max_steps,
        role="You are a careful retail purchasing analyst.")


# %% [markdown]
# ### Step 3 · Hard rules check the model's plan
# If the plan breaks a rule, it is rejected and a simple rule-based plan is used instead. The fallback shows
# what "no AI" would have done, a fair comparison for the business case.

# %%
def validate_plan(plan, table):
    by_sku = {r["sku"]: r for r in table.to_dict("records")}
    problems, lines = [], []
    for line in (plan or {}).get("lines", []):
        sku, qty = str(line.get("sku", "")).upper(), line.get("qty")
        if sku not in by_sku:
            problems.append(f"Unknown SKU {sku}.")
            continue
        try:
            qty = int(qty)
        except (TypeError, ValueError):
            problems.append(f"{sku}: quantity is not a number.")
            continue
        r = by_sku[sku]
        if qty <= 0:
            continue
        if qty % r["moq"]:
            problems.append(f"{sku}: {qty} is not a multiple of the MOQ ({r['moq']}).")
        if qty > r["max_qty_shelf_life"]:
            problems.append(f"{sku}: {qty} cannot be sold within its shelf life (max {r['max_qty_shelf_life']}).")
        if qty > 2 * max(r["recommended_qty"], r["moq"]):
            problems.append(f"{sku}: {qty} is more than twice the recommendation ({r['recommended_qty']}).")
        lines.append({"sku": sku, "qty": qty, "reason": line.get("reason", ""),
                      "cost_inr": qty * r["unit_cost_inr"], "supplier": r["supplier"]})
    total = sum(l["cost_inr"] for l in lines)
    if total > WEEKLY_BUDGET_INR:
        problems.append(f"Plan costs ₹{total:,.0f}, over the ₹{WEEKLY_BUDGET_INR:,} budget.")
    ordered = {l["sku"] for l in lines}
    for r in by_sku.values():
        if r["stockout_before_delivery"] and r["class"] == "A" and r["recommended_qty"] and r["sku"] not in ordered:
            problems.append(f"{r['sku']} ({r['name']}) will run out before delivery but is not ordered.")
    return problems, lines


def rule_based_plan(table):
    """Fallback without AI: first cover expected sales until the delivery after next, in priority order,
    then add safety stock with whatever budget is left."""
    order = table.assign(p=(~table["stockout_before_delivery"]).astype(int) * 10
                         + table["class"].map({"A": 0, "B": 1, "C": 2})).sort_values(["p", "days_of_cover"])
    qty = {r["sku"]: 0 for r in order.to_dict("records")}
    spent = 0.0
    for r in order.to_dict("records"):            # pass 1: no safety stock
        gap = r["avg_daily"] * (r["lead_time_days"] + REVIEW_DAYS) - r["on_hand"] - r["on_order"]
        q = int(math.ceil(max(0, gap) / r["moq"]) * r["moq"])
        while q > 0 and spent + q * r["unit_cost_inr"] > WEEKLY_BUDGET_INR:
            q -= r["moq"]
        qty[r["sku"]], spent = q, spent + q * r["unit_cost_inr"]
    for r in order.to_dict("records"):            # pass 2: top up towards the full recommendation
        while qty[r["sku"]] + r["moq"] <= r["recommended_qty"] and \
                spent + r["moq"] * r["unit_cost_inr"] <= WEEKLY_BUDGET_INR:
            qty[r["sku"]] += r["moq"]
            spent += r["moq"] * r["unit_cost_inr"]
    by_sku = {r["sku"]: r for r in table.to_dict("records")}
    return [{"sku": s, "qty": q, "reason": "rule-based priority", "cost_inr": q * by_sku[s]["unit_cost_inr"],
             "supplier": by_sku[s]["supplier"]} for s, q in qty.items() if q > 0]


# %% [markdown]
# ### Step 4 · One purchase order per supplier, approved by policy or by a person
# POs under ₹5,000 are approved automatically **because a written policy says so**, and the log records that.
# Everything else waits for the store manager.

# %%
def propose_purchase_orders(lines, queue, data_dir="data", plan_source="agent"):
    by_supplier = defaultdict(list)
    for l in lines:
        by_supplier[l["supplier"]].append(l)
    created = []
    for supplier, items in by_supplier.items():
        total = sum(i["cost_inr"] for i in items)
        item = queue.propose("purchase_order", f"PO to {supplier}: ₹{total:,.0f} ({len(items)} items)",
                             {"supplier": supplier, "lines": items, "total_inr": total, "plan_source": plan_source})
        if total < AUTO_APPROVE_BELOW_INR:
            queue.approve(item["id"], approver=f"auto-policy (PO under ₹{AUTO_APPROVE_BELOW_INR:,})",
                          executor=write_po(data_dir))
        created.append(item)
    return created


def write_po(data_dir="data"):
    def run(payload):
        path = os.path.join(data_dir, "purchase_orders_out.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["supplier", "sku", "qty", "cost_inr", "reason"])
            for l in payload["lines"]:
                w.writerow([payload["supplier"], l["sku"], l["qty"], l["cost_inr"], l["reason"]])
        return "written to purchase_orders_out.csv"
    return run


def plan_week(llm, data, queue):
    table = compute_plan_inputs(data)
    agent = agent_plan(llm, table)
    try:
        plan = extract_json(agent["answer"]) if isinstance(agent["answer"], str) else agent["answer"]
    except ValueError:
        plan = None
    problems, lines = validate_plan(plan, table)
    source = "agent"
    if problems or not lines:
        lines, source = rule_based_plan(table), "rule-based fallback"
    pos = propose_purchase_orders(lines, queue, data["data_dir"], source)
    return {"table": table, "agent": agent, "plan": plan, "problems": problems,
            "lines": lines, "source": source, "purchase_orders": pos}
