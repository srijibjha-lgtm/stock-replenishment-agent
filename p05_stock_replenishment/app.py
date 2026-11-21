"""Stock Replenishment Agent — run with:  streamlit run p05_stock_replenishment/app.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "common")]

import pandas as pd
import streamlit as st

import core
from ui import get_queue, render_audit, render_queue, show_trace, sidebar_llm

st.set_page_config(page_title="Stock Replenishment Agent", layout="wide")
st.title("📦 Stock Replenishment Agent · Annapurna Mart")
st.caption("Python does the maths, the model prioritises within the budget, rules check the plan, "
           "and large purchase orders wait for the store manager.")

llm, approver = sidebar_llm()
core.WEEKLY_BUDGET_INR = st.sidebar.number_input("Weekly budget (₹)", 20000, 400000, core.WEEKLY_BUDGET_INR, 10000)
core.AUTO_APPROVE_BELOW_INR = st.sidebar.number_input("Auto-approve POs below (₹)", 0, 100000,
                                                      core.AUTO_APPROVE_BELOW_INR, 1000)

DATA_DIR = os.path.join(HERE, "data")
data = core.load_data(DATA_DIR)
queue = get_queue(os.path.join(DATA_DIR, "audit_log.jsonl"))
table = core.compute_plan_inputs(data)

tab_stock, tab_plan, tab_queue, tab_audit = st.tabs(["Stock position", "This week's plan", "Approval queue", "Audit log"])

with tab_stock:
    need = table["cost_inr"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Full recommendation", f"₹{need:,.0f}")
    c2.metric("Budget", f"₹{core.WEEKLY_BUDGET_INR:,}")
    c3.metric("Items at risk before delivery", int(table["stockout_before_delivery"].sum()))
    st.dataframe(table, width="stretch", hide_index=True)
    sku = st.selectbox("Sales history for", table["sku"])
    hist = data["sales"][data["sales"]["sku"] == sku].set_index("date")["units_sold"]
    st.line_chart(hist)

with tab_plan:
    baseline = core.rule_based_plan(table)
    st.markdown("**Rule-based baseline (no AI)**")
    st.dataframe(pd.DataFrame(baseline), width="stretch", hide_index=True)
    if st.button("Ask the agent for this week's plan", type="primary", disabled=llm is None):
        with st.spinner("Agent is prioritising…"):
            st.session_state["plan"] = core.plan_week(llm, data, queue)
    res = st.session_state.get("plan")
    if res:
        st.markdown(f"**Plan used: {res['source']}**")
        for p in res["problems"]:
            st.warning("Agent plan rejected: " + p)
        show_trace(res["agent"]["trace"])
        st.dataframe(pd.DataFrame(res["lines"]), width="stretch", hide_index=True)
        if isinstance(res["plan"], dict) and res["plan"].get("summary"):
            st.info(res["plan"]["summary"])

with tab_queue:
    render_queue(queue, approver, executors={"purchase_order": core.write_po(DATA_DIR)})

with tab_audit:
    render_audit(queue)
