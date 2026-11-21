import os, sys  # nb:skip
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # nb:skip
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # nb:skip
from agentkit import *  # nb:skip
from core import *  # nb:skip
GROQ_API_KEY = os.environ["GROQ_API_KEY"]  # nb:skip
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # nb:skip

# %% [markdown]
# ## Run it
# First the numbers. Which items will run out before the next delivery? How much would the full
# recommendation cost compared with the budget?

# %%
llm = LLM(GROQ_API_KEY)
data = load_data("data")
queue = ApprovalQueue(log_path="data/audit_log.jsonl")
table = compute_plan_inputs(data)
print(table[["sku", "name", "class", "avg_daily", "on_hand", "days_of_cover",
             "stockout_before_delivery", "recommended_qty", "cost_inr"]].to_string(index=False))
print(f"\nFull recommendation: ₹{table['cost_inr'].sum():,.0f}  ·  Budget: ₹{WEEKLY_BUDGET_INR:,}")

# %% [markdown]
# ### What would a simple rule do?
# This is the baseline the agent has to beat.

# %%
baseline = rule_based_plan(table)
for l in baseline:
    print(l["sku"], l["qty"], f"₹{l['cost_inr']:,.0f}")
print("Total:", f"₹{sum(l['cost_inr'] for l in baseline):,.0f}")

# %% [markdown]
# ### Let the agent plan, then let the rules check it
# Read the trace: did it check its own total with `plan_cost`? If its plan breaks a rule, the code uses the
# rule-based plan instead and tells you why.

# %%
result = plan_week(llm, data, queue)
for step in result["agent"]["trace"]:
    print("STEP", step.get("step"), step.get("action"), str(step.get("input", ""))[:120])
print("\nPlan source:", result["source"])
print("Rule problems with the agent's plan:", result["problems"] or "none")
for l in result["lines"]:
    print(f"  {l['sku']} × {l['qty']}  ₹{l['cost_inr']:,.0f}  · {l['reason']}")

# %% [markdown]
# ### Approvals: some by policy, some by a person

# %%
for item in queue.items.values():
    print(item["status"], "|", item["summary"], "|", item.get("approver", ""))
for item in queue.pending():
    queue.approve(item["id"], approver="Store manager", executor=write_po("data"))
print(open("data/purchase_orders_out.csv").read())

# %% [markdown]
# ### Try this
# 1. Cut the budget to ₹90,000 (`WEEKLY_BUDGET_INR = 90000`) and rerun. What does the agent sacrifice?
# 2. Change the auto-approve limit to ₹50,000. How many POs skip the manager now? Is that acceptable?
# 3. Make the window 7 days instead of 28 in `compute_plan_inputs`. Why do the recommendations jump around?
# 4. **Discussion:** the agent's plan and the rule-based plan differ. How would you test which one is better
#    before trusting either? (Hint: replay past weeks.)
