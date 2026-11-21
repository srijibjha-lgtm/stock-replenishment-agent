"""Small Streamlit helpers shared by the six project apps."""
import json

import streamlit as st

from agentkit import DEFAULT_MODEL, LLM, ApprovalQueue


def _secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:  # no secrets file
        return ""


def sidebar_llm(default_model=DEFAULT_MODEL, ask_approver=True):
    st.sidebar.header("Model · free Groq tier")
    key = _secret("GROQ_API_KEY") or st.sidebar.text_input(
        "Groq API key", type="password", help="Free, no credit card: console.groq.com → API Keys")
    model = st.sidebar.text_input("Model", default_model)
    approver = st.sidebar.text_input("Your name (needed to approve)", "") if ask_approver else ""
    if not key:
        st.sidebar.info("Add a Groq API key to run the agent. You can still browse the data.")
        return None, approver
    cache_key = (key, model)
    if st.session_state.get("_llm_key") != cache_key:
        st.session_state["_llm"] = LLM(key, model=model)
        st.session_state["_llm_key"] = cache_key
    return st.session_state["_llm"], approver


def get_queue(log_path=None):
    if "queue" not in st.session_state:
        st.session_state["queue"] = ApprovalQueue(log_path=log_path)
    return st.session_state["queue"]


def show_trace(trace, title="Agent steps"):
    with st.expander(f"{title} ({len(trace)})"):
        for step in trace:
            st.code(json.dumps(step, indent=2, default=str)[:3000], language="json")


def show_flags(flags):
    if not flags:
        st.success("Rule checks: no flags.")
    for f in flags:
        st.warning(f)


def render_queue(queue, approver, executors=None, editable=None):
    """executors: kind -> function(payload); editable: kind -> payload field a person may edit."""
    executors, editable = executors or {}, editable or {}
    pending = queue.pending()
    if not pending:
        st.info("Nothing is waiting for approval.")
    for item in pending:
        with st.container(border=True):
            st.markdown(f"**{item['summary']}**  \n`{item['id']}` · {item['kind']}")
            show_flags(item["flags"])
            payload = dict(item["payload"])
            field = editable.get(item["kind"])
            if field:
                payload[field] = st.text_area("Edit before approving", payload[field], height=220,
                                              key=f"edit_{item['id']}")
            other = {k: v for k, v in payload.items() if k != field}
            st.json(other, expanded=False)
            if item["evidence"]:
                with st.expander("Evidence the agent used"):
                    st.write(item["evidence"])
            c1, c2, c3 = st.columns([1, 1, 2])
            reason = c3.text_input("Reason (for rejection)", key=f"reason_{item['id']}")
            if c1.button("Approve", key=f"ok_{item['id']}", type="primary", disabled=not approver):
                changed = payload if payload != item["payload"] else None
                queue.approve(item["id"], approver, executor=executors.get(item["kind"]), edited_payload=changed)
                st.rerun()
            if c2.button("Reject", key=f"no_{item['id']}", disabled=not approver):
                queue.reject(item["id"], approver, reason)
                st.rerun()
    if not approver and pending:
        st.caption("Enter your name in the sidebar to approve or reject.")
    done = [i for i in queue.items.values() if i["status"] != "AWAITING_HUMAN_APPROVAL"]
    if done:
        with st.expander(f"Decided ({len(done)})"):
            for i in done:
                st.write(f"`{i['id']}` **{i['status']}** by {i.get('approver')} · {i['summary']}"
                         + (f" · result: {i.get('result')}" if i.get("result") else ""))


def render_audit(queue):
    if queue.audit:
        st.dataframe(queue.audit, width="stretch")
    else:
        st.info("The audit log is empty.")
