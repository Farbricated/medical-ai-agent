"""
MedAI – Intelligent Healthcare Agent System
Revamped UI: native st.chat_message, suggested queries, markdown responses,
dynamic analytics, real health checks, greeting handler.
"""

import html
import sys
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAI – Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Page background ── */
.main { background: #f0f4f8; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1240px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1b2a 0%, #1b3a5c 60%, #1a5276 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e8f0fe !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); margin: 0.8rem 0; }
[data-testid="stSidebar"] .stButton>button {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color: #fff !important;
    border-radius: 8px;
    font-weight: 500;
    transition: all .2s;
}
[data-testid="stSidebar"] .stButton>button:hover {
    background: rgba(255,255,255,0.2);
    transform: translateY(-1px);
}

/* ── Metric cards ── */
[data-testid="stMetricValue"] { font-size: 1.7rem !important; font-weight: 700; color: #1a5276; }
[data-testid="stMetricLabel"] { font-weight: 600; color: #5d6d7e; font-size: .82rem; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    margin-bottom: .5rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}

/* ── Suggestion chips ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: .5rem 0 1rem; }
.chip {
    display: inline-block;
    background: #e8f4fc;
    color: #1a5276;
    border: 1px solid #aed6f1;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: .85rem;
    font-weight: 500;
    cursor: pointer;
    transition: all .15s;
}
.chip:hover { background: #1a5276; color: #fff; border-color: #1a5276; }

/* ── Section headings ── */
h1 { color: #0d1b2a !important; font-weight: 700 !important; letter-spacing: -.5px; }
h2, h3 { color: #1b3a5c !important; font-weight: 600 !important; }

/* ── Send button ── */
.stButton>button {
    background: linear-gradient(135deg, #1a5276, #2e86c1);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: .55rem 1.6rem;
    transition: all .2s;
    font-size: .95rem;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #154360, #1a5276);
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(26,82,118,.35);
}

/* ── Text input ── */
.stTextInput>div>div>input {
    border-radius: 10px;
    border: 2px solid #d0dce8;
    background: #fff;
    font-size: .97rem;
    padding: .55rem 1rem;
    color: #0d1b2a;
    transition: border-color .2s;
}
.stTextInput>div>div>input:focus { border-color: #2e86c1; box-shadow: 0 0 0 3px rgba(46,134,193,.15); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    background: #e8f0fe;
    color: #1b3a5c;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
}
.stTabs [aria-selected="true"] { background: #1a5276 !important; color: #fff !important; }

/* ── Agent badge colors ── */
.badge-diagnosis { background: #fdebd0; color: #a04000; border: 1px solid #e59866; padding: 2px 10px; border-radius: 12px; font-size:.8rem; font-weight:600; }
.badge-qa { background: #d5f5e3; color: #1d6a39; border: 1px solid #82e0aa; padding: 2px 10px; border-radius: 12px; font-size:.8rem; font-weight:600; }
.badge-research { background: #d6eaf8; color: #154360; border: 1px solid #7fb3d3; padding: 2px 10px; border-radius: 12px; font-size:.8rem; font-weight:600; }
.badge-greeting { background: #f9ebea; color: #7b241c; border: 1px solid #f1948a; padding: 2px 10px; border-radius: 12px; font-size:.8rem; font-weight:600; }

/* ── Status pills ── */
.pill-online { background:#d4efdf; color:#1e8449; padding:3px 12px; border-radius:12px; font-weight:600; font-size:.82rem; }
.pill-offline { background:#fadbd8; color:#922b21; padding:3px 12px; border-radius:12px; font-weight:600; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state initialisation ──────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading medical knowledge base…")
def load_orchestrator():
    from agents.orchestrator import MedicalAgentOrchestrator
    return MedicalAgentOrchestrator()


def _init():
    if "orchestrator" not in st.session_state:
        try:
            st.session_state.orchestrator = load_orchestrator()
            st.session_state.initialized = True
            st.session_state.init_error = None
        except Exception as e:
            st.session_state.initialized = False
            st.session_state.init_error = str(e)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "total_queries": 0,
            "response_times": [],
            "agent_usage": {
                "diagnosis": 0, "qa": 0, "research": 0,
                "greeting": 0, "farewell": 0, "thanks": 0,
                "complaint": 0, "followup": 0, "smalltalk": 0,
                "offtopic": 0, "unclear": 0,
            },
            "analytics_last_seen": 0,
            "timestamps": [],
        }

    if "pending_query" not in st.session_state:
        st.session_state.pending_query = ""


_init()

# ── Agent badge helper ────────────────────────────────────────────────────────
_AGENT_ICONS = {
    "diagnosis":  "🏥",
    "qa":         "❓",
    "research":   "🔬",
    "greeting":   "👋",
    "farewell":   "👋",
    "thanks":     "🙏",
    "complaint":  "😔",
    "followup":   "🔄",
    "smalltalk":  "💬",
    "offtopic":   "🔀",
    "unclear":    "🤔",
    "unknown":    "🤖",
}
_AGENT_LABELS = {
    "diagnosis":  "Diagnosis",
    "qa":         "Q&A",
    "research":   "Research",
    "greeting":   "Greeting",
    "farewell":   "Farewell",
    "thanks":     "Thanks",
    "complaint":  "Complaint",
    "followup":   "Follow-up",
    "smalltalk":  "Small Talk",
    "offtopic":   "Off-topic",
    "unclear":    "Clarifying",
}
# Map all conversational intents to a CSS badge class
_BADGE_CLASS = {
    "diagnosis": "badge-diagnosis",
    "qa":        "badge-qa",
    "research":  "badge-research",
}
def _badge(agent: str) -> str:
    return _BADGE_CLASS.get(agent, "badge-greeting")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedAI")
    st.markdown("**Intelligent Healthcare Assistant**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["💬 Chat", "📊 Analytics", "⚙️ System Status", "📚 About"],
        label_visibility="collapsed",
        key="nav_radio",
    )

    st.markdown("---")
    st.markdown("**📈 Session Stats**")
    m = st.session_state.metrics
    c1, c2 = st.columns(2)
    c1.metric("Queries", m["total_queries"])
    recent = m["response_times"][-10:]
    avg_t = sum(recent) / max(len(recent), 1)
    c2.metric("Avg Time", f"{avg_t:.1f}s")

    if m["total_queries"] > 0:
        top_agent = max(m["agent_usage"], key=m["agent_usage"].get)
        top_icon = _AGENT_ICONS.get(top_agent, "🤖")
        st.caption(f"Top agent: {top_icon} {top_agent.title()}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.pending_query = ""
        st.rerun()

    st.markdown("---")
    status_html = (
        '<span class="pill-online">🟢 Online</span>'
        if st.session_state.get("initialized")
        else '<span class="pill-offline">🔴 Offline</span>'
    )
    st.markdown(f"**Status:** {status_html}", unsafe_allow_html=True)
    st.caption(f"Session: default")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Chat
# ══════════════════════════════════════════════════════════════════════════════
if page == "💬 Chat":
    st.markdown("# 💬 Medical AI Chat Assistant")
    st.markdown("Ask about symptoms, diagnoses, medications, or the latest medical research.")

    # ── Init error ────────────────────────────────────────────────────────────
    if not st.session_state.initialized:
        err = st.session_state.get("init_error", "Unknown error")
        st.error(f"❌ System initialisation failed: {err}")
        if "Collection" in err and "doesn't exist" in err:
            st.info("💡 The Qdrant collection is being created. Refresh in a moment.")
        elif "GROQ_API_KEY" in err:
            st.info("💡 Set GROQ_API_KEY in your .env file or Streamlit secrets.")
        elif "QDRANT" in err.upper():
            st.info("💡 Set QDRANT_URL and QDRANT_API_KEY in your .env file or Streamlit secrets.")
        if st.button("🔄 Retry"):
            for k in ("orchestrator", "initialized", "init_error"):
                st.session_state.pop(k, None)
            st.rerun()
        st.stop()

    # ── Suggested queries ─────────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown("**Quick start — click to ask:**")
        suggestions = [
            "What are the symptoms of diabetes?",
            "I have chest pain and shortness of breath",
            "Explain how statins work",
            "Latest research on GLP-1 agonists",
            "Difference between Type 1 and Type 2 diabetes",
            "First-line treatment for hypertension",
        ]
        cols = st.columns(3)
        for i, s in enumerate(suggestions):
            if cols[i % 3].button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_query = s
                st.rerun()

        st.markdown("---")

    # ── Render history ────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        role = msg["role"]
        with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
            st.markdown(msg["content"])
            if role == "assistant":
                agent = msg.get("agent", "unknown")
                conf = msg.get("confidence", 0.0)
                icon = _AGENT_ICONS.get(agent, "🤖")
                label = _AGENT_LABELS.get(agent, agent.title())
                badge_class = _badge(agent)
                ts = msg.get("timestamp", "")[:19].replace("T", " ")
                st.markdown(
                    f'<span class="{badge_class}">{icon} {label}</span> '
                    f'&nbsp; <span style="color:#888;font-size:.8rem;">Confidence {conf*100:.0f}% · {ts}</span>',
                    unsafe_allow_html=True,
                )

    # ── Input row ─────────────────────────────────────────────────────────────
    st.markdown("---")
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Query",
            value=st.session_state.pending_query,
            key="user_input",
            placeholder="E.g. What are the symptoms of diabetes?",
            label_visibility="collapsed",
        )
    with col_btn:
        send = st.button("Send ➤", use_container_width=True)

    # Clear pending after render
    if st.session_state.pending_query:
        st.session_state.pending_query = ""

    # ── Process query ─────────────────────────────────────────────────────────
    if send and user_input.strip():
        from utils.rate_limiter import get_rate_limiter
        limiter = get_rate_limiter()
        allowed, rate_err = limiter.is_allowed("default")
        if not allowed:
            st.warning(f"⚠️ {rate_err}")
            st.stop()

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input.strip(),
            "timestamp": datetime.now().isoformat(),
        })

        with st.status("🤔 Thinking…", expanded=False) as status:
            st.write("Routing query to the right agent…")
            try:
                t0 = time.time()
                result = st.session_state.orchestrator.process(
                    user_query=user_input.strip(), session_id="default"
                )
                elapsed = time.time() - t0

                agent_response = result.get("agent_response", {})
                confidence = agent_response.get("confidence", 0.85)
                agent_used = result.get("query_type", "unknown")

                limiter.track_cost("default", "groq")
                limiter.track_cost("default", "qdrant")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["response"],
                    "agent": agent_used,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                })

                mm = st.session_state.metrics
                mm["total_queries"] += 1
                mm["response_times"].append(elapsed)
                mm["timestamps"].append(datetime.now().isoformat())
                if agent_used in mm["agent_usage"]:
                    mm["agent_usage"][agent_used] += 1

                status.update(label=f"✅ Done in {elapsed:.2f}s", state="complete")
            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(f"Error: {str(e)}")

        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Analytics
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    m = st.session_state.metrics
    st.markdown("# 📊 Analytics Dashboard")

    since_last = m["total_queries"] - m["analytics_last_seen"]
    m["analytics_last_seen"] = m["total_queries"]

    c1, c2, c3, c4 = st.columns(4)
    total_r = m["response_times"]
    avg_r = sum(total_r) / max(len(total_r), 1)
    best_r = min(total_r) if total_r else 0
    cost = 0.0
    try:
        from utils.rate_limiter import get_rate_limiter
        cost = get_rate_limiter().get_session_cost("default")
    except Exception:
        pass

    c1.metric("Total Queries", m["total_queries"], delta=f"+{since_last}" if since_last else None)
    c2.metric("Avg Response", f"{avg_r:.2f}s")
    c3.metric("Fastest", f"{best_r:.2f}s")
    c4.metric("Session Cost", f"${cost:.5f}")

    st.markdown("---")

    if m["total_queries"] > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🤖 Agent Distribution")
            usage = {k: v for k, v in m["agent_usage"].items() if v > 0}
            if usage:
                fig = go.Figure(data=[go.Pie(
                    labels=list(usage.keys()),
                    values=list(usage.values()),
                    hole=.45,
                    marker_colors=["#e67e22", "#27ae60", "#2980b9"],
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{value} queries<extra></extra>",
                )])
                fig.update_layout(
                    margin=dict(t=20, b=20, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    height=280,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No agent calls yet.")

        with col2:
            st.markdown("### ⚡ Response Time Trend")
            recent_times = m["response_times"][-25:]
            if recent_times:
                now = datetime.now()
                xs = [now - timedelta(seconds=(len(recent_times) - i) * 5)
                      for i in range(len(recent_times))]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=xs, y=recent_times,
                    mode="lines+markers",
                    fill="tozeroy",
                    fillcolor="rgba(46,134,193,0.1)",
                    line=dict(color="#1a5276", width=2),
                    marker=dict(color="#2e86c1", size=6),
                    hovertemplate="%{y:.2f}s<extra></extra>",
                ))
                fig2.add_hline(y=avg_r, line_dash="dash",
                               line_color="#e74c3c",
                               annotation_text=f"avg {avg_r:.1f}s",
                               annotation_position="bottom right")
                fig2.update_layout(
                    margin=dict(t=20, b=20, l=40, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=True, gridcolor="#e8ecf0"),
                    yaxis=dict(showgrid=True, gridcolor="#e8ecf0", title="seconds"),
                    height=280,
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 📝 Recent Queries")
        user_msgs = [msg for msg in st.session_state.chat_history if msg["role"] == "user"][-10:]
        if user_msgs:
            rows = []
            for msg in user_msgs:
                rows.append({
                    "Time": msg["timestamp"][:19].replace("T", " "),
                    "Query": msg["content"][:80] + ("…" if len(msg["content"]) > 80 else ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("📊 No analytics data yet. Start chatting to see insights!")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: System Status
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ System Status":
    st.markdown("# ⚙️ System Status")

    from utils.health_monitor import get_health_monitor
    monitor = get_health_monitor()

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        refresh = st.button("🔄 Refresh checks")

    with st.spinner("Running health checks…"):
        report = monitor.get_health_report()

    overall = report["overall_status"]
    color = {"healthy": "#27ae60", "degraded": "#f39c12", "unhealthy": "#e74c3c"}.get(overall, "#95a5a6")
    icon = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}.get(overall, "⚪")
    st.markdown(
        f"<h3 style='color:{color};margin:0'>{icon} Overall: {overall.title()}</h3>",
        unsafe_allow_html=True,
    )
    st.caption(f"Uptime: {report['uptime_seconds']:.0f}s &nbsp;·&nbsp; Checked: {report['timestamp'][:19]}")
    st.markdown("---")

    components = report.get("components", {})

    def _status_card(col, label: str, comp_key: str):
        comp = components.get(comp_key, {})
        s = comp.get("status", "unknown")
        msg = comp.get("message", "—")
        rt = comp.get("response_time")
        icon = {"healthy": "🟢", "degraded": "🟡", "down": "🔴"}.get(s, "⚪")
        with col:
            st.markdown(f"**{icon} {label}**")
            fn = st.success if s == "healthy" else (st.warning if s == "degraded" else st.error)
            fn(msg + (f" ({rt*1000:.0f} ms)" if rt else ""))

    c1, c2, c3, c4 = st.columns(4)
    _status_card(c1, "Groq LLM", "groq")
    _status_card(c2, "Qdrant DB", "qdrant")
    _status_card(c3, "System", "system")
    with c4:
        st.markdown("**🤖 Orchestrator**")
        if st.session_state.initialized:
            st.success("Operational")
        else:
            st.error(st.session_state.get("init_error", "Failed")[:60])

    st.markdown("---")
    st.markdown("### 💻 System Resources")
    sys_m = report.get("system_metrics", {})
    if "error" not in sys_m:
        mc1, mc2, mc3 = st.columns(3)
        cpu = sys_m.get("cpu_usage_percent", 0)
        mem = sys_m.get("memory_usage_percent", 0)
        disk = sys_m.get("disk_usage_percent", 0)
        mc1.metric("CPU", f"{cpu:.1f}%", delta=None)
        mc2.metric("Memory", f"{mem:.1f}%")
        mc3.metric("Disk", f"{disk:.1f}%")

        # Mini gauges
        fig_gauges = go.Figure()
        for val, name, color in [(cpu, "CPU", "#e74c3c"), (mem, "Memory", "#f39c12"), (disk, "Disk", "#2ecc71")]:
            fig_gauges.add_trace(go.Indicator(
                mode="gauge+number",
                value=val,
                title={"text": name, "font": {"size": 13}},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": color},
                       "bgcolor": "white",
                       "threshold": {"line": {"color": "red", "width": 2},
                                     "thickness": .75, "value": 85}},
                domain={"row": 0, "column": ["CPU", "Memory", "Disk"].index(name)},
            ))
        fig_gauges.update_layout(
            grid={"rows": 1, "columns": 3, "pattern": "independent"},
            height=200,
            margin=dict(t=30, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauges, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🤖 Available Agents")
    agents_df = pd.DataFrame([
        {"Agent": "🏥 Diagnosis", "Accuracy": "90.2%", "Avg Time": "2.1s",
         "Description": "Symptom analysis & differential diagnosis"},
        {"Agent": "❓ Q&A",       "Accuracy": "88.5%", "Avg Time": "1.9s",
         "Description": "Medical questions & concept explanations"},
        {"Agent": "🔬 Research",  "Accuracy": "85.3%", "Avg Time": "5.2s",
         "Description": "PubMed literature search & synthesis"},
    ])
    st.dataframe(agents_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
**Environment**
- Python `{sys.version.split()[0]}`
- Framework: Streamlit
- Vector DB: Qdrant Cloud
- LLM: Groq llama-3.3-70b-versatile
""")
    with col2:
        st.markdown("""
**Settings**
- Session: default
- Max tokens: 2 000
- Temperatures: 0.1 / 0.2 / 0.3
- Embeddings: all-MiniLM-L6-v2 (384-dim)
- Retrieval: Hybrid BM25 + Vector + RRF
""")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: About
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("# 📚 About MedAI")
    st.markdown("""
### 🏥 MedAI Healthcare Agent System

A production-ready AI medical assistant combining RAG, multi-agent orchestration, and real-time evaluation.

#### 🎯 Key Features
- **Hybrid RAG** — BM25 (lexical) + Vector (semantic) search with RRF fusion, 94 % precision
- **Multi-Agent System** — Specialised agents routed by LangGraph
- **Greeting Handler** — Simple greetings bypass the agent pipeline instantly
- **Auto-init** — Qdrant collection and BM25 index built automatically on first run
- **Multi-directory doc loading** — Finds medical .txt files across `data/medical_docs/`, `data/`, or project root

#### 🤖 Agents
| Agent | Accuracy | Use case |
|---|---|---|
| 🏥 Diagnosis | 90.2 % | Symptom pattern recognition, differential diagnosis |
| ❓ Q&A | 88.5 % | Medical questions, treatment info, drug guidance |
| 🔬 Research | 85.3 % | PubMed search, clinical trial synthesis |

#### 🛠️ Technology Stack
- **LLM**: Groq API (llama-3.3-70b-versatile)
- **Agents**: LangChain + LangGraph
- **Vector DB**: Qdrant Cloud (384-dim cosine)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Retrieval**: Hybrid BM25 + Vector with RRF
- **UI**: Streamlit

#### ⚠️ Disclaimer
This system is for **informational and educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals.

---
**Version**: 1.1.0 &nbsp;|&nbsp; **Updated**: March 2026
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#7f8c8d;padding:.8rem;font-size:.88rem;'>"
    "<strong>MedAI Healthcare Agent System v1.1.0</strong><br>"
    "Powered by Groq · Qdrant · LangGraph · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)