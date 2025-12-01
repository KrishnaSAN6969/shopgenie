import streamlit as st
import json
import pandas as pd
import os
from backend import app 

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="ShopGenie-E", page_icon="🧞", layout="wide")

# --- CUSTOM THEME (Genie Purple & Gold) ---
st.markdown("""
    <style>
    /* 1. BUTTONS: Deep Purple Gradient */
    .stButton>button {
        background: linear-gradient(90deg, #2b1055 0%, #7597de 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        color: #ffd700; /* Gold text on hover */
    }

    /* 2. CARDS: Clean look with a Gold Accent Border */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        border: 1px solid #e0e0e0 !important;
        border-top: 4px solid #ffd700 !important; /* Gold Top */
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        background-color: #ffffff;
    }

    /* 3. METRICS */
    div[data-testid="stMetricValue"] {
        color: #2b1055;
    }

    /* 4. LANDING TEXT */
    .landing-sub {
        color: #7597de;
        font-size: 1.2rem;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# 2. SESSION STATE
if "messages" not in st.session_state: st.session_state.messages = []
if "selected_product" not in st.session_state: st.session_state.selected_product = None
if "last_json_response" not in st.session_state: st.session_state.last_json_response = None

# 3. SIDEBAR (With PNG Logo)
with st.sidebar:
    # UPDATED: Checks for .png
    if os.path.exists("shopgenie_logo.png"):
        st.image("shopgenie_logo.png", use_container_width=True)
    else:
        st.header("🧞 ShopGenie-E") 
    
    st.divider()
    st.subheader("System Monitor")
    
    col_a, col_b = st.columns(2)
    with col_a: st.metric("Agents", "4", delta="Active")
    with col_b: st.metric("Latency", "24ms", delta="-12ms")
    
    st.divider()
    st.caption("Agent Status")
    status_box = st.empty()
    status_box.info("💤 System Standby")
    
    st.divider()
    if st.button("🔴 Reset System"):
        st.session_state.messages = []
        st.session_state.selected_product = None
        st.session_state.last_json_response = None
        st.rerun()

# 4. LANDING PAGE (Logo Showcase)
if not st.session_state.messages:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # UPDATED: Checks for .png
        if os.path.exists("shopgenie_logo.png"):
            st.image("shopgenie_logo.png", use_container_width=True)
        else:
            st.title("🧞 ShopGenie-E")
            st.warning("⚠️ Save your logo as 'shopgenie_logo.png'!")

        st.markdown("""
            <div style='text-align: center; padding-top: 10px;'>
                <p class='landing-sub'>
                    <b>The Intelligent Multi-Agent Shopping Assistant</b><br>
                    Real-Time Prices • Spec Translation • Bias-Free Ranking
                </p>
            </div>
        """, unsafe_allow_html=True)

# 5. CHAT HISTORY VIEW
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"): st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            if "{" in msg["content"] and "options" in msg["content"]:
                st.markdown("✅ *Market Research Complete (See Dashboard below)*")
            else:
                st.markdown(msg["content"])

# 6. INPUT HANDLING
user_input = st.chat_input("Ex: Best gaming laptop under $1200...")

if user_input:
    st.session_state.selected_product = None 
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    
    # Context
    recent_history = st.session_state.messages[-4:] 
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent_history])
    
    inputs = {"query": user_input, "chat_history": history_text}
    raw_response = ""
    
    with st.spinner("🧞 Genie Agents are working..."):
        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "intent_agent":
                    intent = value.get("intent", "")
                    if intent == "casual_chat": status_box.info("💬 Intent: Casual Chat")
                    elif intent == "ask_budget": status_box.warning("💰 Intent: Asking Budget")
                    else: status_box.info(f"🧠 Context: {value['refined_query']}")
                elif key == "retrieval_agent": status_box.warning(f"🌐 Found {len(value['search_results'])} items")
                elif key == "reasoner_agent": status_box.success("🤔 Analyzing & Scoring...")
                elif key == "image_fetcher_agent": status_box.info("🖼️ Fetching Photos...")
                if "final_recommendation" in value: raw_response = value["final_recommendation"]

    if "{" in raw_response and "options" in raw_response:
        st.session_state.last_json_response = raw_response
    else:
        st.session_state.last_json_response = None

    st.session_state.messages.append({"role": "assistant", "content": raw_response})
    st.rerun()

# 7. DASHBOARD VIEW
if st.session_state.last_json_response:
    try:
        clean_json = st.session_state.last_json_response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        options = data.get("options", [])

        if options:
            st.divider()
            st.markdown("### 🎯 Top 3 Recommendations")
            
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            for i, option in enumerate(options):
                if i < 3:
                    with cols[i]:
                        with st.container(border=True):
                            if "Powerhouse" in option['category']: st.markdown(":rocket: **Powerhouse**")
                            elif "Balanced" in option['category']: st.markdown(":balance_scale: **Balanced**")
                            else: st.markdown(":moneybag: **Budget**")
                            
                            images = option.get('images', [])
                            if images: st.image(images[0], use_container_width=True)
                            st.markdown(f"#### {option['name']}")
                            st.caption(option['price'])
                            
                            insights = option.get('ai_insights', {})
                            score = insights.get('score', 0)
                            st.progress(score / 10, text=f"Genie Score: {score}/10")
                            
                            st.info(f"**Best For:** {insights.get('best_for', 'General Use')}")
                            with st.expander("⚠️ Dealbreaker"): st.warning(insights.get('dealbreaker', 'None found'))
                            
                            if st.button("Select This", key=f"btn_{i}", use_container_width=True):
                                st.session_state.selected_product = option

            st.divider()
            st.markdown("### 📊 Comparison Analysis")
            table_data = []
            for opt in options:
                specs = opt.get('specs', {})
                row = {
                    "Category": opt.get('category', 'Option'),
                    "Product": opt['name'],
                    "Price": opt['price'],
                    "Performance": specs.get('Performance', '-'),
                    "Build Quality": specs.get('Build_Quality', '-'),
                    "Key Feature": specs.get('Key_Feature', '-'), 
                }
                table_data.append(row)
            if table_data:
                df = pd.DataFrame(table_data)
                st.table(df.set_index("Category"))

            if st.session_state.selected_product:
                p = st.session_state.selected_product
                st.divider()
                st.subheader(f"📝 Deep Dive: {p['name']}")
                
                d_col1, d_col2 = st.columns([3, 1])
                with d_col1:
                    st.markdown("#### Why this fits you")
                    if p.get('fit_summary'): st.info(p['fit_summary'])
                    
                    if p.get('tech_specs'):
                        st.markdown("#### 📋 Technical Specifications")
                        for k, v in p['tech_specs'].items(): st.markdown(f"**{k}:** {v}")
                    
                    st.divider()
                    st.markdown("#### Key Benefits (Translated)")
                    details = p.get('full_details', [])
                    if isinstance(details, list): st.markdown("\n".join([f"- {item}" for item in details]))
                    
                    st.markdown("#### Gallery")
                    images = p.get('images', [])
                    if len(images) > 1:
                        g_cols = st.columns(3)
                        for j, img in enumerate(images[:3]):
                            with g_cols[j]: st.image(img, use_container_width=True)

                with d_col2:
                    st.success(f"**Price:** {p['price']}")
                    st.link_button(f"🛒 Buy Now", p['link'], type="primary", use_container_width=True)

    except Exception as e: pass