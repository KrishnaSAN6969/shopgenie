import streamlit as st
import json
import pandas as pd
from backend import app 

# 1. PAGE CONFIGURATION & STYLING
st.set_page_config(page_title="ShopGenie-E", page_icon="🛍️", layout="wide")

# Custom CSS for modern UI and centering
st.markdown("""
    <style>
    .stButton>button {
        border-radius: 20px;
        font-weight: 600;
        border: 1px solid #ddd;
    }
    .agent-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    /* Centering the landing page text */
    .landing-text {
        text-align: center;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. SESSION STATE MANAGEMENT
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_product" not in st.session_state:
    st.session_state.selected_product = None
if "last_json_response" not in st.session_state:
    st.session_state.last_json_response = None

# 3. SIDEBAR (SYSTEM MONITOR)
with st.sidebar:
    st.header("System Monitor")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Agents", "4", delta="Active")
    with col_b:
        st.metric("Latency", "24ms", delta="-12ms")
    
    st.divider()
    st.subheader("Agent Activity Log")
    status_box = st.empty()
    status_box.info("💤 System Standby")
    
    st.divider()
    if st.button("🔴 Reset System"):
        st.session_state.messages = []
        st.session_state.selected_product = None
        st.session_state.last_json_response = None
        st.rerun()

# 4. LANDING PAGE (Logo & Brief)
# Only shows when there are no messages
if not st.session_state.messages:
    # Use columns to center the image
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        # A nice shopping bag / AI icon
        st.image("https://img.icons8.com/clouds/500/000000/online-shop-2.png", use_container_width=True)
    
    st.markdown("""
        <div class="landing-text">
            <h1>ShopGenie-E</h1>
            <h3 style='color: #555;'>The Intelligent Shopping Assistant</h3>
            <p style='font-size: 18px; color: #666;'>
                I research real-time prices, analyze technical specs, and provide honest, 
                bias-free recommendations for electronics. <br>
                Just tell me what you need, and I'll handle the rest.
            </p>
        </div>
    """, unsafe_allow_html=True)

# 5. CHAT HISTORY VIEW
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
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
    
    # Hide the landing page title by rerunning loop (optional, but cleaner)
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Context
    recent_history = st.session_state.messages[-4:] 
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent_history])
    
    inputs = {
        "query": user_input, 
        "chat_history": history_text,
        "refined_query": "",
        "intent": "", 
        "search_results": [], 
        "final_recommendation": "", 
        "critique": "", 
        "revision_needed": False
    }
    
    raw_response = ""
    
    with st.spinner("🔄 Activating Multi-Agent Pipeline..."):
        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "intent_agent":
                    intent = value.get("intent", "")
                    if intent == "casual_chat":
                         status_box.info("💬 Intent: Casual Chat")
                    elif intent == "ask_budget":
                         status_box.warning("💰 Intent: Asking for Budget")
                    else:
                         status_box.info(f"🧠 Context: {value['refined_query']}")
                elif key == "retrieval_agent":
                    status_box.warning(f"🌐 Retrieval: Found {len(value['search_results'])} products")
                elif key == "reasoner_agent":
                    status_box.success("🤔 Reasoner: Analyzing trade-offs...")
                elif key == "image_fetcher_agent":
                    status_box.info("🖼️ Visuals: Fetching images...")
                
                if "final_recommendation" in value:
                    raw_response = value["final_recommendation"]

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
            st.subheader("🎯 Top 3 Recommendations")
            
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            
            for i, option in enumerate(options):
                if i < 3:
                    with cols[i]:
                        with st.container(border=True):
                            if "Powerhouse" in option['category']:
                                st.markdown(":rocket: **Powerhouse**")
                            elif "Balanced" in option['category']:
                                st.markdown(":balance_scale: **Balanced**")
                            else:
                                st.markdown(":moneybag: **Budget**")
                            
                            images = option.get('images', [])
                            if images:
                                st.image(images[0], use_container_width=True)
                                
                            st.markdown(f"#### {option['name']}")
                            st.caption(option['price'])
                            
                            insights = option.get('ai_insights', {})
                            score = insights.get('score', 0)
                            st.progress(score / 10, text=f"AI Score: {score}/10")
                            
                            st.info(f"**Best For:** {insights.get('best_for', 'General Use')}")
                            
                            with st.expander("⚠️ Dealbreaker"):
                                st.warning(insights.get('dealbreaker', 'None found'))

                            if st.button("Select This", key=f"btn_{i}", use_container_width=True):
                                st.session_state.selected_product = option

            st.divider()
            st.subheader("📊 Comparison Analysis")
            
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
                    summary = p.get('fit_summary', '')
                    if summary:
                        st.info(summary)
                    
                    tech_specs = p.get('tech_specs', {})
                    if tech_specs:
                        st.markdown("#### 📋 Technical Specifications")
                        for key, value in tech_specs.items():
                            st.markdown(f"**{key}:** {value}")
                    
                    st.divider()

                    st.markdown("#### Key Benefits (Translated)")
                    details = p.get('full_details', "No details available.")
                    if isinstance(details, list):
                        details = "\n".join([f"- {item}" for item in details])
                    st.markdown(details)
                    
                    st.markdown("#### Gallery")
                    images = p.get('images', [])
                    if len(images) > 1:
                        g_cols = st.columns(3)
                        for j, img in enumerate(images[:3]):
                            with g_cols[j]:
                                st.image(img, use_container_width=True)

                with d_col2:
                    st.success(f"**Price:** {p['price']}")
                    st.link_button(f"🛒 Buy Now", p['link'], type="primary", use_container_width=True)

    except Exception as e:
        pass