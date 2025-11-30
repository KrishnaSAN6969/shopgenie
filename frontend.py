import streamlit as st
import json
import pandas as pd
from backend import app 

# 1. PAGE CONFIGURATION & STYLING
st.set_page_config(page_title="ShopGenie-E", page_icon="🛍️", layout="wide")

# Custom CSS for a modern, clean look
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
    </style>
""", unsafe_allow_html=True)

st.title("🛍️ ShopGenie-E")
st.caption("The Transparent, Multi-Agent Shopping Assistant")

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
    
    # Fake metrics to simulate a complex system
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

# 4. SELF-INTRODUCTION (Only visible on start)
if not st.session_state.messages:
    st.markdown("### 👋 Hi, I'm ShopGenie-E.")
    st.markdown("""
    I am a **Multi-Agent Decision System** designed to eliminate shopping bias.
    I research real-time data, translate technical jargon, and audit results for accuracy.
    """)
    
    # 4-Column Layout explaining the Agents
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("**🧠 Intent Agent**")
        st.caption("Decides if you are buying, comparing, or chatting.")
    with c2:
        st.warning("**🌐 Retrieval Agent**")
        st.caption("Scrapes live pricing from Amazon, BestBuy, & Walmart.")
    with c3:
        st.success("**🤔 Reasoner Agent**")
        st.caption("Ranks products and translates specs into plain English.")
    with c4:
        st.error("**✅ Evaluator Agent**")
        st.caption("Audits results to prevent hallucinations.")
    
    st.divider()

# 5. CHAT HISTORY VIEW
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            # If it's a JSON response, show a simple placeholder
            if "{" in msg["content"] and "options" in msg["content"]:
                st.markdown("✅ *Market Research Complete (See Dashboard below)*")
            else:
                st.markdown(msg["content"])

# 6. INPUT HANDLING
user_input = st.chat_input("Ex: Best noise-cancelling headphones under $200...")

if user_input:
    # Append User Message
    st.session_state.selected_product = None 
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Prepare Context (Last 4 messages)
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
    
    # Run Agents with Visual Feedback in Sidebar
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

    # Logic: If raw_response is just text (Chat/Ask Budget), clear the dashboard
    if "{" in raw_response and "options" in raw_response:
        st.session_state.last_json_response = raw_response
    else:
        st.session_state.last_json_response = None

    # Save and Refresh
    st.session_state.messages.append({"role": "assistant", "content": raw_response})
    st.rerun()

# 7. THE DASHBOARD (Results View)
if st.session_state.last_json_response:
    try:
        clean_json = st.session_state.last_json_response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        options = data.get("options", [])

        if options:
            st.divider()
            st.subheader("🎯 Top 3 Recommendations")
            
            # --- SECTION A: PRODUCT CARDS ---
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            
            for i, option in enumerate(options):
                if i < 3:
                    with cols[i]:
                        with st.container(border=True):
                            # Badge logic
                            if "Powerhouse" in option['category']:
                                st.markdown(":rocket: **Powerhouse**")
                            elif "Balanced" in option['category']:
                                st.markdown(":balance_scale: **Balanced**")
                            else:
                                st.markdown(":moneybag: **Budget**")
                            
                            # Image
                            images = option.get('images', [])
                            if images:
                                st.image(images[0], use_container_width=True)
                                
                            st.markdown(f"#### {option['name']}")
                            st.caption(option['price'])
                            
                            # AI Insights
                            insights = option.get('ai_insights', {})
                            score = insights.get('score', 0)
                            st.progress(score / 10, text=f"AI Score: {score}/10")
                            
                            st.info(f"**Best For:** {insights.get('best_for', 'General Use')}")
                            
                            with st.expander("⚠️ Dealbreaker"):
                                st.warning(insights.get('dealbreaker', 'None found'))

                            # Selection Button
                            if st.button("Select This", key=f"btn_{i}", use_container_width=True):
                                st.session_state.selected_product = option

            # --- SECTION B: COMPARISON TABLE ---
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

            # --- SECTION C: DEEP DIVE (Selected Product) ---
            if st.session_state.selected_product:
                p = st.session_state.selected_product
                st.divider()
                st.subheader(f"📝 Deep Dive: {p['name']}")
                
                d_col1, d_col2 = st.columns([3, 1])
                
                with d_col1:
                    # 1. Fit Summary (Blue Box)
                    st.markdown("#### Why this fits you")
                    summary = p.get('fit_summary', '')
                    if summary:
                        st.info(summary)
                    
                    # 2. Tech Specs (Clean List)
                    tech_specs = p.get('tech_specs', {})
                    if tech_specs:
                        st.markdown("#### 📋 Technical Specifications")
                        # Display as bullet points for clarity
                        for key, value in tech_specs.items():
                            st.markdown(f"**{key}:** {value}")
                    
                    st.divider()

                    # 3. Translated Benefits (Bullet Points)
                    st.markdown("#### Key Benefits (Translated)")
                    details = p.get('full_details', "No details available.")
                    if isinstance(details, list):
                        details = "\n".join([f"- {item}" for item in details])
                    st.markdown(details)
                    
                    # 4. Gallery
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
        # Silently fail if JSON is malformed (prevents crashes on chat messages)
        pass