import streamlit as st
import json
import pandas as pd
from backend import app 

st.set_page_config(page_title="ShopGenie-E", page_icon="🛍️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stButton>button { border-radius: 20px; font-weight: bold; border: 1px solid #ddd; }
    </style>
""", unsafe_allow_html=True)

st.title("🛍️ ShopGenie-E")
st.caption("Powered by Multi-Agent AI & Real-Time Search")

if "messages" not in st.session_state: st.session_state.messages = []
if "selected_product" not in st.session_state: st.session_state.selected_product = None
if "last_json_response" not in st.session_state: st.session_state.last_json_response = None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/brain.png", width=80)
    st.title("System Monitor")
    st.divider()
    if st.button("Start New Search"):
        st.session_state.messages = []
        st.session_state.selected_product = None
        st.session_state.last_json_response = None
        st.rerun()

# Render History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            if "{" in msg["content"]:
                st.markdown("✅ *Market Research Complete (See Dashboard below)*")
            else:
                st.markdown(msg["content"])

# User Input
user_input = st.chat_input("Ex: Best gaming laptop under $1200...")

if user_input:
    st.session_state.selected_product = None 
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    recent_history = st.session_state.messages[-4:] 
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent_history])
    
    inputs = {"query": user_input, "chat_history": history_text}
    
    raw_response = ""
    status_container = st.empty()
    
    with st.spinner("Processing..."):
        for output in app.stream(inputs):
            for key, value in output.items():
                # Visual updates
                if key == "intent_agent":
                    # If it's just chat, show simple status
                    if value.get("intent") == "casual_chat":
                         status_container.info("💬 Intent: Casual Chat detected.")
                    else:
                         status_container.info(f"🧠 Context: {value['refined_query']}")
                elif key == "retrieval_agent":
                    status_container.warning(f"🌐 Retrieval: Found {len(value['search_results'])} products")
                elif key == "reasoner_agent":
                    status_container.success("🤔 Reasoner: Analyzing trade-offs...")
                
                if "final_recommendation" in value:
                    raw_response = value["final_recommendation"]
    
    status_container.empty() # Clear status
    st.session_state.last_json_response = raw_response
    st.session_state.messages.append({"role": "assistant", "content": raw_response})
    st.rerun()

# 6. DASHBOARD VIEW (Safe Mode)
# Only render dashboard if the response actually looks like JSON
if st.session_state.last_json_response and "{" in st.session_state.last_json_response:
    try:
        clean_json = st.session_state.last_json_response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        options = data.get("options", [])

        if options:
            st.divider()
            st.markdown("### 🎯 Top 3 Recommendations")
            
            # Cards
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
                            st.progress(score / 10, text=f"AI Score: {score}/10")
                            
                            st.info(f"**Best For:** {insights.get('best_for', 'General Use')}")
                            
                            with st.expander("⚠️ Dealbreaker"):
                                st.warning(insights.get('dealbreaker', 'None found'))

                            if st.button("Select This", key=f"btn_{i}", use_container_width=True):
                                st.session_state.selected_product = option

            # Table
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

            # Details
            if st.session_state.selected_product:
                p = st.session_state.selected_product
                st.divider()
                st.markdown(f"### 📝 Deep Dive: {p['name']}")
                d_col1, d_col2 = st.columns([3, 1])
                with d_col1:
                    st.markdown("#### Why this fits you")
                    details = p.get('full_details', "No details available.")
                    if isinstance(details, list): details = "\n".join([f"- {item}" for item in details])
                    st.markdown(details)
                    
                    st.markdown("#### Gallery")
                    images = p.get('images', [])
                    if len(images) > 1:
                        g_cols = st.columns(3)
                        for j, img in enumerate(images[:3]):
                            with g_cols[j]: st.image(img, use_container_width=True)
                with d_col2:
                    st.success(f"**Price:** {p['price']}")
                    st.link_button(f"🛒 Buy Now", p['link'], type="primary", use_container_width=True)

    except Exception as e:
        # If it's just text (Casual Chat), this pass catches it silently
        pass