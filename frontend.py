import streamlit as st
import json
import pandas as pd
from backend import app 

# 1. Page Configuration
st.set_page_config(page_title="ShopGenie-E", page_icon="🛍️", layout="wide")

st.title("🛍️ ShopGenie-E")
st.caption("Powered by Multi-Agent AI & Real-Time Search")

# 2. Session State (RAM Only - No Saving to File)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_product" not in st.session_state:
    st.session_state.selected_product = None
if "last_json_response" not in st.session_state:
    st.session_state.last_json_response = None

# 3. Sidebar
with st.sidebar:
    st.header("Agent Activity")
    status_box = st.empty()
    st.divider()
    
    # Simple Clear Button (Resets RAM)
    if st.button("Start New Search"):
        st.session_state.messages = []
        st.session_state.selected_product = None
        st.session_state.last_json_response = None
        st.rerun()

# 4. CHAT HISTORY (Simplified Text View)
# We show past messages so you can see the conversation flow,
# but we keep the heavy UI for the "Current Result" below.
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            if "{" in msg["content"]:
                st.markdown("✅ *Analysis complete (See Dashboard below)*")
            else:
                st.markdown(msg["content"])

# 5. INPUT HANDLING
user_input = st.chat_input("What are you looking for today?")

if user_input:
    # Append User Message
    st.session_state.selected_product = None 
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Prepare Context (Last 4 messages from RAM)
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
    
    # Run Agents
    with st.spinner("Agents are researching..."):
        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "intent_agent":
                    status_box.info(f"🧠 Context: {value['refined_query']}")
                elif key == "retrieval_agent":
                    status_box.info(f"🌐 Found {len(value['search_results'])} items")
                elif key == "reasoner_agent":
                    status_box.info("🤔 Analyzing & Scoring...")
                elif key == "image_fetcher_agent":
                    status_box.info("🖼️ Fetching Photos...")
                
                if "final_recommendation" in value:
                    raw_response = value["final_recommendation"]

    # Save Assistant Response
    st.session_state.last_json_response = raw_response
    st.session_state.messages.append({"role": "assistant", "content": raw_response})
    
    # Force rerun to update the dashboard
    st.rerun()

# 6. DASHBOARD VIEW (The "Big Info" View)
if st.session_state.last_json_response:
    try:
        clean_json = st.session_state.last_json_response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        options = data.get("options", [])

        if options:
            st.divider()
            st.subheader("🎯 Top 3 Recommendations")
            
            # --- A. PRODUCT CARDS ---
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            
            for i, option in enumerate(options):
                if i < 3:
                    with cols[i]:
                        with st.container(border=True):
                            # Badge
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
                            
                            # AI Insights (Score + Best For)
                            insights = option.get('ai_insights', {})
                            score = insights.get('score', 0)
                            st.progress(score / 10, text=f"AI Score: {score}/10")
                            
                            best_for = insights.get('best_for', 'General Use')
                            st.info(f"**Best For:** {best_for}")
                            
                            # Dealbreaker Warning
                            dealbreaker = insights.get('dealbreaker', 'None found')
                            with st.expander("⚠️ Dealbreaker"):
                                st.warning(dealbreaker)

                            # Selection Button
                            if st.button("Select This", key=f"btn_{i}", use_container_width=True):
                                st.session_state.selected_product = option

            # --- B. COMPARISON TABLE ---
            st.divider()
            st.subheader("📊 Comparison Analysis")
            
            table_data = []
            for opt in options:
                specs = opt.get('specs', {})
                row = {
                    "Category": opt.get('category', 'Option'),
                    "Product": opt['name'],
                    "Price": opt['price'],
                    "Performance": specs.get('Performance', 'Standard'),
                    "Build Quality": specs.get('Build_Quality', 'Standard'),
                    "Key Feature": specs.get('Key_Feature', 'Standard'), 
                }
                table_data.append(row)
            
            if table_data:
                df = pd.DataFrame(table_data)
                st.table(df.set_index("Category"))

            # --- C. DETAILED ANALYSIS ---
            if st.session_state.selected_product:
                p = st.session_state.selected_product
                st.divider()
                st.subheader(f"📝 Deep Dive: {p['name']}")
                
                d_col1, d_col2 = st.columns([3, 1])
                
                with d_col1:
                    # Explanation of "Why you need to buy it"
                    st.markdown("### Why this fits you")
                    
                    details = p.get('full_details', "No details available.")
                    if isinstance(details, list):
                        details = "\n".join([f"- {item}" for item in details])
                    st.markdown(details)
                    
                    # Gallery
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
        st.error("Error displaying dashboard.")