import streamlit as st
import json
from backend import app 

st.set_page_config(page_title="ShopGenie-E", page_icon="🛍️", layout="wide")
st.markdown("""<style>.stButton>button {border-radius: 20px; font-weight: 600; border: 1px solid #ddd;}.agent-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 10px;}</style>""", unsafe_allow_html=True)

st.title("🛍️ ShopGenie-E")
st.caption("Powered by Multi-Agent AI & Real-Time Search")

if "messages" not in st.session_state: st.session_state.messages = []
if "selected_product" not in st.session_state: st.session_state.selected_product = None
if "last_json_response" not in st.session_state: st.session_state.last_json_response = None

with st.sidebar:
    st.header("Agent Activity")
    status_box = st.empty()
    st.divider()
    if st.button("Start New Search"):
        st.session_state.messages = []
        st.session_state.selected_product = None
        st.session_state.last_json_response = None
        st.rerun()

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"): st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            if "{" in msg["content"] and "options" in msg["content"]:
                st.markdown("✅ *Analysis complete (See Dashboard below)*")
            else:
                st.markdown(msg["content"])

user_input = st.chat_input("What are you looking for today?")

if user_input:
    st.session_state.selected_product = None 
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    
    recent_history = st.session_state.messages[-4:] 
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent_history])
    
    inputs = {"query": user_input, "chat_history": history_text}
    raw_response = ""
    
    with st.spinner("Agents are researching..."):
        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "intent_agent":
                    intent = value.get("intent", "")
                    if intent == "casual_chat": status_box.info("💬 Intent: Casual Chat")
                    elif intent == "ask_budget": status_box.warning("💰 Intent: Asking Budget")
                    else: status_box.info(f"🧠 Context: {value['refined_query']}")
                elif key == "retrieval_agent": status_box.info(f"🌐 Found {len(value['search_results'])} items")
                elif key == "reasoner_agent": status_box.info("🤔 Analyzing & Scoring...")
                elif key == "image_fetcher_agent": status_box.info("🖼️ Fetching Photos...")
                if "final_recommendation" in value: raw_response = value["final_recommendation"]

    # LOGIC FIX: ONLY Show Dashboard if it is REAL product JSON
    if "{" in raw_response and "options" in raw_response:
        st.session_state.last_json_response = raw_response
    else:
        st.session_state.last_json_response = None

    st.session_state.messages.append({"role": "assistant", "content": raw_response})
    st.rerun()

# DASHBOARD
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
                            with st.expander("⚠️ Dealbreaker"): st.warning(insights.get('dealbreaker', 'None found'))
                            if st.button("Select This", key=f"btn_{i}", use_container_width=True): st.session_state.selected_product = option

            if st.session_state.selected_product:
                p = st.session_state.selected_product
                st.divider()
                st.subheader(f"📝 Deep Dive: {p['name']}")
                d_col1, d_col2 = st.columns([3, 1])
                with d_col1:
                    st.markdown("#### Why this fits you")
                    details = p.get('full_details', "No details.")
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
    except: pass