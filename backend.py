import os
import json
import re
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient
from langgraph.graph import StateGraph, END

load_dotenv()

# --- CONFIGURATION ---
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile", 
    api_key=os.environ["GROQ_API_KEY"]
)

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    query: str
    chat_history: str
    refined_query: str
    use_case: str           # NEW: Tracks "Coding", "Gaming", etc.
    intent: str
    search_results: List[dict]
    final_recommendation: str
    critique: str
    revision_needed: bool
    retry_count: int

# --- AGENT 1: INTENT & CONTEXT MANAGER ---
def intent_agent(state: AgentState):
    user_input = state['query'].strip().lower()
    history = state.get("chat_history", "").lower()
    print(f"DEBUG: Processing '{user_input}'")
    
    # 1. GREETING GUARD
    greetings = ["hi", "hii", "hiii", "hey", "heyy", "hello", "hola", "greetings", "yo", "good morning"]
    clean_input = user_input.strip("!.,?")
    if clean_input in greetings:
        return {
            "intent": "casual_chat",
            "final_recommendation": "Hello! I am ShopGenie-E. Tell me what you do (e.g., 'I am a student' or 'I edit videos') and I'll find the perfect gear for you.",
            "refined_query": "",
            "use_case": "general"
        }

    # 2. BUDGET & USE CASE EXTRACTION
    # We ask the LLM to extract the specific USE CASE (Job/Activity)
    extraction_prompt = ChatPromptTemplate.from_template(
        "Analyze this user query: '{query}'\n"
        "History: {chat_history}\n\n"
        "TASK 1: Extract the USE CASE (e.g. 'Gaming', 'Coding', 'Video Editing', 'Student', 'General Use'). Default to 'General Use' if unclear.\n"
        "TASK 2: Check for Budget.\n"
        "TASK 3: Output format: 'USE_CASE: [Use Case] | REFINED_SEARCH: [Search Query]'\n"
        "   - Example: 'USE_CASE: Coding | REFINED_SEARCH: best laptops for coding under 1000'\n"
        "   - Example: 'USE_CASE: Gaming | REFINED_SEARCH: gaming laptop with rtx 4060'\n"
    )
    chain = extraction_prompt | llm
    analysis = chain.invoke({
        "chat_history": state.get("chat_history", ""), 
        "query": state["query"]
    }).content.strip()
    
    # Parse the LLM output
    use_case = "General Use"
    refined_query = state["query"]
    
    if "USE_CASE:" in analysis and "REFINED_SEARCH:" in analysis:
        parts = analysis.split("|")
        use_case = parts[0].replace("USE_CASE:", "").strip()
        refined_query = parts[1].replace("REFINED_SEARCH:", "").strip()

    # 3. BUDGET GUARD
    price_keywords = ['$', 'usd', 'price', 'budget', 'cheap', 'expensive', 'cost', 'under', 'over', 'less', 'max']
    has_price = any(w in user_input or w in history for w in price_keywords) or bool(re.search(r'\d+', user_input))
    
    # If looking to buy but no budget found
    check_shop = ChatPromptTemplate.from_template("Is '{query}' a shopping request? YES/NO.")
    is_shopping = (check_shop | llm).invoke({"query": state["query"]}).content.strip().upper()
    
    if "YES" in is_shopping and not has_price:
        return {
            "intent": "ask_budget",
            "final_recommendation": f"I see you are looking for something for **{use_case}**. To give you the best recommendation, what is your price range?",
            "refined_query": "",
            "use_case": use_case
        }

    return {
        "intent": "buy_request", 
        "refined_query": refined_query,
        "use_case": use_case,
        "retry_count": state.get("retry_count", 0)
    }

# --- AGENT 2: RETRIEVAL AGENT (Use-Case Aware) ---
def retrieval_agent(state: AgentState):
    query = state.get("refined_query", "")
    use_case = state.get("use_case", "general")
    
    if not query: return {"search_results": []}

    print(f"DEBUG: Searching for '{query}' | Context: {use_case}")
    
    # We explicitly add the use case to the search to filter bad results early
    search_query = f"{query} best for {use_case} price reviews site:amazon.com OR site:bestbuy.com OR site:walmart.com"
    
    try:
        results = tavily.search(query=search_query, search_depth="advanced", max_results=8)
        return {"search_results": results['results']}
    except:
        return {"search_results": []}

# --- AGENT 3: REASONER AGENT (Strict Suitability Filter) ---
def reasoner_agent(state: AgentState):
    print("DEBUG: Reasoner Agent thinking...")
    data = state["search_results"]
    use_case = state.get("use_case", "General Use")
    
    prompt = ChatPromptTemplate.from_template(
        "You are ShopGenie-E. Return a JSON OBJECT only.\n"
        "GOAL: Provide top 3 options fitting the user's budget AND USE CASE: '{use_case}'.\n"
        "\n"
        "CRITICAL SUITABILITY RULES:\n"
        "1. IF Use Case is 'Gaming': DISCARD any laptop without a dedicated GPU (RTX/GTX). Chromebooks are BANNED.\n"
        "2. IF Use Case is 'Video Editing/3D': DISCARD anything with less than 16GB RAM.\n"
        "3. IF Use Case is 'Student/Travel': Prioritize Battery Life and Weight.\n"
        "4. IF Use Case is 'Coding': Prioritize Processor Speed and Screen Real Estate.\n"
        "\n"
        "JSON STRUCTURE:\n"
        "{{ 'options': [ {{ \n"
        "      'category': 'Powerhouse', \n"
        "      'name': '...', \n"
        "      'price': '...', \n"
        "      'summary': '...', \n"
        "      'link': '...', \n"
        "      'fit_summary': 'Since you need this for {use_case}, this fits because...', \n"
        "      'full_details': ['Point 1', 'Point 2', 'Point 3', 'Point 4'],\n"
        "      'tech_specs': {{ 'Spec1': '...', 'Spec2': '...' }}, \n"
        "      'specs': {{ 'Performance': '...', 'Build_Quality': '...', 'Key_Feature': '...' }}, \n"
        "      'ai_insights': {{ 'score': 8, 'best_for': '...', 'dealbreaker': '...' }} \n"
        "   }} ] }}\n"
        "\n"
        "SEARCH DATA: {data}"
    )
    chain = prompt | llm
    response = chain.invoke({
        "data": data, 
        "use_case": use_case
    })
    return {"final_recommendation": response.content}

# --- AGENT 3.5: IMAGE FETCHER AGENT ---
def image_fetcher_agent(state: AgentState):
    recommendation = state["final_recommendation"]
    try:
        clean_text = recommendation.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        for option in data.get('options', []):
            product_name = option.get('name')
            response = tavily.search(
                query=f"{product_name} product photo white background", 
                search_depth="basic", include_images=True, max_results=1
            )
            option['images'] = response.get('images', [])
        return {"final_recommendation": json.dumps(data)}
    except:
        return {"final_recommendation": recommendation} 

# --- AGENT 4: EVALUATOR AGENT ---
def evaluator_agent(state: AgentState):
    try:
        json.loads(state["final_recommendation"].replace("```json", "").replace("```", "").strip())
        return {"revision_needed": False}
    except:
        return {"revision_needed": True, "critique": "Invalid JSON", "retry_count": state.get("retry_count", 0) + 1}

# --- GRAPH CONSTRUCTION ---
def route_intent(state):
    if state["intent"] in ["casual_chat", "ask_budget"]: return END
    return "retrieval_agent"

def decide_next_step(state):
    if state["revision_needed"]: return "retrieval_agent"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("intent_agent", intent_agent)
workflow.add_node("retrieval_agent", retrieval_agent)
workflow.add_node("reasoner_agent", reasoner_agent)
workflow.add_node("image_fetcher_agent", image_fetcher_agent)
workflow.add_node("evaluator_agent", evaluator_agent)

workflow.set_entry_point("intent_agent")

workflow.add_conditional_edges("intent_agent", route_intent, {END: END, "retrieval_agent": "retrieval_agent"})
workflow.add_edge("retrieval_agent", "reasoner_agent")
workflow.add_edge("reasoner_agent", "image_fetcher_agent")
workflow.add_edge("image_fetcher_agent", "evaluator_agent")
workflow.add_conditional_edges("evaluator_agent", decide_next_step, {"retrieval_agent": "retrieval_agent", END: END})

app = workflow.compile()