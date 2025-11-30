import os
import json
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
    intent: str
    search_results: List[dict]
    final_recommendation: str
    critique: str
    revision_needed: bool
    retry_count: int

# --- AGENT 1: INTENT AGENT ---
def intent_agent(state: AgentState):
    print(f"DEBUG: Processing '{state['query']}'")
    
    prompt = ChatPromptTemplate.from_template(
        "You are ShopGenie-E. Analyze user input.\n"
        "INPUT: {query}\n"
        "HISTORY: {chat_history}\n\n"
        "CRITICAL PRIORITY RULES:\n"
        "1. CASUAL/GREETING? -> Output 'CHAT: [Response]'\n"
        "2. SHOPPING BUT MISSING BUDGET? -> If user wants to buy something but NO price/budget is mentioned (e.g. '$500', 'cheap', 'budget') in INPUT or HISTORY:\n"
        "   - Output 'ASK_BUDGET: To help you find the best option, do you have a specific price range in mind?'\n"
        "3. SHOPPING WITH BUDGET? -> Output 'SEARCH: [Refined Query]'\n"
    )
    chain = prompt | llm
    response = chain.invoke({
        "chat_history": state.get("chat_history", ""), 
        "query": state["query"]
    }).content.strip()
    
    if response.startswith("CHAT:"):
        return {"intent": "casual_chat", "final_recommendation": response.replace("CHAT:", "").strip(), "refined_query": ""}
    elif response.startswith("ASK_BUDGET:"):
        return {"intent": "ask_budget", "final_recommendation": response.replace("ASK_BUDGET:", "").strip(), "refined_query": ""}
    else:
        return {"intent": "buy_request", "refined_query": response.replace("SEARCH:", "").strip(), "retry_count": state.get("retry_count", 0)}

# --- AGENT 2: RETRIEVAL AGENT ---
def retrieval_agent(state: AgentState):
    query = state.get("refined_query", "")
    if not query: return {"search_results": []}
    
    # Generic search query works for all products
    search_query = f"{query} price buy online site:amazon.com OR site:bestbuy.com OR site:walmart.com"
    try:
        results = tavily.search(query=search_query, search_depth="advanced", max_results=7)
        return {"search_results": results['results']}
    except:
        return {"search_results": []}

# --- AGENT 3: REASONER AGENT (Dynamic Specs Fix) ---
def reasoner_agent(state: AgentState):
    print("DEBUG: Reasoner Agent thinking...")
    data = state["search_results"]
    
    prompt = ChatPromptTemplate.from_template(
        "You are ShopGenie-E. Return a JSON OBJECT only.\n"
        "GOAL: Provide top 3 options fitting the user's budget.\n"
        "CONTEXT: User asked '{query}'.\n"
        "\n"
        "RULES:\n"
        "1. Output valid JSON only.\n"
        "2. Structure: {{ 'options': [ {{ \n"
        "      'category': 'Powerhouse', \n"
        "      'name': '...', \n"
        "      'price': '...', \n"
        "      'summary': '...', \n"
        "      'link': '...', \n"
        "      'fit_summary': 'Why this fits...', \n"
        "      'full_details': ['Point 1', 'Point 2', 'Point 3', 'Point 4'],\n"
        "      'tech_specs': {{ 'Spec1': 'Value1', 'Spec2': 'Value2', ... }}, \n"
        "      'specs': {{ 'Performance': '...', 'Build_Quality': '...', 'Key_Feature': '...' }}, \n"
        "      'ai_insights': {{ 'score': 8, 'best_for': '...', 'dealbreaker': '...' }} \n"
        "   }} ] }}\n"
        "3. CATEGORY RULE: Use 'Powerhouse', 'Balanced', 'Budget'.\n"
        "4. DYNAMIC SPECS RULE (CRITICAL):\n"
        "   - 'tech_specs': Extract 4-5 raw technical specs RELEVANT TO THE PRODUCT TYPE.\n"
        "   - IF LAPTOP: Processor, RAM, Storage, Screen, Battery.\n"
        "   - IF HEADPHONES: Battery Life, Noise Cancellation, Driver Size, Connectivity, Weight.\n"
        "   - IF KEYBOARD: Switch Type, Connectivity, Backlight, Size.\n"
        "   - DO NOT output 'None' for fields that don't exist. Just pick different fields.\n"
        "5. TRANSLATION RULE: 'full_details' must explain benefits in plain English.\n"
        "\n"
        "SEARCH DATA: {data}"
    )
    chain = prompt | llm
    response = chain.invoke({"data": data, "query": state["refined_query"]})
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
    recommendation = state["final_recommendation"]
    retries = state.get("retry_count", 0)
    if retries >= 3: return {"revision_needed": False}

    try:
        clean_text = recommendation.replace("```json", "").replace("```", "").strip()
        json.loads(clean_text)
        return {"revision_needed": False, "critique": "None"}
    except:
        return {"revision_needed": True, "critique": "Invalid JSON", "retry_count": retries + 1}

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