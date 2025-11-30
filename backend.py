import os
import json
import re # Needed for the Budget Guard
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

# --- AGENT 1: INTENT AGENT (Restored Budget & Greeting Guards) ---
def intent_agent(state: AgentState):
    user_input = state['query'].strip().lower()
    history = state.get("chat_history", "").lower()
    print(f"DEBUG: Processing '{user_input}'")
    
    # 1. PYTHON GREETING GUARD (Fast & Reliable)
    greetings = ["hi", "hii", "hiii", "hey", "heyy", "hello", "hola", "greetings", "yo", "good morning", "good evening"]
    clean_input = user_input.strip("!.,?")
    
    if clean_input in greetings:
        return {
            "intent": "casual_chat",
            "final_recommendation": "Hello! I am ShopGenie-E. I can help you find the best electronics (Laptops, Headphones, etc.). What are you looking for?",
            "refined_query": ""
        }

    # 2. PYTHON BUDGET GUARD (The Logic You Wanted)
    # We check if the input contains price indicators or numbers
    price_keywords = ['$', 'usd', 'price', 'budget', 'cheap', 'expensive', 'cost', 'affordable', 'premium', 'under', 'over', 'less', 'more', 'range', 'spending', 'max']
    
    has_price_word = any(word in user_input for word in price_keywords)
    # Check for digits (e.g. "500", "1000") in the current input
    has_number = bool(re.search(r'\d+', user_input))
    
    # We ask the LLM: "Is this a shopping request?"
    # If YES, and Python says "No Price Found", we force ASK_BUDGET.
    
    check_prompt = ChatPromptTemplate.from_template(
        "Analyze text: '{query}'.\n"
        "Is the user looking to buy or find a product? Answer YES or NO only."
    )
    is_shopping = (check_prompt | llm).invoke({"query": state["query"]}).content.strip().upper()
    
    if "YES" in is_shopping and not (has_price_word or has_number):
        # User wants to buy but didn't give a number -> Ask for it.
        return {
            "intent": "ask_budget",
            "final_recommendation": "To find the best option for you, could you please specify your **budget**? (e.g. 'under $500' or 'cheap')",
            "refined_query": ""
        }

    # 3. IF WE GOT HERE, IT'S A VALID SEARCH (OR COMPLEX CHAT)
    refine_prompt = ChatPromptTemplate.from_template(
        "You are ShopGenie-E. User Input: {query}\n"
        "History: {chat_history}\n\n"
        "Task: Create a refined search query for the product.\n"
        "Output: Just the search string."
    )
    chain = refine_prompt | llm
    refined_query = chain.invoke({
        "chat_history": state.get("chat_history", ""), 
        "query": state["query"]
    }).content.strip()
    
    return {
        "intent": "buy_request", 
        "refined_query": refined_query,
        "retry_count": state.get("retry_count", 0)
    }

# --- AGENT 2: RETRIEVAL AGENT ---
def retrieval_agent(state: AgentState):
    query = state.get("refined_query", "")
    critique = state.get("critique", "")
    
    if not query: return {"search_results": []}

    if state.get("revision_needed"):
        print(f"DEBUG: Retrying search: {critique}")
        search_query = f"{query} buy page amazon bestbuy walmart {critique}"
    else:
        print("DEBUG: Initial search...")
        search_query = f"{query} price buy online site:amazon.com OR site:bestbuy.com OR site:walmart.com"
    
    try:
        results = tavily.search(query=search_query, search_depth="advanced", max_results=7)
        return {"search_results": results['results']}
    except:
        return {"search_results": []}

# --- AGENT 3: REASONER AGENT (Dynamic Specs + Translations) ---
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
        "4. DYNAMIC SPECS (CRITICAL): \n"
        "   - IF LAPTOP: Processor, RAM, Storage, Screen.\n"
        "   - IF HEADPHONES: Battery Life, ANC, Driver, Weight.\n"
        "   - IF KEYBOARD: Switch Type, Connectivity, Size.\n"
        "5. TRANSLATE BENEFITS: 'full_details' must explain specs in plain English (e.g. '30h Battery: Lasts a whole week').\n"
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
    try:
        json.loads(state["final_recommendation"].replace("```json", "").replace("```", "").strip())
        return {"revision_needed": False}
    except:
        return {"revision_needed": True, "critique": "Invalid JSON", "retry_count": state.get("retry_count", 0) + 1}

# --- GRAPH CONSTRUCTION ---
def route_intent(state):
    # Stops the graph if it is just a greeting or budget question
    if state["intent"] in ["casual_chat", "ask_budget"]:
        return END
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

workflow.add_conditional_edges(
    "intent_agent", 
    route_intent, 
    {
        END: END, 
        "retrieval_agent": "retrieval_agent"
    }
)

workflow.add_edge("retrieval_agent", "reasoner_agent")
workflow.add_edge("reasoner_agent", "image_fetcher_agent")
workflow.add_edge("image_fetcher_agent", "evaluator_agent")
workflow.add_conditional_edges("evaluator_agent", decide_next_step, {"retrieval_agent": "retrieval_agent", END: END})

app = workflow.compile()