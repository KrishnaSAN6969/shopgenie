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
# Use "llama-3.3-70b-versatile" for intelligence.
# Use "llama-3.1-8b-instant" if you hit Rate Limits.
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

# --- AGENT 1: INTENT & MEMORY MANAGER ---
def intent_agent(state: AgentState):
    user_input = state['query'].strip().lower()
    history = state.get("chat_history", "").lower()
    print(f"DEBUG: Processing '{user_input}'")
    
    # 1. GREETING GUARD (Instant Reply)
    greetings = ["hi", "hii", "hiii", "hey", "heyy", "hello", "hola", "greetings", "yo", "good morning"]
    clean_input = user_input.strip("!.,?")
    if clean_input in greetings:
        return {
            "intent": "casual_chat",
            "final_recommendation": "Hello! I am ShopGenie-E. I can help you find the best electronics. What are you looking for today?",
            "refined_query": ""
        }

    # 2. BUDGET GUARD (Forces user to give price)
    price_keywords = ['$', 'usd', 'price', 'budget', 'cheap', 'expensive', 'cost', 'affordable', 'premium', 'under', 'over', 'less', 'spending', 'max']
    has_price_word = any(word in user_input or word in history for word in price_keywords)
    has_number = bool(re.search(r'\d+', user_input)) or bool(re.search(r'\d+', history))
    
    # Quick check if it's a shopping query
    check_prompt = ChatPromptTemplate.from_template("Is '{query}' a request to buy/find a product? Answer YES or NO.")
    is_shopping = (check_prompt | llm).invoke({"query": state["query"]}).content.strip().upper()
    
    if "YES" in is_shopping and not (has_price_word or has_number):
        return {
            "intent": "ask_budget",
            "final_recommendation": "To help you find the best option, could you please specify your **budget**? (e.g. 'under $500' or 'cheap')",
            "refined_query": ""
        }

    # 3. QUERY REFINEMENT (Uses History)
    refine_prompt = ChatPromptTemplate.from_template(
        "You are ShopGenie-E. User Input: {query}\n"
        "History: {chat_history}\n\n"
        "Task: Create a refined search query for the product. Combine history if needed (e.g. 'cheaper' -> 'cheaper gaming laptop').\n"
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

# --- AGENT 3: REASONER AGENT (The Mega-Brain) ---
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
        "      'fit_summary': 'Write 3 sentences explaining why this fits the user...', \n"
        "      'full_details': ['Benefit 1', 'Benefit 2', 'Benefit 3', 'Benefit 4'],\n"
        "      'tech_specs': {{ 'Spec1': 'Value1', 'Spec2': 'Value2', ... }}, \n"
        "      'specs': {{ 'Performance': '...', 'Build_Quality': '...', 'Key_Feature': '...' }}, \n"
        "      'ai_insights': {{ 'score': 8, 'best_for': '...', 'dealbreaker': '...' }} \n"
        "   }} ] }}\n"
        "3. CATEGORY RULE: Use 'Powerhouse', 'Balanced', 'Budget'.\n"
        "4. DYNAMIC SPECS: Extract 4-5 raw specs relevant to the product (Laptop=CPU/RAM, Audio=Battery/ANC).\n"
        "5. NOOB TRANSLATION: 'full_details' must explain benefits in plain English.\n"
        "\n"
        "SEARCH DATA: {data}"
    )
    chain = prompt | llm
    response = chain.invoke({"data": data, "query": state["refined_query"]})
    return {"final_recommendation": response.content}

# --- AGENT 3.5: IMAGE FETCHER AGENT ---
def image_fetcher_agent(state: AgentState):
    print("DEBUG: Image Fetcher looking for photos...")
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
    # Stop graph if just chatting or asking for budget
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