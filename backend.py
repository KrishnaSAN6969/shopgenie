import os
import json
import re # NEW: For checking numbers
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

# --- AGENT 1: INTENT AGENT (With Python Budget Guard) ---
def intent_agent(state: AgentState):
    user_input = state['query'].strip().lower()
    history = state.get("chat_history", "").lower()
    print(f"DEBUG: Processing '{user_input}'")
    
    # 1. GREETING GUARD
    greetings = ["hi", "hii", "hiii", "hey", "heyy", "hello", "hola", "greetings", "yo"]
    clean_input = user_input.strip("!.,?")
    if clean_input in greetings:
        return {
            "intent": "casual_chat",
            "final_recommendation": "Hello! I am ShopGenie-E. How can I help you find the perfect electronics today?",
            "refined_query": ""
        }

    # 2. PYTHON BUDGET GUARD (The Fix)
    # We check if the input OR history contains specific price indicators
    price_keywords = ['$', 'usd', 'price', 'budget', 'cheap', 'expensive', 'cost', 'affordable', 'premium', 'under', 'over', 'less', 'more', 'range', 'spending', 'max']
    
    has_price_word = any(word in user_input or word in history for word in price_keywords)
    has_number = bool(re.search(r'\d+', user_input)) or bool(re.search(r'\d+', history))
    
    # If it looks like a shopping request (has product words) but NO price info:
    if not (has_price_word or has_number):
        return {
            "intent": "ask_budget",
            "final_recommendation": "To find the best option for you, could you please specify your **budget**? (e.g. 'under $500' or 'cheap')",
            "refined_query": ""
        }

    # 3. LLM ANALYSIS (Only runs if budget check passed)
    prompt = ChatPromptTemplate.from_template(
        "You are ShopGenie-E. Analyze user input.\n"
        "INPUT: {query}\n"
        "HISTORY: {chat_history}\n\n"
        "TASK: Generate a search query.\n"
        "OUTPUT: Just the search query string (e.g. 'gaming laptop under 1000')."
    )
    chain = prompt | llm
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


# --- AGENT 3: REASONER AGENT ---
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
        "      'fit_summary': 'This is the perfect choice for you because...', \n"
        "      'full_details': ['Point 1', 'Point 2', 'Point 3', 'Point 4'],\n"
        "      'tech_specs': {{ 'Processor': '...', 'RAM': '...', 'Storage': '...', 'Display': '...', 'Battery': '...' }}, \n"
        "      'specs': {{ 'Performance': '...', 'Build_Quality': '...', 'Key_Feature': '...' }}, \n"
        "      'ai_insights': {{ 'score': 8, 'best_for': '...', 'dealbreaker': '...' }} \n"
        "   }} ] }}\n"
        "3. CATEGORY RULE: Use 'Powerhouse', 'Balanced', 'Budget'.\n"
        "4. CONTENT RULES: \n"
        "   - 'fit_summary': Short paragraph (2 sentences) on why it matches the user.\n"
        "   - 'full_details': Translate benefits (e.g. '8GB RAM: Good for multitasking').\n"
        "   - 'tech_specs': Raw technical data (e.g. 'Intel Core i5-1235U').\n"
        "5. INSIGHTS RULE:\n"
        "   - 'score': 1-10 integer.\n"
        "   - 'dealbreaker': Honest warning.\n"
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
    # Stop immediately if chat or budget question
    if state["intent"] in ["casual_chat", "ask_budget"]:
        return END
    return "retrieval_agent"

def decide_next_step(state):
    if state["revision_needed"]:
        return "retrieval_agent"
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