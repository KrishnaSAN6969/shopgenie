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
# We are sticking with the powerful 70b model as requested
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile", 
    api_key=os.environ["GROQ_API_KEY"]
)

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    query: str              # Raw user input
    chat_history: str       # Conversation history
    refined_query: str      # Search-optimized query
    intent: str
    search_results: List[dict]
    final_recommendation: str
    critique: str
    revision_needed: bool
    retry_count: int

# --- AGENT 1: INTENT & MEMORY MANAGER ---
def intent_agent(state: AgentState):
    print(f"DEBUG: Processing '{state['query']}' with History.")
    
    # Refine the query using the 70b model's superior understanding
    refine_prompt = ChatPromptTemplate.from_template(
        "You are a Search Query Refiner. \n"
        "CHAT HISTORY:\n{chat_history}\n\n"
        "CURRENT USER INPUT: {query}\n\n"
        "TASK: Rewrite the user input to be a standalone search query. \n"
        "If the user says 'cheaper one' or 'compare them', use the history to know WHAT they are referring to.\n"
        "If the history is empty, just return the user input as is.\n"
        "OUTPUT: Only the rewritten query string."
    )
    chain = refine_prompt | llm
    refined_query = chain.invoke({
        "chat_history": state["chat_history"], 
        "query": state["query"]
    }).content.strip()
    
    print(f"DEBUG: Refined Query: {refined_query}")
    return {"refined_query": refined_query, "intent": "buy_request", "retry_count": state.get("retry_count", 0)}

# --- AGENT 2: RETRIEVAL AGENT ---
def retrieval_agent(state: AgentState):
    query = state["refined_query"]
    critique = state.get("critique", "")
    
    if state.get("revision_needed"):
        print(f"DEBUG: Retrying search: {critique}")
        search_query = f"{query} buy page amazon bestbuy walmart {critique}"
    else:
        print("DEBUG: Initial search...")
        search_query = f"{query} price buy online site:amazon.com OR site:bestbuy.com OR site:walmart.com"
    
    results = tavily.search(query=search_query, search_depth="advanced", max_results=7)
    return {"search_results": results['results']}

# --- AGENT 3: REASONER AGENT (The AI Analyst) ---
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
        "      'full_details': ['Point 1', 'Point 2'],\n"
        "      'specs': {{ 'Performance': '...', 'Build_Quality': '...', 'Key_Feature': '...' }}, \n"
        "      'ai_insights': {{ 'score': 8, 'best_for': '...', 'dealbreaker': '...' }} \n"
        "   }} ] }}\n"
        "3. AI INSIGHTS RULES:\n"
        "   - 'score': Give an integer rating (1-10) based on reviews you read.\n"
        "   - 'best_for': 2-3 words describing the perfect user (e.g., 'Frequent Flyers', 'Bass Heads').\n"
        "   - 'dealbreaker': 1 short sentence about the biggest DOWNSIDE (e.g., 'Battery life is below average', 'No water resistance'). Be honest!\n"
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
    print("DEBUG: Evaluator Agent auditing...")
    recommendation = state["final_recommendation"]
    retries = state.get("retry_count", 0)
    
    if retries >= 3:
        return {"revision_needed": False, "critique": "Max retries reached."}

    try:
        clean_text = recommendation.replace("```json", "").replace("```", "").strip()
        json.loads(clean_text)
        return {"revision_needed": False, "critique": "None"}
    except:
        return {"revision_needed": True, "critique": "Invalid JSON", "retry_count": retries + 1}

# --- GRAPH CONSTRUCTION ---
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
workflow.add_edge("intent_agent", "retrieval_agent")
workflow.add_edge("retrieval_agent", "reasoner_agent")
workflow.add_edge("reasoner_agent", "image_fetcher_agent")
workflow.add_edge("image_fetcher_agent", "evaluator_agent")
workflow.add_conditional_edges("evaluator_agent", decide_next_step, {"retrieval_agent": "retrieval_agent", END: END})

app = workflow.compile()