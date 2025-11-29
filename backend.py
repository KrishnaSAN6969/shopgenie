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

# --- AGENT 1: INTENT AGENT (With Budget Guard) ---
def intent_agent(state: AgentState):
    print(f"DEBUG: Processing '{state['query']}'")
    
    # UPDATED PROMPT: Adds the "ASK_BUDGET" logic
    prompt = ChatPromptTemplate.from_template(
        "You are ShopGenie-E. Analyze the user input.\n"
        "USER INPUT: {query}\n"
        "HISTORY: {chat_history}\n\n"
        "TASK: Classify and Respond.\n"
        "1. IF CASUAL (Greeting, Thanks, Small Talk): Reply as a helpful AI.\n"
        "   - Format: 'CHAT: [Response]'\n"
        "2. IF SHOPPING request but NO BUDGET/PRICE is mentioned in input or history:\n"
        "   - Format: 'ASK_BUDGET: [Polite question asking for price range]'\n"
        "3. IF SHOPPING with Budget identified: Refine the query.\n"
        "   - Format: 'SEARCH: [Refined Query]'\n"
    )
    chain = prompt | llm
    response = chain.invoke({
        "chat_history": state.get("chat_history", ""), 
        "query": state["query"]
    }).content.strip()
    
    # Routing Logic
    if response.startswith("CHAT:"):
        return {
            "intent": "casual_chat", 
            "final_recommendation": response.replace("CHAT:", "").strip(),
            "refined_query": ""
        }
    elif response.startswith("ASK_BUDGET:"):
        return {
            "intent": "ask_budget", 
            "final_recommendation": response.replace("ASK_BUDGET:", "").strip(),
            "refined_query": ""
        }
    else:
        refined = response.replace("SEARCH:", "").strip()
        return {
            "intent": "buy_request", 
            "refined_query": refined,
            "retry_count": state.get("retry_count", 0)
        }

# --- AGENT 2: RETRIEVAL AGENT ---
def retrieval_agent(state: AgentState):
    query = state.get("refined_query", "")
    critique = state.get("critique", "")
    
    if not query:
        return {"search_results": []}

    if state.get("revision_needed"):
        print(f"DEBUG: Retrying search: {critique}")
        search_query = f"{query} buy page amazon bestbuy walmart {critique}"
    else:
        print("DEBUG: Initial search...")
        search_query = f"{query} price buy online site:amazon.com OR site:bestbuy.com OR site:walmart.com"
    
    try:
        results = tavily.search(query=search_query, search_depth="advanced", max_results=7)
        return {"search_results": results['results']}
    except Exception as e:
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
        "      'full_details': ['Point 1', 'Point 2'],\n"
        "      'specs': {{ 'Performance': '...', 'Build_Quality': '...', 'Key_Feature': '...' }}, \n"
        "      'ai_insights': {{ 'score': 8, 'best_for': '...', 'dealbreaker': '...' }} \n"
        "   }} ] }}\n"
        "3. CATEGORY RULE: Use 'Powerhouse', 'Balanced', 'Budget'.\n"
        "4. NOOB TRANSLATION RULE: \n"
        "   - Do NOT just list specs. Translate them.\n"
        "   - GOOD: '4GB RAM: Good for basic browsing but keep tabs closed.'\n"
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
    # If Chat OR Ask Budget, stop here and reply to user
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

# ROUTING LOGIC
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