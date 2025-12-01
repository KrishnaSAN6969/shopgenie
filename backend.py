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
    use_case: str
    intent: str
    search_results: List[dict]
    final_recommendation: str
    critique: str
    revision_needed: bool
    retry_count: int

# --- AGENT 1: INTENT AGENT (Product Guard Fix) ---
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
            "final_recommendation": "Hello! I am ShopGenie-E. Tell me what you do (e.g. 'I am a student') or what you need, and I'll find the perfect gear.",
            "refined_query": "",
            "use_case": "general"
        }

    # 2. PRODUCT DETECTION (The Critical Fix)
    # We ask the AI: Is there a physical product mentioned here?
    product_check_prompt = ChatPromptTemplate.from_template(
        "Analyze text: '{query}'.\n"
        "Does this text explicitly mention an electronic product category to buy (e.g. 'laptop', 'phone', 'monitor', 'keyboard', 'headphones')?\n"
        "Answer YES or NO."
    )
    has_product = (product_check_prompt | llm).invoke({"query": state["query"]}).content.strip().upper()

    # 3. USE CASE EXTRACTION
    extraction_prompt = ChatPromptTemplate.from_template(
        "Extract USE CASE from: '{query}'. Examples: 'Student', 'Gaming', 'Coding'. Default to 'General'."
    )
    use_case = (extraction_prompt | llm).invoke({"query": state["query"]}).content.strip()

    # --- LOGIC BRANCHING ---
    
    # CASE A: User gave context but NO product (e.g. "I am a student")
    if "NO" in has_product:
        return {
            "intent": "casual_chat",
            "final_recommendation": f"Got it! As a **{use_case}**, you have specific needs. What exact product are you looking for? (e.g. 'A laptop', 'Headphones')",
            "refined_query": "",
            "use_case": use_case
        }

    # CASE B: User mentioned a product. Now check BUDGET.
    price_keywords = ['$', 'usd', 'price', 'budget', 'cheap', 'expensive', 'cost', 'under', 'over', 'less', 'max']
    has_price = any(w in user_input or w in history for w in price_keywords) or bool(re.search(r'\d+', user_input))
    
    if not has_price:
        return {
            "intent": "ask_budget",
            "final_recommendation": f"To find the best **{use_case}** device for you, I need a price range. What is your budget? (e.g. '$500', 'cheap')",
            "refined_query": "",
            "use_case": use_case
        }

    # CASE C: Product + Budget = SEARCH
    refine_prompt = ChatPromptTemplate.from_template(
        "Refine search query for: '{query}'. Context: {use_case}. Include 'best' and 'price'."
    )
    refined_query = (refine_prompt | llm).invoke({"query": state["query"], "use_case": use_case}).content.strip()

    return {
        "intent": "buy_request", 
        "refined_query": refined_query,
        "use_case": use_case,
        "retry_count": state.get("retry_count", 0)
    }

# --- AGENT 2: RETRIEVAL AGENT ---
def retrieval_agent(state: AgentState):
    query = state.get("refined_query", "")
    use_case = state.get("use_case", "General")
    
    if not query: return {"search_results": []}

    # Add use-case to search to filter results intelligently
    search_query = f"{query} best for {use_case} price reviews site:amazon.com OR site:bestbuy.com OR site:walmart.com"
    
    try:
        results = tavily.search(query=search_query, search_depth="advanced", max_results=8)
        return {"search_results": results['results']}
    except:
        return {"search_results": []}

# --- AGENT 3: REASONER AGENT (The Detailed Analyst) ---
def reasoner_agent(state: AgentState):
    print("DEBUG: Reasoner Agent thinking...")
    data = state["search_results"]
    use_case = state.get("use_case", "General Use")
    
    prompt = ChatPromptTemplate.from_template(
        "You are ShopGenie-E. Return a JSON OBJECT only.\n"
        "GOAL: Provide top 3 options fitting the user's budget AND USE CASE: '{use_case}'.\n"
        "\n"
        "RULES:\n"
        "1. Output valid JSON only.\n"
        "2. Structure: {{ 'options': [ {{ \n"
        "      'category': 'Powerhouse', \n"
        "      'name': '...', \n"
        "      'price': '...', \n"
        "      'summary': '...', \n"
        "      'link': '...', \n"
        "      'fit_summary': '...', \n"
        "      'full_details': ['Point 1', 'Point 2', 'Point 3', 'Point 4'],\n"
        "      'tech_specs': {{ 'Spec1': '...', 'Spec2': '...' }}, \n"
        "      'specs': {{ 'Performance': '...', 'Build_Quality': '...', 'Key_Feature': '...' }}, \n"
        "      'ai_insights': {{ 'score': 8, 'best_for': '...', 'dealbreaker': '...' }} \n"
        "   }} ] }}\n"
        "3. CATEGORY RULE: Use 'Powerhouse', 'Balanced', 'Budget'.\n"
        "4. 'fit_summary' RULE (CRITICAL):\n"
        "   - Write a PERSUASIVE, MEDIUM-LENGTH PARAGRAPH (approx 50-70 words).\n"
        "   - Do NOT write a single sentence.\n"
        "   - Connect specific specs to the user's '{use_case}'.\n"
        "   - Example: 'This laptop is an excellent choice for your engineering studies because the dedicated GPU handles CAD software smoothly. The 16GB RAM ensures you can multitask without lag, while the durable build quality is perfect for carrying around campus daily.'\n"
        "5. TRANSLATE BENEFITS: 'full_details' must explain specs in plain English.\n"
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