import os
import requests
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from typing import TypedDict, List, Optional

class CustomMessagesState(TypedDict):
    messages: List[BaseMessage]
    decision: Optional[str]

# Set up API keys from environment variables
GOOGLE_CUSTOM_SEARCH_API_KEY = os.environ.get("AIzaSyCr3eLaGhFEz4-cjQsfUbylM3kleQoX2uo")
GOOGLE_CUSTOM_SEARCH_ENGINE_ID = os.environ.get("e57c12194a2c34e10")
GOOGLE_PLACES_API_KEY = os.environ.get("AIzaSyCpAS9Rarl_7nUoU1nG47iPOGzFCcB4Lks")
GOOGLE_MAPS_API_KEY = os.environ.get("AIzaSyCpAS9Rarl_7nUoU1nG47iPOGzFCcB4Lks")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# Constants for the services
LOCATION = "us-central1"
LLM = "gemini-2.0-flash-001"

# 1. Define Tools using external APIs

@tool
def search_web(query: str) -> str:
    """
    Performs a web search using the Google Custom Search API.
    API Documentation: https://developers.google.com/custom-search/v1/overview
    Set the environment variables GOOGLE_CUSTOM_SEARCH_API_KEY and GOOGLE_CUSTOM_SEARCH_ENGINE_ID.
    """
    if not GOOGLE_CUSTOM_SEARCH_API_KEY or not GOOGLE_CUSTOM_SEARCH_ENGINE_ID:
        return "Google Custom Search API key or Search Engine ID is not set."
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_CUSTOM_SEARCH_API_KEY,
        "cx": GOOGLE_CUSTOM_SEARCH_ENGINE_ID,
        "q": query
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json()
        # Return top 3 search results (titles and snippets)
        items = results.get("items", [])[:3]
        output = "\n".join([f"{item['title']}: {item['snippet']}" for item in items])
        return output if output else "No results found."
    else:
        return f"Error during web search: {response.status_code}"

@tool
def get_places(query: str) -> str:
    """
    Retrieves place information using the Google Places API.
    API Documentation: https://developers.google.com/places/web-service/overview
    Set the environment variable GOOGLE_PLACES_API_KEY.
    """
    if not GOOGLE_PLACES_API_KEY:
        return "Google Places API key is not set."
    
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "key": GOOGLE_PLACES_API_KEY,
        "query": query
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json()
        places = results.get("results", [])[:3]
        output = "\n".join([f"{place.get('name')}: {place.get('formatted_address')}" for place in places])
        return output if output else "No places found."
    else:
        return f"Error retrieving places: {response.status_code}"

@tool
def get_directions(query: str) -> str:
    """
    Provides directions using the Google Maps Directions API.
    API Documentation: https://developers.google.com/maps/documentation/directions/overview
    Set the environment variable GOOGLE_MAPS_API_KEY.
    Note: Query should be in the format "origin: <origin> destination: <destination>"
    """
    if not GOOGLE_MAPS_API_KEY:
        return "Google Maps API key is not set."
    
    # Simple parsing: extract origin and destination from the query
    try:
        parts = query.split("destination:")
        origin_part = parts[0].replace("origin:", "").strip()
        destination_part = parts[1].strip()
    except Exception:
        return "Invalid query format. Use 'origin: <origin> destination: <destination>'."
    
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "key": GOOGLE_MAPS_API_KEY,
        "origin": origin_part,
        "destination": destination_part,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        routes = data.get("routes", [])
        if routes:
            leg = routes[0]["legs"][0]
            steps = leg["steps"]
            # Clean HTML from directions and join steps with arrows
            directions = " -> ".join(
                [step["html_instructions"].replace("<b>", "").replace("</b>", "") for step in steps]
            )
            return directions
        else:
            return "No directions found."
    else:
        return f"Error retrieving directions: {response.status_code}"

@tool
def get_weather(query: str) -> str:
    """
    Retrieves current weather data using the OpenWeatherMap API.
    API Documentation: https://openweathermap.org/api
    Set the environment variable OPENWEATHER_API_KEY.
    Note: Query should contain the city name.
    """
    if not OPENWEATHER_API_KEY:
        return "OpenWeatherMap API key is not set."
    
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": query,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"The weather in {query} is {weather_desc} with a temperature of {temp}°C."
    else:
        return f"Error retrieving weather data: {response.status_code}"

# List all tools to bind with the LLM
tools = [search_web, get_places, get_directions, get_weather]

# 2. Set up the language model and bind tools
llm = ChatVertexAI(
    model=LLM, location=LOCATION, temperature=0, max_tokens=1024, streaming=True
).bind_tools(tools)

# 3. Define the Router and Sub-Agent Nodes

# Update call_router to return both the messages and the decision
def call_router(state: CustomMessagesState) -> dict:
    """
    Uses LLM reasoning to determine which sub-agent to call next based on the message content.
    """
    last_message = state["messages"][-1]
    
    # Only route if the last message is from the human
    if not isinstance(last_message, HumanMessage):
        print("Last message is from AI. Exiting")
        return {"messages": state["messages"], "decision": "END"}
    
    # Extract message text
    message_content = last_message.content
    if isinstance(message_content, list):
        # Handle structured content
        message_text = " ".join([item.get('text', '') for item in message_content 
                               if isinstance(item, dict) and item.get('type') == 'text'])
    else:
        # Handle string content
        message_text = str(message_content)
    
    # Use the LLM to determine routing
    system_prompt = """You are a routing assistant that determines which specialized agent should handle a user query.
    Analyze the user's message and select ONE of the following agents:
    - "weather_node": For questions about weather conditions or forecasts
    - "places_node": For questions about restaurants, attractions, or points of interest
    - "maps_node": For questions about directions, navigation, or travel routes
    - "web_search_node": For general information queries that require web search
    - "END": If the query doesn't clearly match any of the above categories
    
    Return ONLY the agent name without any explanation or additional text."""
    
    router_llm = ChatVertexAI(
        model="gemini-2.0-flash-001", 
        location=LOCATION, 
        temperature=0,
        max_tokens=5
    )
    
    # Get routing decision from LLM
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Route this query: {message_text}")
    ]
    
    response = router_llm.invoke(messages)
    routing_decision = response.content.strip()
    
    # Validate the routing decision
    valid_routes = ["weather_node", "places_node", "maps_node", "web_search_node", "END"]
    if routing_decision in valid_routes:
        return {"messages": state["messages"], "decision": routing_decision}
    else:
        # Default to web search if the LLM returns an invalid route
        return {"messages": state["messages"], "decision": "web_search_node"}


# Update route_next to correctly handle the state
def route_next(state: CustomMessagesState) -> str:
    """
    Routes to the next node based on the decision.
    """
    return state["decision"]

# Update the specialized agent functions to maintain the decision field
def call_web_search(state: CustomMessagesState, config: RunnableConfig) -> dict:
    """
    Sub-agent for handling web search queries.
    """
    system_message = """You are a web search agent specializing in travel-related information.
    
    IMPORTANT: When a user asks for information that requires searching the web:
    1. Use the search_web tool to find relevant information
    2. Analyze the search results
    3. Provide a helpful summary based on the search results
    
    Always use the search_web tool to retrieve up-to-date information for travel questions."""
    
    # Convert messages to the format expected by the LLM
    formatted_messages = [SystemMessage(content=system_message)] + state["messages"]
    
    # Invoke the LLM
    response = llm.invoke(formatted_messages, config)
    
    # Return the updated messages state with the original decision
    return {"messages": state["messages"] + [response], "decision": state["decision"]}


def call_places(state: CustomMessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    """
    Sub-agent for providing places recommendations.
    Instructs the LLM to use the get_places tool when appropriate.
    """
    system_message = """You are a places recommendation agent specializing in travel destinations.
    
    IMPORTANT: When a user asks about restaurants, attractions, or places:
    1. Use the get_places tool to find relevant place information
    2. Analyze the returned place data
    3. Provide helpful recommendations based on the places data
    
    Always use the get_places tool to retrieve up-to-date information about locations."""
    
    # Convert messages to the format expected by the LLM
    formatted_messages = [SystemMessage(content=system_message)] + state["messages"]
    
    # Invoke the LLM
    response = llm.invoke(formatted_messages, config)
    
    # Return the updated messages state
    return {"messages": state["messages"] + [response], "decision": state["decision"]}

def call_maps(state: CustomMessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    """
    Sub-agent for delivering directions and maps information.
    Instructs the LLM to use the get_directions tool when appropriate.
    """
    system_message = """You are a maps and directions specialist.
    
    IMPORTANT: When a user asks for directions or how to get from one place to another:
    1. Use the get_directions tool to retrieve routing information
    2. Format the query as 'origin: <origin> destination: <destination>'
    3. Provide clear step-by-step directions based on the results
    
    Always use the get_directions tool to provide accurate navigation guidance."""
    
    # Convert messages to the format expected by the LLM
    formatted_messages = [SystemMessage(content=system_message)] + state["messages"]
    
    # Invoke the LLM
    response = llm.invoke(formatted_messages, config)
    
    # Return the updated messages state
    return {"messages": state["messages"] + [response], "decision": state["decision"]}

def call_weather(state: CustomMessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    """
    Sub-agent for providing weather updates.
    Instructs the LLM to use the get_weather tool when appropriate.
    """
    system_message = """You are a weather information specialist.
    
    IMPORTANT: When a user asks about weather:
    1. Use the get_weather tool to retrieve current weather data
    2. Extract the location from the user's query
    3. Provide a clear weather summary based on the results
    
    Always use the get_weather tool to provide accurate and up-to-date weather information."""
    
    # Convert messages to the format expected by the LLM
    formatted_messages = [SystemMessage(content=system_message)] + state["messages"]
    
    # Invoke the LLM
    response = llm.invoke(formatted_messages, config)
    
    # Return the updated messages state
    return {"messages": state["messages"] + [response], "decision": state["decision"]}

# 4. Create the workflow graph with the new nodes
workflow = StateGraph(CustomMessagesState)

# Add our nodes: the router and the specialized sub-agents.
workflow.add_node("router", call_router)
workflow.add_node("web_search_node", call_web_search)
workflow.add_node("places_node", call_places)
workflow.add_node("maps_node", call_maps)
workflow.add_node("weather_node", call_weather)

# Optionally add a tools node if you wish to invoke tool calls directly
#workflow.add_node("tools", ToolNode(tools))

# 5. Define Graph Edges
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route_next)
workflow.add_edge("web_search_node", "router")
workflow.add_edge("places_node", "router") 
workflow.add_edge("maps_node", "router")
workflow.add_edge("weather_node", "router")

# 6. Compile the Workflow
agent = workflow.compile()

# 7. Display the compiled graph image (requires workflow graph support for draw_mermaid_png)
# display(Image(workflow.get_graph().draw_mermaid_png()))