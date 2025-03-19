import os
import requests
from typing import Annotated, List, TypedDict, Dict, Any, Optional
import operator
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, START, StateGraph
from langgraph.constants import Send

LOCATION = "us-central1"
LLM = "gemini-2.0-flash-001"

# API keys
GOOGLE_PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "AIzaSyCpAS9Rarl_7nUoU1nG47iPOGzFCcB4Lks")

# Define valid place types for API calls
VALID_PLACE_TYPES = [
    'tourist_attraction', 'bank', 'bar', 'car_rental', 'electric_vehicle_charging_station', 
    'gas_station', 'parking', 'art_gallery', 'cultural_landmark', 'historical_place', 
    'monument', 'museum', 'performing_arts_theater', 'amusement_park', 'aquarium', 
    'botanical_garden', 'dog_park', 'historical_landmark', 'night_club', 'zoo', 
    'atm', 'coffee_shop', 'bed_and_breakfast', 'hotel', 'church', 'clothing_store', 
    'department_store', 'gift_shop', 'market', 'restaurant', 'cafe'
]

# 1. Define tools for travel information
def search_weather(location: str) -> str:
    """Search for weather information in a specific location"""
    # Simplified mock response - will replace with actual API call later
    if "san francisco" in location.lower() or "sf" in location.lower():
        return "It's 60 degrees and foggy in San Francisco."
    elif "new york" in location.lower() or "nyc" in location.lower():
        return "It's 75 degrees and partly cloudy in New York City."
    else:
        return f"The weather in {location} is sunny and 72 degrees."

def search_places(location: str, place_type: str) -> Dict:
    """Search for places in a specific location using Google Places API"""
    try:
        # Format the search query for attractions in the specified location
        query = f"{place_type} in {location}"
        
        # Call the Google Places API Text Search endpoint
        url = "https://places.googleapis.com/v1/places:searchText"
        params = {
            "textQuery": query,
            "key": GOOGLE_PLACES_API_KEY,
            "includedType": place_type,
            "fields": "places.displayName,places.formattedAddress,places.priceLevel,places.priceRange,places.rating,places.userRatingCount,places.currentOpeningHours,places.location,places.accessibilityOptions"
        }

        response = requests.post(url, params=params)
        data = response.json()
        
        # Check if the API call was successful
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"API error: {str(data)}",
                "places": []
            }
        
        # Extract and format place information
        places_data = []
        results = data.get("places", [])
        
        if not results:
            return {
                "success": True,
                "places": [],
                "message": f"No {place_type} found in {location}."
            }
        
        # Process up to 5 places
        for place in results[:5]:
            place_data = {}
            
            # Get name (handle possible none or missing text attribute)
            display_name = place.get("displayName", {})
            if isinstance(display_name, dict) and "text" in display_name:
                place_data["name"] = display_name["text"]
            else:
                place_data["name"] = "Unknown place"
                
            place_data["rating"] = place.get("rating", "No rating")
            place_data["reviews"] = place.get("userRatingCount", 0)
            place_data["address"] = place.get("formattedAddress", "No address available")
            
            # Handle price level if available
            if "priceLevel" in place:
                place_data["price_level"] = place["priceLevel"]
            
            places_data.append(place_data)
        print(str(places_data))
        return {
            "success": True,
            "places": places_data,
            "place_type": place_type,
            "location": location,
            "count": len(places_data)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "places": [],
            "place_type": place_type,
            "location": location
        }

# 2. Set up the language model
llm = ChatVertexAI(
    model=LLM,
    location=LOCATION,
    temperature=0.2,
    max_tokens=1024,
    streaming=True
)

# Define schema for worker tasks
class WorkerTask(BaseModel):
    worker_id: str = Field(
        description="Unique identifier for the worker to execute this task",
    )
    action: str = Field(
        description="Specific action for the worker to perform",
    )
    location: str = Field(
        description="The location to get information about",
    )
    parameters: Dict[str, Any] = Field(
        default={},
        description="Additional parameters required for the task",
    )
    priority: int = Field(
        default=1,
        description="Priority of task (1=highest, 3=lowest)",
    )

class TravelPlan(BaseModel):
    destination: str = Field(
        description="The main destination the user is interested in",
    )
    visit_purpose: Optional[str] = Field(
        default="tourism",
        description="The purpose of the visit (tourism, business, food tour, etc.)",
    )
    tasks: List[WorkerTask] = Field(
        description="List of tasks for workers to execute",
    )

# Augment the LLM with schema for structured output
planner = llm.with_structured_output(TravelPlan)

# Define state schema for our traveler agent
class TravelState(TypedDict):
    messages: List[BaseMessage]  # Conversation messages
    destination: str  # Main destination 
    visit_purpose: str  # Purpose of visit
    tasks: List[WorkerTask]  # Tasks identified by orchestrator
    completed_tasks: Annotated[List[Dict], operator.add]  # Results from workers
    final_response: str  # Final synthesized response

# Define worker state schema
class WorkerState(TypedDict):
    worker_id: str  # Worker identifier
    action: str  # Action to perform
    location: str  # Location to search
    parameters: Dict[str, Any]  # Additional parameters
    completed_tasks: Annotated[List[Dict], operator.add]  # Worker results
    priority: int  # Task priority

# 3. Define workflow components - Orchestrator-Worker pattern

def orchestrator(state: TravelState) -> dict:
    """Analyze user request and create a comprehensive travel plan with worker tasks"""
    # Get the last user message
    messages = state.get("messages", [])
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    
    if not user_messages:
        return {"final_response": "No query provided"}
    
    last_message = user_messages[-1]
    
    # Use planner to create a structured travel plan with worker assignments
    travel_plan = planner.invoke(
        [
            SystemMessage(content=f"""You are an AI travel planning expert who analyzes user queries and creates detailed travel plans.

            Your job is to:
            1. Identify the location(s) the user is interested in
            2. Understand the purpose of their visit
            3. Create a list of specific worker tasks to gather relevant information
            
            AVAILABLE WORKERS AND THEIR CAPABILITIES:
            
            1. "weather_worker":
               - action: "get_weather"
               - Provides weather information for a location
            
            2. "places_worker":
               - action: "search_places"
               - Searches for places by type in a location
               - Returns minimum of 5 places
               - Required parameters:
                 * place_type: The type of places to search for
                 * Valid types include: {VALID_PLACE_TYPES}
            
            For each task, you must specify:
            1. worker_id: Which worker should handle this task
            2. action: What action the worker should perform
            3. location: The relevant location
            4. parameters: Any additional parameters needed (like place_type=restaurant)
            5. priority: How important this task is (1=highest, 3=lowest)
            
            Be comprehensive and consider all aspects of the user's query.
            
            Below are the sections that should be included in the travel guide:
            1. 📍 WELCOME TO [DESTINATION] - An enthusiastic, personalized introduction
            2. 🌤️ WEATHER & WHEN TO VISIT - Current conditions and practical advice
            3. 🏛️ TOP ATTRACTIONS - Must-see places with brief descriptions
            4. 🍽️ DINING RECOMMENDATIONS - Where to eat based on available information
            5. 💡 INSIDER TIPS - Helpful advice for the visitor
            6. 🚶 SUGGESTED ITINERARY - A brief outline for their visit

            When creating tasks, make sure to include all the sections in the travel guide.
            If the user asks about a specific place type, create a task for places_worker with that type.
            Always include weather information unless clearly irrelevant.
            Try to include as many places as possible.
            If the query is general, include tasks for important tourist attractions.
            """),
            HumanMessage(content=f"User query: {last_message.content}")
        ]
    )
    
    return {
        "destination": travel_plan.destination,
        "visit_purpose": travel_plan.visit_purpose,
        "tasks": travel_plan.tasks
    }

def weather_worker(state: WorkerState) -> dict:
    """Worker that retrieves weather information"""
    location = state.get("location", "")
    priority = state.get("priority", 1)
    
    weather_info = search_weather(location)
    
    result = {
        "worker_id": "weather_worker",
        "action": "get_weather",
        "location": location,
        "priority": priority,
        "data": weather_info,
        "data_type": "weather"
    }
    
    return {"completed_tasks": [result]}

def places_worker(state: WorkerState) -> dict:
    """Worker that searches for places by type"""
    location = state.get("location", "")
    parameters = state.get("parameters", {})
    priority = state.get("priority", 1)
    
    place_type = parameters.get("place_type", "tourist_attraction")
    # Validate place type
    if place_type not in VALID_PLACE_TYPES:
        place_type = "tourist_attraction"
    
    places_data = search_places(location, place_type)
    
    result = {
        "worker_id": "places_worker",
        "action": "search_places",
        "location": location,
        "place_type": place_type,
        "priority": priority,
        "data": places_data,
        "data_type": "places"
    }
    
    return {"completed_tasks": [result]}

def synthesizer(state: TravelState) -> dict:
    """Create an engaging, well-formatted travel guide from worker results"""
    destination = state.get("destination", "your destination")
    visit_purpose = state.get("visit_purpose", "your trip")
    results = state.get("completed_tasks", [])
    
    if not results:
        return {"final_response": f"I'm sorry, I couldn't find information about {destination}."}
    
    # Prepare all results data for the LLM
    synthesizer_prompt = f"""
    Create an engaging, personalized travel guide for {destination}. The user is planning a {visit_purpose}.
    
    Use the following information to create your guide:
    
    DESTINATION: {destination}
    VISIT PURPOSE: {visit_purpose}
    
    GATHERED INFORMATION:
    """
    
    # Add all worker results to the prompt
    for result in results:
        worker_id = result.get("worker_id", "unknown")
        action = result.get("action", "unknown")
        data_type = result.get("data_type", "unknown")
        data = result.get("data", {})
        
        synthesizer_prompt += f"\n--- {data_type.upper()} INFORMATION ---\n"
        synthesizer_prompt += f"Data: {data}\n"
    
    # Generate the final response using the LLM
    response = llm.invoke([
        SystemMessage(content="""You are an expert travel assistant creating personalized travel guides.
        
        Create a beautifully formatted travel guide with:
        
        1. 📍 WELCOME TO [DESTINATION] - An enthusiastic, personalized introduction
        2. 🌤️ WEATHER & WHEN TO VISIT - Current conditions and practical advice
        3. 🏛️ TOP ATTRACTIONS - Must-see places with brief descriptions
        4. 🍽️ DINING RECOMMENDATIONS - Where to eat based on available information
        5. 💡 INSIDER TIPS - Helpful advice for the visitor
        6. 🚶 SUGGESTED ITINERARY - A brief outline for their visit
        
        Use markdown formatting to make your guide visually appealing:
        - Use headers (## and ###) for sections
        - Use emoji icons for visual appeal
        - Use bullet points for lists
        - Use **bold** and *italic* for emphasis
        
        Make your response conversational and engaging, like advice from a knowledgeable friend.
        Be specific about places mentioned in the data rather than generic.
        If information is limited in some areas, focus on what you do have and make it helpful.
        """),
        HumanMessage(content=synthesizer_prompt)
    ])
    
    return {"final_response": response.content}

def handle_message(state: TravelState) -> dict:
    """Process new user messages and maintain conversation history"""
    return {}  # Don't return any messages to prevent history duplication

def respond_to_user(state: TravelState) -> dict:
    """Format the final response as a message to return to the user"""
    final_response = state.get("final_response", "I'm sorry, I couldn't process your travel query.")
    
    return {"final_response": final_response}  # Just return the final response without adding to messages

def route_to_worker(state: TravelState):
    """Dynamically route tasks to appropriate workers based on worker_id"""
    tasks = state.get("tasks", [])
    
    if not tasks:
        return "synthesizer"
    
    # Create a list of Send operations, one for each task
    worker_assignments = []
    
    for task in tasks:
        worker_id = task.worker_id
        
        # Prepare the worker state
        worker_state = {
            "worker_id": task.worker_id,
            "action": task.action,
            "location": task.location,
            "parameters": task.parameters,
            "priority": task.priority
        }
        
        # Route to the appropriate worker based on worker_id
        if worker_id == "weather_worker":
            worker_assignments.append(Send("weather_worker", worker_state))
        elif worker_id == "places_worker":
            worker_assignments.append(Send("places_worker", worker_state))
    
    return worker_assignments

# 4. Create the workflow graph
workflow = StateGraph(TravelState)

# Add the nodes
workflow.add_node("handle_message", handle_message)
workflow.add_node("orchestrator", orchestrator)
workflow.add_node("weather_worker", weather_worker)
workflow.add_node("places_worker", places_worker)
workflow.add_node("synthesizer", synthesizer)
workflow.add_node("respond_to_user", respond_to_user)

# 5. Define graph edges
workflow.set_entry_point("handle_message")
workflow.add_edge("handle_message", "orchestrator")
workflow.add_conditional_edges("orchestrator", route_to_worker, ["weather_worker", "places_worker", "synthesizer"])
workflow.add_edge("weather_worker", "synthesizer")
workflow.add_edge("places_worker", "synthesizer")
workflow.add_edge("synthesizer", "respond_to_user")
workflow.add_edge("respond_to_user", END)

# 6. Compile the workflow
agent = workflow.compile()

