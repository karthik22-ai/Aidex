# main.py
# Main application file for the Aidex FastAPI backend.
# This file initializes the FastAPI app, sets up CORS, and defines the API endpoints.

import os
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
import json
from aiohttp import ClientSession
from agents import medical_guard_agent, symptom_analysis_agent, language_translation_agent, visual_analysis_agent
from dotenv import load_dotenv

# --- Load Environment Variables ---
# This will load variables from a .env file in the same directory.
load_dotenv()

# Load the Gemini API Key from environment variables
# IMPORTANT: You need to set the GEMINI_API_KEY in your .env file.
# Example .env file content:
# GEMINI_API_KEY="your_actual_api_key_here"
API_KEY = os.getenv("GEMINI_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Aidex Medical Assistant API",
    description="Backend for the Aidex application with agentic AI capabilities.",
    version="1.0.0"
)

# --- CORS (Cross-Origin Resource Sharing) Configuration ---
# This allows the frontend hosted on a different domain (or locally)
# to communicate with this backend.

# FIX: Changed `origins` to ["*"] to allow all origins for local development.
# This resolves the CORS preflight OPTIONS request error.
# For production, you should replace ["*"] with the specific URL of your deployed frontend
# for security reasons, e.g., ["https://your-aidex-app.com"].
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- In-memory Session Management ---
# For a production system, consider using a more robust solution like Redis.
chat_sessions = {}

# --- Helper Function for Gemini API Calls ---
async def call_gemini_api(prompt: str, model: str = "gemini-2.0-flash", is_json_output: bool = False, image_data: str = None):
    """
    Generic helper function to call the Google Gemini API.
    """
    if not API_KEY:
        # This error will now be more informative if the .env file is missing or the key isn't set.
        return {"error": "GEMINI_API_KEY not found. Make sure it is set in your .env file."}

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"

    # Construct the payload
    parts = [{"text": prompt}]
    if image_data:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_data}})

    payload = {
        "contents": [{"parts": parts}]
    }

    if is_json_output:
        payload["generationConfig"] = {"responseMimeType": "application/json"}

    try:
        async with ClientSession() as session:
            async with session.post(api_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    # Safely extract the text from the response
                    if result.get("candidates"):
                        content_part = result["candidates"][0]["content"]["parts"][0]
                        return content_part.get("text")
                    else:
                        # Handle cases where the response structure is unexpected
                        print(f"Unexpected API response: {result}")
                        return {"error": "Unexpected response from Gemini API."}
                else:
                    error_text = await response.text()
                    print(f"API Error: {response.status}, {error_text}")
                    return {"error": f"Failed to get response from Gemini API. Status: {response.status}"}
    except Exception as e:
        print(f"An exception occurred: {e}")
        return {"error": f"An exception occurred while calling Gemini API: {e}"}


# --- API Endpoints ---

@app.get("/", summary="Root endpoint", description="A simple hello world endpoint to check if the server is running.")
async def read_root():
    """
    Root endpoint to check server status.
    """
    return {"message": "Welcome to the Aidex AI Backend. We are ready to assist."}

@app.post("/chat", summary="Main chat endpoint", description="Handles text-based user interactions by orchestrating AI agents.")
async def chat(request: Request):
    """
    This endpoint processes the user's text message.
    1. It uses the Medical Guard Agent to check if the query is medical.
    2. If medical, it passes the query to the Symptom Analysis Agent.
    3. It uses the Language Translation Agent to respond in the user's language.
    """
    body = await request.json()
    user_message = body.get("message")
    session_id = body.get("session_id", "default_session")
    language = body.get("language", "en") # Default to English

    if not user_message:
        return {"error": "Message cannot be empty."}

    # --- Agent Orchestration ---

    # Step 1: Call the Medical Guard Agent
    guard_prompt = medical_guard_agent.get_prompt(user_message)
    guard_response_str = await call_gemini_api(guard_prompt, is_json_output=True)
    try:
        guard_response = json.loads(guard_response_str)
    except (json.JSONDecodeError, TypeError):
        print(f"Error decoding guard agent response: {guard_response_str}")
        return {"reply": "I'm sorry, I'm having trouble understanding the nature of your request right now."}


    if not guard_response.get("is_medical"):
        # If the query is not medical, return a polite refusal
        refusal_message = "I am an AI medical assistant named Aidex. I can only answer questions related to medical symptoms, health conditions, and wellness. How can I help you with a medical topic?"
        # Translate the refusal message if needed
        if language != 'en':
             refusal_message = await call_gemini_api(language_translation_agent.get_prompt(refusal_message, language))
        return {"reply": refusal_message}


    # Step 2: If medical, proceed to the Symptom Analysis Agent
    # Initialize session history if it doesn't exist
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    history = chat_sessions[session_id]
    analysis_prompt = symptom_analysis_agent.get_prompt(user_message, history)
    ai_response = await call_gemini_api(analysis_prompt, model="gemini-2.5-pro") # Use a more powerful model for analysis

    # Update chat history
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": ai_response})
    # Keep history from getting too long
    chat_sessions[session_id] = history[-10:]

    # Step 3: Translate the response if the user's language is not English
    final_response = ai_response
    if language != 'en' and language != 'en-US':
        translation_prompt = language_translation_agent.get_prompt(ai_response, language)
        translated_text = await call_gemini_api(translation_prompt)
        if "error" not in translated_text:
            final_response = translated_text

    return {"reply": final_response}


@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video analysis.
    It receives video frames from the frontend, sends them to the Visual Analysis Agent,
    and streams the AI's response back to the client.
    """
    await websocket.accept()
    try:
        while True:
            # The frontend should send a JSON object with 'image' and 'prompt'
            data = await websocket.receive_json()
            image_data = data.get("image") # This should be a base64 encoded string
            prompt_text = data.get("prompt", "Analyze the user's visual state and emotion.")

            if not image_data:
                continue

            # Get the analysis from the visual agent
            analysis_prompt = visual_analysis_agent.get_prompt(prompt_text)
            visual_analysis_response = await call_gemini_api(
                prompt=analysis_prompt,
                model="gemini-2.5-pro", # Vision model
                image_data=image_data
            )

            # Stream the response back to the client
            await websocket.send_json({"analysis": visual_analysis_response})

    except WebSocketDisconnect:
        print("Client disconnected from video websocket.")
    except Exception as e:
        print(f"An error occurred in the video websocket: {e}")
        await websocket.close(code=1011, reason=f"An internal error occurred: {e}")


# --- Uvicorn Server Runner ---
# This allows running the server directly with `python main.py`
if __name__ == "__main__":
    print("Starting Aidex backend server...")
    # To run with SSL for webcam access on deployed environments:
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, ssl_keyfile="./key.pem", ssl_certfile="./cert.pem")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)







