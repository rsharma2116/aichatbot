# app/services/gemini.py (new implementation)
import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from google.api_core import exceptions as google_exceptions

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key='AIzaSyB_Vt2KsTL1SWmgZNCnG0a2NDFao0PN_xw')

# Set up the model
generation_config = {
    "temperature": 0.7,
    "max_output_tokens": 100,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]

model = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash-latest",
    generation_config=generation_config,
    safety_settings=safety_settings
)

def query_gemini(prompt: str) -> str:
    try:
        logger.info(f"Sending to Gemini API: {prompt}")
        
        response = model.generate_content(prompt)
        
        logger.info(f"Gemini API response: {response}")
        
        if not response.text:
            logger.warning("Empty response from Gemini API")
            return "I didn't get a response. Please try again."
            
        return response.text

    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Google API error: {str(e)}")
        return f"Gemini service error: {str(e)}"
    except ValueError as e:
        logger.error(f"Content generation error: {str(e)}")
        return "Sorry, I couldn't generate a proper response."
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"AI service error: {str(e)}"