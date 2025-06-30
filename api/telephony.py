"""
telephony.py

Module handles all Twilio telephony interactions for the Alfons backend.
It provides a function to initiate outbound phone calls using the Twilio API.
https://www.twilio.com/docs/voice

Environment variables required:
- TWILIO_ACCOUNT_SID: Twilio account SID
- TWILIO_AUTH_TOKEN: Twilio authentication token
- TWILIO_PHONE_NUMBER: The Twilio phone number to make calls from
- BASE_URL: The public URL where Twilio can reach your /voice webhook
"""

import os
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

# Load environment variables from .env file
load_dotenv()

# Print all environment variables for debugging
print("ALL ENV:", dict(os.environ))

# Initialize Twilio client with account SID and auth token
client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

def make_call(phone_number: str) -> str:
    """
    Initiates an outbound call to the specified phone number using Twilio.
    Args:
        phone_number (str): The destination phone number in E.164 format (+1234567890).
    Returns:
        str: The SID of the initiated call.
    """
    print("DEBUG: phone_number received:", phone_number)
    print("Twilio call URL:", f"{os.getenv('BASE_URL')}/voice")
    call = client.calls.create(
        to=phone_number,
        from_=os.getenv("TWILIO_PHONE_NUMBER"),
        url=f"{os.getenv('BASE_URL')}/voice"
    )
    return call.sid