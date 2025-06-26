from langchain_xai import ChatXAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

class AuthorizationData(BaseModel):
    patient_id: str = None
    procedure_code: str = None
    insurance: str = None

async def process_message(text: str) -> tuple[str, dict]:
    llm = ChatXAI(api_key=os.getenv("XAI_API_KEY"))
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        You are Alfons, a prior authorization bot. Extract patient ID, procedure code, and insurance from the input.
        If the query is complex or requests escalation, respond with an escalation message.
        Respond empathetically and naturally, as if speaking to a patient or provider.
        Input: {text}
        Output format:
        Response: [Your response]
        Data: {{ "patient_id": "...", "procedure_code": "...", "insurance": "..." }}
        """
    )
    response = await llm.agenerate([prompt.format(text=text)])
    result = response.generations[0][0].text
    response_text, data_str = result.split("\nData: ")
    data = eval(data_str)
    return response_text, data