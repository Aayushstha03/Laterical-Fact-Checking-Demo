import os
import json
import requests
from typing import Dict, Any, List, Annotated
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.7)

def extract_claims(text: str) -> List[str]:
    system_message = SystemMessage(content="""
    You are an expert at extracting claims from text.
    Your task is to identify and list all claims present, true or false,
    in the given text. Each claim should be a single, verifiable statement.
    Consider various forms of claims, including assertions, statistics, and quotes.
    Output a JSON array of strings, and nothing else.
    Do not include any additional text or formatting, like markdown code blocks.
    """)

    human_message = HumanMessage(content=f"Extract factual claims from this text: {text}")
    response = llm.invoke([system_message, human_message])
    print(response.content)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return []
    
def custom_search_api(query: str) -> List[str]:
    # Calls the Laterical custom search API and returns a list of formatted source strings.
    # Each source is formatted as <source><url>...</url><text>...</text></source>

    url = "https://laterical.com/api/call/"
    payload = {
        "path": "search",
        "entity": [query]
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        json_data = response.json()

        # Navigate to the web results list
        results = (
            json_data.get("data", [{}])[0]
            .get("results", {})
            .get("web", [])
        )

        sources = []
        for item in results:
            url_text = item.get("url", "No URL")
            content_text = item.get("text", "No text available")
            source = f"<source>\n<url>{url_text}</url>\n<text>{content_text}</text>\n</source>"
            sources.append(source)

        return sources

    except requests.exceptions.RequestException as e:
        print(f"[API Error] Request failed: {e}")
        return []
    except (ValueError, IndexError, KeyError) as e:
        print(f"[Parsing Error] Unexpected API response format: {e}")
        return []

def verify_claim(claim: str, sources: List[str]) -> Dict[str, Any]:
    if not sources:
        return {
            "claim": claim,
            "assessment": "Insufficient information",
            "confidence_score": 0.5,
            "supporting_sources": [],
            "refuting_sources": []
        }

    combined_sources = "\n\n".join(sources)

    system_message = SystemMessage(content="""
    You are an expert fact-checker.
    Given a claim and a set of sources, determine whether the claim is supported, refuted, or if there is insufficient information in the sources to make a determination.
    For your analysis, consider all the sources collectively.
    Provide your answer as a JSON object with the following structure:
    {
        "claim": "...",
        "assessment": "supported" or "refuted" or "Insufficient information",
        "confidence_score": a number between 0 and 1 (1 means fully confident the claim is true, 0 means fully confident the claim is false),
        "supporting_sources": [list of sources that support the claim],
        "refuting_sources": [list of sources that refute the claim]
    }
    Do not include any additional text or formatting, like markdown code blocks.
    """)

    human_message = HumanMessage(content=f"""
    Claim: "{claim}"
    Sources:
    {combined_sources}
    """)

    response = llm.invoke([system_message, human_message])

    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {
            "claim": claim,
            "assessment": "Insufficient information",
            "confidence_score": 0.5,
            "supporting_sources": [],
            "refuting_sources": []
        }
