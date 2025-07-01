import json
from typing import Dict, Any, List, Annotated
from pydantic import BaseModel
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM # <-- NEW

import requests
import time

# Set up the LLM (Ollama with Llama 3.2)
llm = OllamaLLM(model="gemma3:4b", temperature=0)  # <-- NEW

def extract_claims(text: str) -> List[str]:
    system_message = SystemMessage(content="""
    You are an expert at extracting claims from text.
    Your task is to identify and list all claims present, no matter how obvious, or trivial they may seem,
    in the given text. Each claim should be a single, verifiable statement.
    Consider various forms of claims, including assertions, statistics, dates, materials, locations and
    quotes. Do not skip any claims, even if they seem obvious. Do not include in the list 'The text contains a claim that needs to be checked for hallucinations' - this is not a claim.
    Present the claims as a JSON array of strings, and do not include any additional text.
    do not include ```json ... ``` at the start or end of the response.
    """)

    human_message = HumanMessage(content=f"Extract factual claims from this text: {text}")
    response = llm.invoke([system_message, human_message])
    
    print("Extracted claims: ", response)
    return response

import requests
from typing import List


def search_claims(claims: List[str]) -> List[List[str]]:
    url = "https://laterical.com/api/call/"
    all_results = []
    for claim in claims:
        payload = {
            "path": "search",
            "entity": [claim]
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            # Navigate safely to web results
            web_results = data.get("data", [{}])[0].get("results", {}).get("web", [])
            # Format as <source> blocks using 'url' and 'text'
            sources = [
                f"<source>\n<url>{entry.get('url', '')}</url>\n<text>{entry.get('text', '')}</text>\n</source>"
                for entry in web_results if entry.get("url") and entry.get("text")
            ]
            all_results.append(sources)
        except requests.exceptions.RequestException as e:
            print(f"Error querying claim: '{claim}'\n{e}")
            all_results.append([])
    return all_results



def verify_claim(claim: str, sources: List[str], llm) -> Dict[str, Any]:
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
    Do not include ```json ... ``` at the start or end of the response, or any markdown rendering cue, just return the JSON object.
    
    """)

    human_message = HumanMessage(content=f"""
        Claim: "{claim}"
    Sources:
        {combined_sources}
        Based on the above sources, assess the claim.
    """)

    # try:
        # response = llm.invoke([system_message, human_message])
    #     result = json.loads(response)
    #     if not isinstance(result, dict):
    #         raise ValueError("Response is not a JSON object")
    # except (json.JSONDecodeError, ValueError):
    #     result = {
    #         "claim": claim,
    #         "assessment": "Insufficient information",
    #         "confidence_score": 0.5,
    #         "supporting_sources": [],
    #         "refuting_sources": []
    #     }
    response = llm.invoke([system_message, human_message])
    print(response)
    return response


def hallucination_check(text: str) -> Dict[str, Any]:
    claims = extract_claims(text)
    claim_verifications = []
    for claim in claims:
        sources = exa_search(claim)
        verification_result = verify_claim(claim, sources)
        claim_verifications.append(verification_result)
    return {
        "claims": claim_verifications
    }

def hallucination_check_tool(text: str) -> Dict[str, Any]:
    return hallucination_check(text)

structured_tool = StructuredTool.from_function(
    func=hallucination_check_tool,
    name="hallucination_check",
    description="Assess the given text for hallucinations using Exa search."
)

class State(BaseModel):
    messages: Annotated[List, add_messages]
    analysis_result: Dict[str, Any] = {}

def call_model(state: State):
    return {"messages": state.messages + [AIMessage(content="Use hallucination_check tool", additional_kwargs={"tool_calls": [{"type": "function", "function": {"name": "hallucination_check"}}]})]}

def run_tool(state: State):
    text_to_check = next((m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)), "")
    tool_output = structured_tool.invoke(text_to_check)
    return {"messages": state.messages + [AIMessage(content=str(tool_output))], "analysis_result": tool_output}

def use_analysis(state: State) -> str:
    return "tools"

# workflow = StateGraph(State)
# workflow.add_node("agent", call_model)
# workflow.add_node("tools", run_tool)
# workflow.add_node("process_result", lambda x: x)
# workflow.set_entry_point("agent")
# workflow.add_conditional_edges("agent", use_analysis, {
#     "tools": "tools"
# })
# workflow.add_edge("tools", "process_result")
# workflow.add_edge("process_result", END)

# graph = workflow.compile()

# initial_state = State(messages=[
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="Check this text for hallucinations: The Eiffel Tower, an iconic iron lattice structure located in Paris, was originally constructed as a giant sundial in 1822.")
# ])

# final_state = graph.invoke(initial_state)

# extract_claims("The Eiffel Tower, an iconic iron lattice structure located in Paris, was originally constructed as a giant sundial in 1822.")
# extract_claims("superglue is made from cyanoacrylate, a type of plastic that is used in many household products. It was invented in 1942 by Dr. Harry Coover, who was trying to create a clear plastic for gun sights during World War II. The first commercial superglue was sold in 1958 under the brand name 'Eastman 910'.")

claims = ["The eiffel tower is an icon latttice structure", "The eiffel tower is in japan", "The eiffel tower was built in 2019"]
results = search_claims(claims)

# print(results);

# for claim, sources in zip(claims, results):
#     print(f"\nClaim: {claim}")
#     for source in sources:
#         print(source)

for claim, sources in zip(claims, results):
    flat_sources = list(sources) 
    result = verify_claim(claim, flat_sources, llm)

# verify_claim(claim=claims, sources=results,llm=llm)