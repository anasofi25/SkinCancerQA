import requests
import json
import logging

logger = logging.getLogger(__name__)

# Update this every Colab session
AGENT3_URL = "https://rectify-prevail-rectified.ngrok-free.dev"

def explain_answer(question: str, sparql_results: str) -> str:
    """
    Drop-in replacement for agent_explainer.explain_answer()
    Calls the fine-tuned model running on Colab via ngrok.

    Args:
        question: Original user question
        sparql_results: String of results from execute_query()

    Returns:
        Plain English answer
    """
    try:
        response = requests.post(
            f"{AGENT3_URL}/generate",
            json={
                "question": question,
                "sparql_results": sparql_results
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["answer"]

    except requests.exceptions.ConnectionError:
        logger.error("Agent 3 (Colab) is not reachable.")
        return "Error: Agent 3 is offline. Please start the Colab server."

    except requests.exceptions.Timeout:
        logger.error("Agent 3 timed out.")
        return "Error: Agent 3 timed out. Try again."

    except Exception as e:
        logger.error(f"Agent 3 unexpected error: {e}")
        return f"Error: {str(e)}"


def is_online() -> bool:
    """Optional startup check."""
    try:
        r = requests.get(f"{AGENT3_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False