import requests
import ollama

print("Testing GraphDB...")
try:
    response = requests.get(
        "http://localhost:7200/repositories/skin_cancer_expanded",
        params={"query": "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"},
        headers={"Accept": "application/sparql-results+json"}
    )
    data = response.json()
    print("GraphDB connected successfully")
    print(f"Sample result: {data['results']['bindings'][0]}")
except Exception as e:
    print(f"GraphDB failed: {e}")

print("\nTesting Ollama...")
try:
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "Say only: hello"}]
    )
    print("Ollama connected successfully")
    print(f"Response: {response['message']['content']}")
except Exception as e:
    print(f"Ollama failed: {e}")