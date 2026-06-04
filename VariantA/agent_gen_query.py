import ollama
import datetime

ONTOLOGY_CONTEXT = """
PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

Classes: SkinCancer, Melanoma, NonMelanomaCancer, BasalCellCarcinoma, 
         SquamousCellCarcinoma, ActinicKeratosis, Symptom, Asymmetry, 
         ColorVariation, DiameterChange, IrregularBorder, Bleeding, Itching,
         Elevation, PearlAppearance, OpenSore, ScalyPatch,
         RiskFactor, UVExposure, FamilyHistory, FairSkin, Smoking,
         DiagnosticMethod, Biopsy, Dermoscopy, VisualExamination,
         Treatment, SurgicalExcision, Immunotherapy, RadiationTherapy,
         Chemotherapy, Cryotherapy, MohsSurgery, TopicalTherapy,
         BodyRegion, Face, Neck, Back, Arms, Legs, Scalp

Properties (subject → object):
- sc:Melanoma sc:hasSymptom ?symptom
- sc:BasalCellCarcinoma sc:treatedWith ?treatment
- sc:SquamousCellCarcinoma sc:hasRiskFactor ?risk
- sc:Melanoma sc:diagnosedBy ?method
- sc:Melanoma sc:commonlyFoundOn ?region
- ?type rdfs:subClassOf sc:SkinCancer

IMPORTANT EXAMPLES - follow these exactly:

To list all cancer types:
PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?type WHERE { ?type rdfs:subClassOf sc:SkinCancer }

To list all risk factors for skin cancer in general:
PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?cancer ?risk WHERE { 
  ?cancer rdfs:subClassOf sc:SkinCancer .
  ?cancer sc:hasRiskFactor ?risk 
}

To get treatments (NOT diagnoses) for melanoma:
PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
SELECT ?treatment WHERE { sc:Melanoma sc:treatedWith ?treatment }

To get diagnostic methods for melanoma:
PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
SELECT ?method WHERE { sc:Melanoma sc:diagnosedBy ?method }

IMPORTANT: 
- "treated" or "treatment" → use sc:treatedWith
- "diagnosed" or "diagnosis" → use sc:diagnosedBy
- NEVER mix these two properties up

# Reverse lookup examples
To find which cancers are treated with a specific treatment:
PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
SELECT ?cancer WHERE { ?cancer sc:treatedWith sc:Immunotherapy }

To find which cancers have a specific risk factor:
PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
SELECT ?cancer WHERE { ?cancer sc:hasRiskFactor sc:UVExposure }

To find which cancers appear in a specific body region:
PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
SELECT ?cancer WHERE { ?cancer sc:commonlyFoundOn sc:Face }

IMPORTANT for reverse lookups:
- ALWAYS use ?cancer as the subject variable
- ALWAYS use the specific value (sc:Immunotherapy, sc:UVExposure etc.) as the object
- NEVER swap subject and object

ALWAYS use PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
NEVER use any other prefix URL.
"""

def generate_sparql(question, feedback=None):
    feedback_section = ""
    if feedback:
        feedback_section = f"""
The previous query was invalid. Feedback: {feedback}
Please fix the query based on this feedback.
"""

    prompt = f"""
You are a SPARQL expert. Generate a SPARQL query for this ontology:

{ONTOLOGY_CONTEXT}

Question: {question}
{feedback_section}

Rules:
- Reply with ONLY the SPARQL query, nothing else
- ALWAYS start with PREFIX
- ALWAYS include SELECT and WHERE {{ }}
- Use exact names from the ontology above

Example:
PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
SELECT ?x WHERE {{ sc:Case01 sc:hasSymptom ?x }}
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    query = response['message']['content']
    query = query.replace("```sparql", "").replace("```", "").strip()
    query = query.replace("<sc:", "sc:")
    query = query.replace("http://wwwsemanticweb", "http://www.semanticweb")
    if "PREFIX sc:" not in query:
        query = "PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>\n" + query

    return query


#only when running this file
if __name__ == "__main__":
    questions = [
        "Which skin cancers are treated with immunotherapy?",
        "Which skin cancers commonly appear on the face?",
        "What do melanoma and basal cell carcinoma have in common?",
        "Which cancers have UV exposure as a risk factor?",
        "What are all the treatments available for skin cancer?",
        "Which cancer has the most symptoms?",
        "What skin cancers can be treated with cryotherapy?",
    ]

    with open("../Results/agent3_results.txt", "w") as f:
        f.write(f"Agent 3 Test Results\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        for question in questions:
            print(f"Processing: {question}")
            sparql = generate_sparql(question)
            f.write(f"Question: {question}\n")
            f.write(f"Generated SPARQL:\n{sparql}\n")
            f.write("-" * 60 + "\n\n")

    print("\nResults saved to agent3_results.txt")
    print("Done")