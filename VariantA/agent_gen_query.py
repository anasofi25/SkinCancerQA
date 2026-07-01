import ollama
import datetime

ONTOLOGY_CONTEXT = """
PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

Cancer types
------------
sc:Melanoma
sc:BasalCellCarcinoma
sc:SquamousCellCarcinoma
sc:ActinicKeratosis

Symptoms (ontology resources)
-----------------------------
sc:Asymmetry
sc:ColorVariation
sc:DiameterChange
sc:IrregularBorder
sc:Bleeding
sc:Itching
sc:Elevation
sc:PearlAppearance
sc:OpenSore
sc:ScalyPatch

Risk factors
------------
sc:UVExposure
sc:FamilyHistory
sc:FairSkin
sc:Smoking

Body regions
------------
sc:Face
sc:Neck
sc:Back
sc:Arms
sc:Legs
sc:Scalp

Diagnostic methods
------------------
sc:Biopsy
sc:Dermoscopy
sc:VisualExamination

Treatments
----------
sc:SurgicalExcision
sc:Immunotherapy
sc:RadiationTherapy
sc:Chemotherapy
sc:Cryotherapy
sc:MohsSurgery
sc:TopicalTherapy

Properties (direction is IMPORTANT)

Cancer ----sc:hasSymptom----> Symptom

Cancer ----sc:hasRiskFactor----> RiskFactor

Cancer ----sc:treatedWith----> Treatment

Cancer ----sc:diagnosedBy----> DiagnosticMethod

Cancer ----sc:commonlyFoundOn----> BodyRegion

These properties are NEVER used in reverse.
IMPORTANT

The ontology uses ontology resources, NOT string literals.

Correct

?cancer sc:hasSymptom sc:Itching .

Wrong

?symptom rdfs:label "itching"

Wrong

?cancer sc:hasSymptom [
    rdfs:label "itching"
]

Never use

rdfs:label

FILTER

CONTAINS

Regex

Blank nodes

Always use the ontology URI exactly as listed.

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

# Forward, single-entity lookup (use this when the question asks about
# ONE specific cancer's risk factors/symptoms/treatments — e.g.
# "Is smoking a risk factor for basal cell carcinoma?",
# "What are the risk factors for melanoma?")
To get the risk factors for a SPECIFIC cancer (forward lookup):
PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
SELECT ?risk WHERE { sc:BasalCellCarcinoma sc:hasRiskFactor ?risk }

To get the symptoms for a SPECIFIC cancer (forward lookup):
PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
SELECT ?symptom WHERE { sc:Melanoma sc:hasSymptom ?symptom }

# Reverse lookup examples
# (use this ONLY when the question asks "WHICH cancers have property X",
# i.e. the property value is already known and you're searching for cancers)
To find which cancers are treated with a specific treatment:
PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
SELECT ?cancer WHERE { ?cancer sc:treatedWith sc:Immunotherapy }

To find which cancers have a specific risk factor:
PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
SELECT ?cancer WHERE { ?cancer sc:hasRiskFactor sc:UVExposure }

To find which cancers appear in a specific body region:
PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>
SELECT ?cancer WHERE { ?cancer sc:commonlyFoundOn sc:Face }

To find cancers with multiple symptoms and a body region:

PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>

SELECT ?cancer
WHERE{
    ?cancer sc:commonlyFoundOn sc:Face .
    ?cancer sc:hasSymptom sc:Itching .
    ?cancer sc:hasSymptom sc:Bleeding .
    ?cancer sc:hasSymptom sc:Elevation .
}

IMPORTANT for choosing forward vs reverse lookup:
- If the question names ONE cancer and asks about ITS risk factors/symptoms/
  treatments → use the FORWARD lookup pattern (cancer is the subject,
  the unknown property value is the variable).
- If the question names a property value (e.g. "smoking", "immunotherapy",
  "face") and asks WHICH cancers have it → use the REVERSE lookup pattern
  (?cancer is the variable, the known value is the object).
- Example: "Is smoking a risk factor for basal cell carcinoma?" is about
  ONE cancer (BasalCellCarcinoma) → use FORWARD lookup, NOT reverse lookup.

IMPORTANT for reverse lookups specifically:
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

- Reply ONLY with SPARQL.
- Do not explain your answer.
- Never invent ontology classes or properties.
- Never use rdfs:label to identify ontology entities.
- Never create blank nodes.
- Always use ontology resources such as sc:Itching or sc:Face.
- Follow the property direction exactly.
- Use only names appearing in the ontology context.
- Always start with PREFIX.

"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0,
            "seed": 42,
        },
    )

    query = response['message']['content']
    query = query.replace("```sparql", "").replace("```", "").strip()
    query = query.replace("<sc:", "sc:")
    query = query.replace("http://wwwsemanticweb", "http://www.semanticweb")
    if "PREFIX sc:" not in query:
        query = "PREFIX sc: <http://www.semanticweb.org/skincancer_exp#>\n" + query

    return query

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
    case_questions = [
        "A 73-year-old patient has a lesion on the face with itching, bleeding and elevation. What type of skin cancer commonly appears on the face with these symptoms?",
        "A patient who smokes has a growing, itchy lesion on the nose. Smoking is a known risk factor for which skin cancer?",
        "A patient with a history of previous skin cancer develops a new lesion on the back with color changes. What skin cancer has previous skin cancer and UV exposure as risk factors?",
        "A patient has an itchy, growing lesion on the face. Squamous cell carcinoma commonly appears on which body regions?",
        "A patient with fair skin and high UV exposure develops a pearl-like bump on the scalp. What symptoms are associated with basal cell carcinoma?",
        "An elderly patient has a rough, scaly patch on the forearm from years of sun exposure. What treatments are used for actinic keratosis?",
        "A patient notices a mole with asymmetry, irregular border and color variation on their leg. How is melanoma diagnosed?",
    ]

    with open("../Results/agent1_results.txt", "w") as f:
        f.write(f"Agent 1 Test Results\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        for question in case_questions:
            print(f"Processing: {question}")
            sparql = generate_sparql(question)
            f.write(f"Question: {question}\n")
            f.write(f"Generated SPARQL:\n{sparql}\n")
            f.write("-" * 60 + "\n\n")

    print("\nResults saved to agent1_results.txt")
    print("Done")