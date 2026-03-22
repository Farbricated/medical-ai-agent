import os
import operator
import sys
from typing import TypedDict, Annotated, Sequence

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rag.hybrid_retriever import HybridRetriever

load_dotenv()


class AgentState(TypedDict):
    symptoms: str
    patient_history: str
    retrieved_docs: list
    diagnosis: str
    confidence: float
    recommendations: str
    full_analysis: str
    messages: Annotated[Sequence[str], operator.add]


class DiagnosisAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=2000,
        )
        self.graph = self._build_graph()

    def _retrieve_knowledge(self, state: AgentState) -> AgentState:
        query = f"{state['symptoms']} {state.get('patient_history', '')}"
        results = self.retriever.search(query, top_k=5)
        state["retrieved_docs"] = results
        state["messages"] = list(state.get("messages", [])) + [
            f"Retrieved {len(results)} relevant documents"
        ]
        return state

    def _analyze_symptoms(self, state: AgentState) -> AgentState:
        context_parts = []
        for i, doc in enumerate(state["retrieved_docs"]):
            if isinstance(doc, dict):
                content = (
                    doc.get("content")
                    or doc.get("text")
                    or doc.get("document", {}).get("text", "")
                )
                context_parts.append(f"Document {i+1}:\n{content[:600]}")
            else:
                context_parts.append(f"Document {i+1}:\n{str(doc)[:600]}")
        context = "\n\n".join(context_parts) if context_parts else "No context retrieved."

        prompt = f"""You are an expert medical AI assistant. Analyze the symptoms and provide a diagnostic assessment.

MEDICAL KNOWLEDGE:
{context}

PATIENT SYMPTOMS:
{state['symptoms']}

PATIENT HISTORY:
{state.get('patient_history', 'No additional history provided')}

Provide your analysis in this EXACT format:
PRIMARY DIAGNOSIS: [Most likely condition]
CONFIDENCE LEVEL: [High/Medium/Low] - [percentage e.g. 85%]
SUPPORTING EVIDENCE: [Key symptoms/findings supporting this diagnosis]
DIFFERENTIAL DIAGNOSES: [2-3 other possible conditions]
RECOMMENDED NEXT STEPS: [Tests, examinations, or immediate actions]

Be thorough but concise. This is for informational purposes only."""

        response = self.llm.invoke([
            SystemMessage(content="You are an expert medical diagnostic AI assistant."),
            HumanMessage(content=prompt),
        ])
        analysis = response.content

        state["full_analysis"] = analysis
        state["diagnosis"] = self._extract_section(analysis, "PRIMARY DIAGNOSIS:")
        state["confidence"] = self._extract_confidence(analysis)
        state["recommendations"] = self._extract_section(analysis, "RECOMMENDED NEXT STEPS:")
        state["messages"] = list(state.get("messages", [])) + ["Completed diagnostic analysis"]
        return state

    def _extract_section(self, text: str, marker: str) -> str:
        try:
            # Handle bold markdown variants like **PRIMARY DIAGNOSIS:**
            for variant in [marker, marker.replace(":", "**:"), f"**{marker}"]:
                if variant in text:
                    section = text.split(variant)[1]
                    line = section.split("\n")[0].strip().lstrip("1234567890.*- ")
                    if line:
                        return line
        except Exception:
            pass
        return f"Unable to extract {marker}"

    def _extract_confidence(self, text: str) -> float:
        try:
            marker = "CONFIDENCE LEVEL:"
            if marker in text:
                section = text.split(marker)[1].split("\n")[0].lower()
                for pct in ["95", "90", "85", "80", "75", "70", "65", "60", "50"]:
                    if pct in section:
                        return float(pct) / 100
                if "high" in section:
                    return 0.85
                if "medium" in section:
                    return 0.70
                if "low" in section:
                    return 0.50
        except Exception:
            pass
        return 0.70

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve", self._retrieve_knowledge)
        workflow.add_node("analyze", self._analyze_symptoms)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "analyze")
        workflow.add_edge("analyze", END)
        return workflow.compile()

    def diagnose(self, symptoms: str, patient_history: str = "") -> dict:
        initial_state: AgentState = {
            "symptoms": symptoms,
            "patient_history": patient_history,
            "retrieved_docs": [],
            "diagnosis": "",
            "confidence": 0.0,
            "recommendations": "",
            "full_analysis": "",
            "messages": [],
        }
        result = self.graph.invoke(initial_state)
        return {
            "diagnosis": result["diagnosis"],
            "confidence": result["confidence"],
            "recommendations": result["recommendations"],
            "full_analysis": result["full_analysis"],
            "retrieved_docs_count": len(result["retrieved_docs"]),
            "process_log": list(result["messages"]),
        }