from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
import operator
import os
from dotenv import load_dotenv

# Import your RAG components
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rag.hybrid_retriever import HybridRetriever

load_dotenv()

# State definition for LangGraph
class AgentState(TypedDict):
    symptoms: str
    patient_history: str
    retrieved_docs: list
    diagnosis: str
    confidence: float
    recommendations: str
    messages: Annotated[Sequence[str], operator.add]

class DiagnosisAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        
        # Initialize Groq LLM with updated model
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=2000
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _retrieve_knowledge(self, state: AgentState) -> AgentState:
        """Retrieve relevant medical knowledge"""
        query = f"{state['symptoms']} {state.get('patient_history', '')}"
        
        # Use search method instead of retrieve
        results = self.retriever.search(query, top_k=5)
        
        state['retrieved_docs'] = results
        state['messages'] = state.get('messages', []) + [
            f"Retrieved {len(results)} relevant documents"
        ]
        return state
    
    def _analyze_symptoms(self, state: AgentState) -> AgentState:
        """Analyze symptoms using LLM with retrieved knowledge"""
        
        # Prepare context from retrieved documents
        # Handle different result formats
        context_parts = []
        for i, doc in enumerate(state['retrieved_docs']):
            if isinstance(doc, dict):
                # Check for different possible keys
                content = doc.get('content') or doc.get('text') or doc.get('document', {}).get('text', '')
                context_parts.append(f"Document {i+1}:\n{content[:500]}...")
            else:
                context_parts.append(f"Document {i+1}:\n{str(doc)[:500]}...")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are an expert medical AI assistant. Based on the provided medical knowledge and patient information, provide a detailed diagnostic analysis.

MEDICAL KNOWLEDGE:
{context}

PATIENT SYMPTOMS:
{state['symptoms']}

PATIENT HISTORY:
{state.get('patient_history', 'No additional history provided')}

Provide your analysis in the following format:
1. PRIMARY DIAGNOSIS: [Most likely condition]
2. CONFIDENCE LEVEL: [High/Medium/Low] - [percentage]
3. SUPPORTING EVIDENCE: [Key symptoms/findings that support this diagnosis]
4. DIFFERENTIAL DIAGNOSES: [Other possible conditions to consider]
5. RECOMMENDED NEXT STEPS: [Tests, examinations, or immediate actions]

Be thorough but concise. Focus on evidence-based reasoning."""

        # Get LLM response
        messages = [
            SystemMessage(content="You are an expert medical diagnostic AI assistant."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        analysis = response.content
        
        # Parse the response
        state['diagnosis'] = self._extract_diagnosis(analysis)
        state['confidence'] = self._extract_confidence(analysis)
        state['recommendations'] = self._extract_recommendations(analysis)
        state['messages'] = state.get('messages', []) + [
            "Completed diagnostic analysis"
        ]
        
        return state
    
    def _extract_diagnosis(self, analysis: str) -> str:
        """Extract primary diagnosis from analysis"""
        try:
            if "PRIMARY DIAGNOSIS:" in analysis or "PRIMARY DIAGNOSIS**:" in analysis:
                # Handle both formats
                marker = "PRIMARY DIAGNOSIS**:" if "PRIMARY DIAGNOSIS**:" in analysis else "PRIMARY DIAGNOSIS:"
                diagnosis_section = analysis.split(marker)[1]
                # Get first line and clean it
                diagnosis = diagnosis_section.split("\n")[0].strip()
                # Remove leading numbers, asterisks, etc.
                diagnosis = diagnosis.lstrip("1234567890.*- ")
                return diagnosis
            return analysis[:200]
        except:
            return "Unable to extract diagnosis"
    
    def _extract_confidence(self, analysis: str) -> float:
        """Extract confidence level"""
        try:
            if "CONFIDENCE LEVEL:" in analysis or "CONFIDENCE LEVEL**:" in analysis:
                marker = "CONFIDENCE LEVEL**:" if "CONFIDENCE LEVEL**:" in analysis else "CONFIDENCE LEVEL:"
                conf_section = analysis.split(marker)[1]
                conf_text = conf_section.split("\n")[0].lower()
                
                # Look for percentage
                if "90" in conf_text or "95" in conf_text:
                    return 0.90
                elif "high" in conf_text or "80" in conf_text or "85" in conf_text:
                    return 0.85
                elif "medium" in conf_text or "70" in conf_text or "75" in conf_text:
                    return 0.70
                elif "low" in conf_text or "50" in conf_text or "60" in conf_text:
                    return 0.50
            return 0.70
        except:
            return 0.50
    
    def _extract_recommendations(self, analysis: str) -> str:
        """Extract recommendations"""
        try:
            if "RECOMMENDED NEXT STEPS:" in analysis or "RECOMMENDED NEXT STEPS**:" in analysis:
                marker = "RECOMMENDED NEXT STEPS**:" if "RECOMMENDED NEXT STEPS**:" in analysis else "RECOMMENDED NEXT STEPS:"
                rec_section = analysis.split(marker)[1]
                # Get the section, clean up
                rec_text = rec_section.strip()
                # If there's another numbered section after, cut it off
                if "\n\n" in rec_text:
                    rec_text = rec_text.split("\n\n")[0]
                return rec_text.lstrip("1234567890.*- ")
            return "Consult with healthcare provider for further evaluation"
        except:
            return "Further medical evaluation recommended"
    
    def _build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_knowledge)
        workflow.add_node("analyze", self._analyze_symptoms)
        
        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "analyze")
        workflow.add_edge("analyze", END)
        
        return workflow.compile()
    
    def diagnose(self, symptoms: str, patient_history: str = "") -> dict:
        """Main diagnosis method"""
        initial_state = {
            "symptoms": symptoms,
            "patient_history": patient_history,
            "retrieved_docs": [],
            "diagnosis": "",
            "confidence": 0.0,
            "recommendations": "",
            "messages": []
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "diagnosis": result["diagnosis"],
            "confidence": result["confidence"],
            "recommendations": result["recommendations"],
            "retrieved_docs_count": len(result["retrieved_docs"]),
            "process_log": result["messages"]
        }
