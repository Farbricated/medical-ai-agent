from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence, Literal, Optional
from langgraph.graph import StateGraph, END
import operator
import os
from dotenv import load_dotenv

# Import all agents
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rag.hybrid_retriever import HybridRetriever
from agents.diagnosis_agent import DiagnosisAgent
from agents.qa_agent import MedicalQAAgent
from agents.research_agent import MedicalResearchAgent
from utils.conversation_memory import ConversationMemory

load_dotenv()

# State definition
class OrchestratorState(TypedDict):
    user_query: str
    session_id: str
    query_type: str
    routing_reasoning: str
    agent_response: dict
    final_response: str
    patient_context: str
    messages: Annotated[Sequence[str], operator.add]

class MedicalAgentOrchestrator:
    def __init__(self):
        # Initialize LLM for routing
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=500
        )
        
        # Initialize memory
        self.memory = ConversationMemory()
        
        # Initialize all agents
        print("  Initializing agents...")
        hybrid_retriever = HybridRetriever()
        
        self.diagnosis_agent = DiagnosisAgent(hybrid_retriever)
        self.qa_agent = MedicalQAAgent(hybrid_retriever)
        self.research_agent = MedicalResearchAgent()
        
        print("  All agents initialized!")
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _route_query(self, state: OrchestratorState) -> OrchestratorState:
        """Determine which agent should handle the query"""
        
        query = state['user_query']
        session_id = state.get('session_id', 'default')
        
        # Get patient context
        context_summary = self.memory.get_context_summary(session_id)
        state['patient_context'] = context_summary
        
        # Enhanced prompt with context
        context_note = f"\n\nPrevious patient context:\n{context_summary}" if context_summary != "No patient context available." else ""
        
        # Use LLM to classify query type
        prompt = f"""You are a medical AI query router. Classify the following query into ONE of these categories:

1. DIAGNOSIS: Patient presents symptoms and needs diagnostic analysis
   - Examples: "Patient has chest pain and shortness of breath", "65-year-old with fatigue and weight loss"
   
2. QA: Specific medical question that needs a factual answer
   - Examples: "What is the first-line treatment for diabetes?", "What are symptoms of pneumonia?"
   
3. RESEARCH: Requesting latest research or scientific literature
   - Examples: "What's the latest research on CAR-T therapy?", "Recent studies on GLP-1 agonists"

QUERY: {query}{context_note}

Respond with ONLY ONE WORD: DIAGNOSIS, QA, or RESEARCH"""

        messages = [
            SystemMessage(content="You are a medical query classification expert."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        classification = response.content.strip().upper()
        
        # Ensure valid classification
        if "DIAGNOSIS" in classification:
            query_type = "diagnosis"
        elif "RESEARCH" in classification:
            query_type = "research"
        else:
            query_type = "qa"
        
        state['query_type'] = query_type
        state['routing_reasoning'] = f"Classified as {query_type.upper()} query"
        state['messages'] = state.get('messages', []) + [
            f"Routed to {query_type.upper()} agent"
        ]
        
        return state
    
    def _execute_diagnosis(self, state: OrchestratorState) -> OrchestratorState:
        """Execute diagnosis agent"""
        # Include patient context in diagnosis
        patient_history = state.get('patient_context', '')
        
        result = self.diagnosis_agent.diagnose(
            symptoms=state['user_query'],
            patient_history=patient_history
        )
        state['agent_response'] = result
        state['messages'] = state.get('messages', []) + [
            "Diagnosis agent completed"
        ]
        return state
    
    def _execute_qa(self, state: OrchestratorState) -> OrchestratorState:
        """Execute QA agent"""
        result = self.qa_agent.ask(state['user_query'])
        state['agent_response'] = result
        state['messages'] = state.get('messages', []) + [
            "QA agent completed"
        ]
        return state
    
    def _execute_research(self, state: OrchestratorState) -> OrchestratorState:
        """Execute research agent"""
        result = self.research_agent.research(state['user_query'])
        state['agent_response'] = result
        state['messages'] = state.get('messages', []) + [
            "Research agent completed"
        ]
        return state
    
    def _format_response(self, state: OrchestratorState) -> OrchestratorState:
        """Format the final response and save to memory"""
        query_type = state['query_type']
        response = state['agent_response']
        session_id = state.get('session_id', 'default')
        
        # Save user query to memory
        self.memory.add_message(
            session_id=session_id,
            role='user',
            content=state['user_query']
        )
        
        if query_type == "diagnosis":
            formatted = f"""
🏥 DIAGNOSIS ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 Diagnosis: {response['diagnosis']}
📊 Confidence: {response['confidence']:.1%}

💊 RECOMMENDATIONS:
{response['recommendations']}

📚 Evidence: Based on {response['retrieved_docs_count']} medical documents
"""
            # Save diagnosis to memory with metadata
            self.memory.add_message(
                session_id=session_id,
                role='assistant',
                content=formatted,
                metadata={
                    'query_type': 'diagnosis',
                    'diagnosis': response['diagnosis'],
                    'confidence': response['confidence']
                }
            )
        
        elif query_type == "qa":
            formatted = f"""
❓ MEDICAL Q&A
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q: {response['question']}

A: {response['answer']}

📚 Sources: {response['retrieved_docs_count']} documents
"""
            self.memory.add_message(
                session_id=session_id,
                role='assistant',
                content=formatted,
                metadata={'query_type': 'qa'}
            )
        
        else:  # research
            formatted = f"""
🔬 RESEARCH SYNTHESIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📖 Topic: {response['query']}

{response['findings']}

📄 Analyzed {response['total_papers']} recent papers from PubMed

🔗 KEY PAPERS:
"""
            for i, paper in enumerate(response['key_papers'], 1):
                formatted += f"\n{i}. {paper['title']}\n   {paper['url']}\n"
            
            self.memory.add_message(
                session_id=session_id,
                role='assistant',
                content=formatted,
                metadata={'query_type': 'research'}
            )
        
        state['final_response'] = formatted
        state['messages'] = state.get('messages', []) + [
            "Response formatted and saved to memory"
        ]
        return state
    
    def _decide_agent(self, state: OrchestratorState) -> Literal["diagnosis", "qa", "research"]:
        """Decide which agent to use based on query type"""
        return state['query_type']
    
    def _build_graph(self):
        """Build LangGraph workflow with conditional routing"""
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes
        workflow.add_node("route", self._route_query)
        workflow.add_node("diagnosis", self._execute_diagnosis)
        workflow.add_node("qa", self._execute_qa)
        workflow.add_node("research", self._execute_research)
        workflow.add_node("format", self._format_response)
        
        # Define edges
        workflow.set_entry_point("route")
        
        # Conditional routing based on query type
        workflow.add_conditional_edges(
            "route",
            self._decide_agent,
            {
                "diagnosis": "diagnosis",
                "qa": "qa",
                "research": "research"
            }
        )
        
        # All agents flow to format
        workflow.add_edge("diagnosis", "format")
        workflow.add_edge("qa", "format")
        workflow.add_edge("research", "format")
        workflow.add_edge("format", END)
        
        return workflow.compile()
    
    def process(self, user_query: str, session_id: str = "default") -> dict:
        """Main orchestrator method with session support"""
        initial_state = {
            "user_query": user_query,
            "session_id": session_id,
            "query_type": "",
            "routing_reasoning": "",
            "agent_response": {},
            "final_response": "",
            "patient_context": "",
            "messages": []
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "query": result["user_query"],
            "query_type": result["query_type"],
            "response": result["final_response"],
            "patient_context": result["patient_context"],
            "process_log": result["messages"]
        }
    
    def get_conversation_history(self, session_id: str = "default", 
                                 last_n: Optional[int] = None) -> list:
        """Get conversation history for a session"""
        return self.memory.get_conversation(session_id, last_n)
    
    def get_patient_summary(self, session_id: str = "default") -> str:
        """Get patient context summary"""
        return self.memory.get_context_summary(session_id)
    
    def clear_session(self, session_id: str = "default") -> None:
        """Clear a session"""
        self.memory.clear_session(session_id)
    
    def export_session(self, session_id: str = "default") -> dict:
        """Export session data"""
        return self.memory.export_session(session_id)