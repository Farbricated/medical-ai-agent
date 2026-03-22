import os
import operator
import sys
from typing import TypedDict, Annotated, Sequence, Literal, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rag.hybrid_retriever import HybridRetriever
from agents.diagnosis_agent import DiagnosisAgent
from agents.qa_agent import MedicalQAAgent
from agents.research_agent import MedicalResearchAgent
from utils.conversation_memory import ConversationMemory

load_dotenv()


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
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=500,
        )
        self.memory = ConversationMemory()

        print("  Initializing hybrid retriever...")
        hybrid_retriever = HybridRetriever()

        print("  Initializing agents...")
        self.diagnosis_agent = DiagnosisAgent(hybrid_retriever)
        self.qa_agent = MedicalQAAgent(hybrid_retriever)
        self.research_agent = MedicalResearchAgent()
        print("  All agents ready.")

        self.graph = self._build_graph()

    # ------------------------------------------------------------------ routing
    def _route_query(self, state: OrchestratorState) -> OrchestratorState:
        query = state["user_query"]
        session_id = state.get("session_id", "default")
        context_summary = self.memory.get_context_summary(session_id)
        state["patient_context"] = context_summary

        context_note = (
            f"\n\nPrevious patient context:\n{context_summary}"
            if context_summary != "No patient context available."
            else ""
        )

        prompt = f"""Classify the medical query into ONE category:

1. DIAGNOSIS — patient describes symptoms needing diagnostic analysis
2. QA — factual medical question needing an informational answer
3. RESEARCH — request for latest research, studies, or clinical trial data

QUERY: {query}{context_note}

Reply with ONLY ONE WORD: DIAGNOSIS, QA, or RESEARCH"""

        response = self.llm.invoke([
            SystemMessage(content="You are a medical query classification expert."),
            HumanMessage(content=prompt),
        ])
        classification = response.content.strip().upper()

        if "DIAGNOSIS" in classification:
            query_type = "diagnosis"
        elif "RESEARCH" in classification:
            query_type = "research"
        else:
            query_type = "qa"

        state["query_type"] = query_type
        state["routing_reasoning"] = f"Classified as {query_type.upper()}"
        state["messages"] = list(state.get("messages", [])) + [
            f"Routed to {query_type.upper()} agent"
        ]
        return state

    # ------------------------------------------------------------------ agents
    def _execute_diagnosis(self, state: OrchestratorState) -> OrchestratorState:
        result = self.diagnosis_agent.diagnose(
            symptoms=state["user_query"],
            patient_history=state.get("patient_context", ""),
        )
        state["agent_response"] = result
        state["messages"] = list(state.get("messages", [])) + ["Diagnosis agent completed"]
        return state

    def _execute_qa(self, state: OrchestratorState) -> OrchestratorState:
        result = self.qa_agent.ask(state["user_query"])
        state["agent_response"] = result
        state["messages"] = list(state.get("messages", [])) + ["QA agent completed"]
        return state

    def _execute_research(self, state: OrchestratorState) -> OrchestratorState:
        result = self.research_agent.research(state["user_query"])
        state["agent_response"] = result
        state["messages"] = list(state.get("messages", [])) + ["Research agent completed"]
        return state

    # ------------------------------------------------------------------ format
    def _format_response(self, state: OrchestratorState) -> OrchestratorState:
        query_type = state["query_type"]
        response = state["agent_response"]
        session_id = state.get("session_id", "default")

        self.memory.add_message(session_id=session_id, role="user", content=state["user_query"])

        if query_type == "diagnosis":
            formatted = (
                f"🏥 DIAGNOSIS ANALYSIS\n"
                f"{'━'*48}\n"
                f"📋 Diagnosis: {response['diagnosis']}\n"
                f"📊 Confidence: {response['confidence']:.0%}\n\n"
                f"💊 RECOMMENDATIONS:\n{response['recommendations']}\n\n"
                f"📚 Evidence: Based on {response['retrieved_docs_count']} medical documents\n\n"
                f"⚠️ This is for informational purposes only. Please consult a qualified healthcare professional."
            )
            self.memory.add_message(
                session_id=session_id,
                role="assistant",
                content=formatted,
                metadata={
                    "query_type": "diagnosis",
                    "diagnosis": response["diagnosis"],
                    "confidence": response["confidence"],
                },
            )

        elif query_type == "qa":
            formatted = (
                f"❓ MEDICAL Q&A\n"
                f"{'━'*48}\n"
                f"Q: {response['question']}\n\n"
                f"A: {response['answer']}\n\n"
                f"📚 Sources: {response['retrieved_docs_count']} documents\n\n"
                f"⚠️ Always consult a healthcare professional for medical advice."
            )
            self.memory.add_message(
                session_id=session_id,
                role="assistant",
                content=formatted,
                metadata={"query_type": "qa"},
            )

        else:  # research
            formatted = (
                f"🔬 RESEARCH SYNTHESIS\n"
                f"{'━'*48}\n"
                f"📖 Topic: {response['query']}\n\n"
                f"{response['findings']}\n\n"
                f"📄 Analysed {response['total_papers']} recent PubMed papers\n"
            )
            if response["key_papers"]:
                formatted += "\n🔗 KEY PAPERS:\n"
                for i, paper in enumerate(response["key_papers"], 1):
                    formatted += f"\n{i}. {paper['title']}\n   {paper['url']}\n"
            self.memory.add_message(
                session_id=session_id,
                role="assistant",
                content=formatted,
                metadata={"query_type": "research"},
            )

        state["final_response"] = formatted
        state["messages"] = list(state.get("messages", [])) + ["Response formatted"]
        return state

    def _decide_agent(self, state: OrchestratorState) -> Literal["diagnosis", "qa", "research"]:
        return state["query_type"]

    def _build_graph(self):
        workflow = StateGraph(OrchestratorState)
        workflow.add_node("route", self._route_query)
        workflow.add_node("diagnosis", self._execute_diagnosis)
        workflow.add_node("qa", self._execute_qa)
        workflow.add_node("research", self._execute_research)
        workflow.add_node("format", self._format_response)

        workflow.set_entry_point("route")
        workflow.add_conditional_edges(
            "route",
            self._decide_agent,
            {"diagnosis": "diagnosis", "qa": "qa", "research": "research"},
        )
        workflow.add_edge("diagnosis", "format")
        workflow.add_edge("qa", "format")
        workflow.add_edge("research", "format")
        workflow.add_edge("format", END)
        return workflow.compile()

    # ------------------------------------------------------------------ public API
    def process(self, user_query: str, session_id: str = "default") -> dict:
        initial_state: OrchestratorState = {
            "user_query": user_query,
            "session_id": session_id,
            "query_type": "",
            "routing_reasoning": "",
            "agent_response": {},
            "final_response": "",
            "patient_context": "",
            "messages": [],
        }
        result = self.graph.invoke(initial_state)
        return {
            "query": result["user_query"],
            "query_type": result["query_type"],
            "response": result["final_response"],
            "agent_response": result["agent_response"],   # FIX: was missing
            "patient_context": result["patient_context"],
            "process_log": list(result["messages"]),
        }

    def get_conversation_history(
        self, session_id: str = "default", last_n: Optional[int] = None
    ) -> list:
        return self.memory.get_conversation(session_id, last_n)

    def get_patient_summary(self, session_id: str = "default") -> str:
        return self.memory.get_context_summary(session_id)

    def clear_session(self, session_id: str = "default") -> None:
        self.memory.clear_session(session_id)

    def export_session(self, session_id: str = "default") -> dict:
        return self.memory.export_session(session_id)