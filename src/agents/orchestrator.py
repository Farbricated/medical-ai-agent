"""
MedAI – Intelligent Orchestrator v2
Uses the LLM itself to classify intent so it handles ANY conversational
situation: greetings, farewells, thanks, follow-ups, complaints,
off-topic questions, small-talk — and the three medical intents.
"""

import os
import operator
import sys
from typing import TypedDict, Annotated, Sequence, Optional

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

# ── Intent catalogue ─────────────────────────────────────────────────────────
INTENT_GREETING   = "greeting"
INTENT_FAREWELL   = "farewell"
INTENT_THANKS     = "thanks"
INTENT_COMPLAINT  = "complaint"
INTENT_FOLLOWUP   = "followup"
INTENT_SMALLTALK  = "smalltalk"
INTENT_OFFTOPIC   = "offtopic"
INTENT_UNCLEAR    = "unclear"
INTENT_DIAGNOSIS  = "diagnosis"
INTENT_QA         = "qa"
INTENT_RESEARCH   = "research"

ALL_INTENTS = {
    INTENT_GREETING, INTENT_FAREWELL, INTENT_THANKS, INTENT_COMPLAINT,
    INTENT_FOLLOWUP, INTENT_SMALLTALK, INTENT_OFFTOPIC, INTENT_UNCLEAR,
    INTENT_DIAGNOSIS, INTENT_QA, INTENT_RESEARCH,
}
MEDICAL_INTENTS = {INTENT_DIAGNOSIS, INTENT_QA, INTENT_RESEARCH}
CONV_INTENTS    = ALL_INTENTS - MEDICAL_INTENTS

# ── Conversational instruction map ───────────────────────────────────────────
_CONV_SYSTEM = (
    "You are MedAI, a warm, professional AI healthcare assistant. "
    "Be concise, empathetic, and always remind users that a real doctor "
    "should be consulted for personal medical decisions. "
    "Never fabricate medical facts."
)

_CONV_INSTRUCTIONS: dict[str, str] = {
    INTENT_GREETING: (
        "The user said hello or greeted you. Welcome them warmly, introduce "
        "yourself as MedAI in one sentence, briefly list the three things you "
        "can help with (symptom analysis, medical Q&A, research summaries), "
        "and invite them to ask their question. Max 4 sentences."
    ),
    INTENT_FAREWELL: (
        "The user is saying goodbye or ending the chat. Respond warmly, wish "
        "them well, remind them to consult a doctor for personal concerns, and "
        "say they can return anytime. Max 3 sentences."
    ),
    INTENT_THANKS: (
        "The user is expressing gratitude. Acknowledge graciously, add a "
        "brief reminder to follow up with a healthcare professional for "
        "personal concerns, and tell them you're available anytime. "
        "Max 3 sentences."
    ),
    INTENT_COMPLAINT: (
        "The user seems frustrated or is complaining. Acknowledge their "
        "feelings with empathy, apologise briefly, and ask what you can do "
        "better or what medical question you can help with. Do not be "
        "defensive. Max 3 sentences."
    ),
    INTENT_SMALLTALK: (
        "The user is making casual small talk unrelated to medicine. Respond "
        "naturally and briefly, then gently redirect to how you can help them "
        "with health topics. Max 3 sentences."
    ),
    INTENT_OFFTOPIC: (
        "The user asked something outside of medicine (e.g. code, finance, "
        "sports, weather). Acknowledge politely, explain you specialise in "
        "healthcare information, and offer to help with any medical topics. "
        "Max 3 sentences."
    ),
    INTENT_FOLLOWUP: (
        "The user is asking a follow-up question or requesting more detail "
        "about something already discussed. Use the conversation context "
        "provided to give a concise, relevant continuation. "
        "Max 4 sentences."
    ),
    INTENT_UNCLEAR: (
        "The user's message is too vague or ambiguous to categorise. "
        "Ask one clear clarifying question so you can help them better. "
        "Max 2 sentences."
    ),
}


class OrchestratorState(TypedDict):
    user_query:           str
    session_id:           str
    intent:               str
    routing_reasoning:    str
    agent_response:       dict
    final_response:       str
    patient_context:      str
    conversation_context: str
    messages:             Annotated[Sequence[str], operator.add]


class MedicalAgentOrchestrator:

    def __init__(self):
        # Classifier LLM — deterministic
        self.classifier_llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=20,
        )
        # Conversational LLM — warmer
        self.conv_llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.6,
            max_tokens=350,
        )

        self.memory = ConversationMemory()

        print("  Initialising hybrid retriever…")
        retriever = HybridRetriever()

        print("  Initialising agents…")
        self.diagnosis_agent = DiagnosisAgent(retriever)
        self.qa_agent        = MedicalQAAgent(retriever)
        self.research_agent  = MedicalResearchAgent()
        print("  All agents ready.")

        self.graph = self._build_graph()

    # ── LLM-powered intent classifier ────────────────────────────────────────
    def _classify_intent(self, query: str, conv_snippet: str) -> str:
        """
        Ask the LLM to pick exactly one intent label.
        Falls back gracefully if the model returns something unexpected.
        """
        prompt = f"""You are an intent classifier for a medical assistant chatbot.

Given the user message and the recent conversation, output EXACTLY ONE intent word
from this list (no other text):

  greeting   – saying hello / hi / hey / good morning / what's up etc.
  farewell   – saying bye / goodbye / see you / take care / cya / later etc.
  thanks     – expressing thanks / gratitude / great / awesome / perfect etc.
  complaint  – frustrated / criticising / this is wrong / you're useless etc.
  followup   – continuing / asking more about a previously discussed medical topic
  smalltalk  – casual chit-chat unrelated to health (jokes, weather, how are you)
  offtopic   – question clearly outside medicine (code, finance, sports, recipes)
  unclear    – too vague or ambiguous to classify
  diagnosis  – user describes personal symptoms and wants diagnostic analysis
  qa         – user asks a factual medical / health question
  research   – user wants latest studies, papers or clinical trial data

Recent conversation:
{conv_snippet or "(no history)"}

User message: "{query}"

Intent:"""

        try:
            resp = self.classifier_llm.invoke([
                SystemMessage(content="Reply with ONE word only — the intent label."),
                HumanMessage(content=prompt),
            ])
            raw = resp.content.strip().lower().split()[0].rstrip(".,!?:")

            if raw in ALL_INTENTS:
                return raw

            # Soft fuzzy fallback
            mapping = {
                "diagnos": INTENT_DIAGNOSIS, "symptom": INTENT_DIAGNOSIS,
                "researc": INTENT_RESEARCH,  "pubmed":  INTENT_RESEARCH,
                "study":   INTENT_RESEARCH,  "paper":   INTENT_RESEARCH,
                "bye":     INTENT_FAREWELL,  "goodbye": INTENT_FAREWELL,
                "farewell":INTENT_FAREWELL,  "ciao":    INTENT_FAREWELL,
                "thank":   INTENT_THANKS,    "great":   INTENT_THANKS,
                "hello":   INTENT_GREETING,  "hi":      INTENT_GREETING,
                "hey":     INTENT_GREETING,
            }
            for key, intent in mapping.items():
                if key in raw:
                    return intent

            return INTENT_QA   # safe medical default

        except Exception:
            return INTENT_QA

    # ── Conversational reply generator ────────────────────────────────────────
    def _conversational_reply(self, intent: str, query: str, context: str) -> str:
        instruction = _CONV_INSTRUCTIONS.get(intent, _CONV_INSTRUCTIONS[INTENT_UNCLEAR])
        context_note = (
            f"\n\nConversation so far:\n{context}"
            if context and context not in ("No patient context available.", "No context available.")
            else ""
        )
        full_prompt = f"{instruction}{context_note}\n\nUser said: \"{query}\""

        try:
            resp = self.conv_llm.invoke([
                SystemMessage(content=_CONV_SYSTEM),
                HumanMessage(content=full_prompt),
            ])
            return resp.content.strip()
        except Exception as e:
            return (
                "I apologise — something went wrong on my end. "
                f"How can I help you with a health question? (error: {e})"
            )

    # ── LangGraph nodes ───────────────────────────────────────────────────────
    def _route_query(self, state: OrchestratorState) -> OrchestratorState:
        session = state.get("session_id", "default")

        # Build patient context summary
        state["patient_context"] = self.memory.get_context_summary(session)

        # Build short conversation snippet
        recent = self.memory.get_conversation(session, last_n=6)
        snippet = "\n".join(
            f"{m['role'].title()}: {m['content'][:150]}" for m in recent
        )
        state["conversation_context"] = snippet

        intent = self._classify_intent(state["user_query"], snippet)
        state["intent"]            = intent
        state["routing_reasoning"] = f"LLM intent: {intent}"
        state["messages"]          = list(state.get("messages", [])) + [f"Intent → {intent}"]
        return state

    def _execute_diagnosis(self, state: OrchestratorState) -> OrchestratorState:
        result = self.diagnosis_agent.diagnose(
            symptoms=state["user_query"],
            patient_history=state.get("patient_context", ""),
        )
        state["agent_response"] = result
        state["messages"]       = list(state.get("messages", [])) + ["Diagnosis ✓"]
        return state

    def _execute_qa(self, state: OrchestratorState) -> OrchestratorState:
        result = self.qa_agent.ask(state["user_query"])
        state["agent_response"] = result
        state["messages"]       = list(state.get("messages", [])) + ["Q&A ✓"]
        return state

    def _execute_research(self, state: OrchestratorState) -> OrchestratorState:
        result = self.research_agent.research(state["user_query"])
        state["agent_response"] = result
        state["messages"]       = list(state.get("messages", [])) + ["Research ✓"]
        return state

    def _execute_conversational(self, state: OrchestratorState) -> OrchestratorState:
        reply = self._conversational_reply(
            intent  = state["intent"],
            query   = state["user_query"],
            context = state.get("conversation_context", ""),
        )
        state["agent_response"] = {"confidence": 1.0, "reply": reply}
        state["messages"]       = list(state.get("messages", [])) + [
            f"Conversational ({state['intent']}) ✓"
        ]
        return state

    def _format_response(self, state: OrchestratorState) -> OrchestratorState:
        intent  = state["intent"]
        resp    = state["agent_response"]
        session = state.get("session_id", "default")

        self.memory.add_message(session_id=session, role="user", content=state["user_query"])

        if intent == INTENT_DIAGNOSIS:
            formatted = (
                f"### 🏥 Diagnosis Analysis\n\n"
                f"**Primary Diagnosis:** {resp['diagnosis']}\n\n"
                f"**Confidence:** {resp['confidence']:.0%}\n\n"
                f"---\n\n"
                f"**Recommendations:**\n{resp['recommendations']}\n\n"
                f"**Full Analysis:**\n{resp['full_analysis']}\n\n"
                f"> 📚 Based on {resp['retrieved_docs_count']} medical documents\n\n"
                f"> ⚠️ *For informational purposes only. Please consult a qualified healthcare professional.*"
            )
            self.memory.add_message(
                session_id=session, role="assistant", content=formatted,
                metadata={"query_type": "diagnosis",
                          "diagnosis": resp["diagnosis"],
                          "confidence": resp["confidence"]},
            )

        elif intent == INTENT_QA:
            formatted = (
                f"### ❓ Medical Q&A\n\n"
                f"{resp['answer']}\n\n"
                f"> 📚 Sources: {resp['retrieved_docs_count']} documents\n\n"
                f"> ⚠️ *Always consult a healthcare professional for personal medical advice.*"
            )
            self.memory.add_message(
                session_id=session, role="assistant", content=formatted,
                metadata={"query_type": "qa"},
            )

        elif intent == INTENT_RESEARCH:
            formatted = (
                f"### 🔬 Research Synthesis\n\n"
                f"**Topic:** {resp['query']}\n\n"
                f"{resp['findings']}\n\n"
                f"> 📄 Analysed {resp['total_papers']} recent PubMed papers"
            )
            if resp.get("key_papers"):
                formatted += "\n\n**🔗 Key Papers:**\n"
                for i, p in enumerate(resp["key_papers"], 1):
                    formatted += f"\n{i}. [{p['title']}]({p['url']})"
            self.memory.add_message(
                session_id=session, role="assistant", content=formatted,
                metadata={"query_type": "research"},
            )

        else:
            # All conversational intents
            formatted = resp.get("reply", "How can I help you with a health question?")
            self.memory.add_message(
                session_id=session, role="assistant", content=formatted,
                metadata={"query_type": intent},
            )

        state["final_response"] = formatted
        state["messages"]       = list(state.get("messages", [])) + ["Formatted ✓"]
        return state

    # ── Routing decision ─────────────────────────────────────────────────────
    def _decide_path(self, state: OrchestratorState) -> str:
        intent = state["intent"]
        if intent == INTENT_DIAGNOSIS: return "diagnosis"
        if intent == INTENT_RESEARCH:  return "research"
        if intent == INTENT_QA:        return "qa"
        return "conversational"

    # ── Build LangGraph ───────────────────────────────────────────────────────
    def _build_graph(self):
        wf = StateGraph(OrchestratorState)
        wf.add_node("route",          self._route_query)
        wf.add_node("diagnosis",      self._execute_diagnosis)
        wf.add_node("qa",             self._execute_qa)
        wf.add_node("research",       self._execute_research)
        wf.add_node("conversational", self._execute_conversational)
        wf.add_node("format",         self._format_response)

        wf.set_entry_point("route")
        wf.add_conditional_edges(
            "route", self._decide_path,
            {
                "diagnosis":      "diagnosis",
                "qa":             "qa",
                "research":       "research",
                "conversational": "conversational",
            },
        )
        for node in ("diagnosis", "qa", "research", "conversational"):
            wf.add_edge(node, "format")
        wf.add_edge("format", END)
        return wf.compile()

    # ── Public API ────────────────────────────────────────────────────────────
    def process(self, user_query: str, session_id: str = "default") -> dict:
        initial: OrchestratorState = {
            "user_query":           user_query,
            "session_id":           session_id,
            "intent":               "",
            "routing_reasoning":    "",
            "agent_response":       {},
            "final_response":       "",
            "patient_context":      "",
            "conversation_context": "",
            "messages":             [],
        }
        result = self.graph.invoke(initial)
        return {
            "query":           result["user_query"],
            "query_type":      result["intent"],
            "response":        result["final_response"],
            "agent_response":  result["agent_response"],
            "patient_context": result["patient_context"],
            "process_log":     list(result["messages"]),
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