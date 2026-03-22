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


class QAState(TypedDict):
    question: str
    context: str
    retrieved_docs: list
    answer: str
    sources: list
    confidence: float
    messages: Annotated[Sequence[str], operator.add]


class MedicalQAAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1500,
        )
        self.graph = self._build_graph()

    def _retrieve_context(self, state: QAState) -> QAState:
        results = self.retriever.search(state["question"], top_k=5)
        state["retrieved_docs"] = results
        state["messages"] = list(state.get("messages", [])) + [
            f"Retrieved {len(results)} relevant documents"
        ]
        return state

    def _generate_answer(self, state: QAState) -> QAState:
        context_parts = []
        sources = []

        for i, doc in enumerate(state["retrieved_docs"]):
            if isinstance(doc, dict):
                content = (
                    doc.get("content")
                    or doc.get("text")
                    or doc.get("document", {}).get("text", "")
                )
                source = doc.get("source") or doc.get("metadata", {}).get(
                    "source", f"Document {i+1}"
                )
                context_parts.append(f"[Source {i+1} - {source}]: {content[:500]}")
                sources.append(source)
            else:
                context_parts.append(f"[Source {i+1}]: {str(doc)[:500]}")
                sources.append(f"Document {i+1}")

        context = "\n\n".join(context_parts) if context_parts else "No context available."

        prompt = f"""You are an expert medical AI assistant. Answer the following medical question accurately and clearly.

CONTEXT FROM MEDICAL KNOWLEDGE BASE:
{context}

QUESTION:
{state['question']}

INSTRUCTIONS:
1. Answer directly and clearly based on the provided context
2. If context is insufficient, say so honestly
3. Cite sources where relevant (e.g. "According to Source 1...")
4. Keep the answer concise but complete
5. This is for educational purposes only — always recommend consulting a healthcare professional

ANSWER:"""

        response = self.llm.invoke([
            SystemMessage(content="You are an expert medical AI assistant providing accurate, evidence-based answers."),
            HumanMessage(content=prompt),
        ])

        state["answer"] = response.content
        state["context"] = context
        state["sources"] = sources
        state["confidence"] = 0.85 if context_parts else 0.50
        state["messages"] = list(state.get("messages", [])) + [
            "Generated answer from retrieved knowledge"
        ]
        return state

    def _build_graph(self):
        workflow = StateGraph(QAState)
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("generate", self._generate_answer)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def ask(self, question: str) -> dict:
        initial_state: QAState = {
            "question": question,
            "context": "",
            "retrieved_docs": [],
            "answer": "",
            "sources": [],
            "confidence": 0.0,
            "messages": [],
        }
        result = self.graph.invoke(initial_state)
        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "retrieved_docs_count": len(result["retrieved_docs"]),
            "process_log": list(result["messages"]),
        }