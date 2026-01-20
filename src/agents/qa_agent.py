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
class QAState(TypedDict):
    question: str
    context: str
    retrieved_docs: list
    answer: str
    sources: list
    messages: Annotated[Sequence[str], operator.add]

class MedicalQAAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.2,  # Slightly higher for more natural answers
            max_tokens=1500
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _retrieve_context(self, state: QAState) -> QAState:
        """Retrieve relevant medical knowledge for the question"""
        
        # Search for relevant documents
        results = self.retriever.search(state['question'], top_k=5)
        
        state['retrieved_docs'] = results
        state['messages'] = state.get('messages', []) + [
            f"Retrieved {len(results)} relevant documents"
        ]
        return state
    
    def _generate_answer(self, state: QAState) -> QAState:
        """Generate answer using LLM with retrieved context"""
        
        # Prepare context from retrieved documents
        context_parts = []
        sources = []
        
        for i, doc in enumerate(state['retrieved_docs']):
            if isinstance(doc, dict):
                content = doc.get('content') or doc.get('text') or doc.get('document', {}).get('text', '')
                source = doc.get('source') or doc.get('metadata', {}).get('source', f'Document {i+1}')
                
                context_parts.append(f"[Source {i+1}]: {content[:400]}...")
                sources.append(source)
            else:
                context_parts.append(f"[Source {i+1}]: {str(doc)[:400]}...")
                sources.append(f'Document {i+1}')
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are an expert medical AI assistant. Answer the following medical question using ONLY the provided context. Be accurate, concise, and cite your sources.

CONTEXT FROM MEDICAL KNOWLEDGE BASE:
{context}

QUESTION:
{state['question']}

INSTRUCTIONS:
1. Answer the question directly and clearly
2. Use only information from the provided context
3. If the context doesn't contain enough information, say so honestly
4. Keep your answer concise but complete
5. Mention which sources support your answer (e.g., "According to Source 1...")

ANSWER:"""

        # Get LLM response
        messages = [
            SystemMessage(content="You are an expert medical AI assistant. Provide accurate, evidence-based answers."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        state['answer'] = answer
        state['context'] = context
        state['sources'] = sources
        state['messages'] = state.get('messages', []) + [
            "Generated answer from retrieved knowledge"
        ]
        
        return state
    
    def _build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(QAState)
        
        # Add nodes - CHANGED: "answer" -> "generate" to avoid conflict with state key
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("generate", self._generate_answer)
        
        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def ask(self, question: str) -> dict:
        """Main QA method"""
        initial_state = {
            "question": question,
            "context": "",
            "retrieved_docs": [],
            "answer": "",
            "sources": [],
            "messages": []
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_docs_count": len(result["retrieved_docs"]),
            "process_log": result["messages"]
        }
