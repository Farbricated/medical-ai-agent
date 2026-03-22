import os
import operator
import sys
import time
from typing import TypedDict, Annotated, Sequence, List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

try:
    from Bio import Entrez
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

load_dotenv()

if BIOPYTHON_AVAILABLE:
    Entrez.email = os.getenv("EMAIL", "medai@example.com")


class ResearchState(TypedDict):
    query: str
    pubmed_results: list
    synthesized_findings: str
    key_papers: list
    total_papers: int
    messages: Annotated[Sequence[str], operator.add]


class MedicalResearchAgent:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2000,
        )
        self.graph = self._build_graph()

    def _search_pubmed(self, state: ResearchState) -> ResearchState:
        if not BIOPYTHON_AVAILABLE:
            state["pubmed_results"] = []
            state["messages"] = list(state.get("messages", [])) + [
                "BioPython not installed — PubMed search skipped"
            ]
            return state

        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=state["query"],
                retmax=10,
                sort="relevance",
                reldate=730,
            )
            record = Entrez.read(handle)
            handle.close()
            id_list = record["IdList"]

            if not id_list:
                state["pubmed_results"] = []
                state["messages"] = list(state.get("messages", [])) + ["No PubMed results found"]
                return state

            handle = Entrez.efetch(
                db="pubmed", id=id_list, rettype="abstract", retmode="xml"
            )
            papers = Entrez.read(handle)
            handle.close()

            results = []
            for paper in papers["PubmedArticle"]:
                try:
                    article = paper["MedlineCitation"]["Article"]
                    title = str(article.get("ArticleTitle", "No title"))

                    abstract = ""
                    if "Abstract" in article and "AbstractText" in article["Abstract"]:
                        parts = article["Abstract"]["AbstractText"]
                        abstract = " ".join(str(p) for p in parts) if isinstance(parts, list) else str(parts)

                    authors = []
                    for author in article.get("AuthorList", [])[:3]:
                        if "LastName" in author and "Initials" in author:
                            authors.append(f"{author['LastName']} {author['Initials']}")

                    pub_date = ""
                    try:
                        date = article["Journal"]["JournalIssue"]["PubDate"]
                        pub_date = f"{date.get('Month', '')} {date.get('Year', '')}".strip()
                    except Exception:
                        pass

                    pmid = str(paper["MedlineCitation"]["PMID"])
                    results.append({
                        "title": title,
                        "abstract": abstract[:500],
                        "authors": ", ".join(authors) if authors else "Unknown",
                        "date": pub_date,
                        "pmid": pmid,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    })
                except Exception:
                    continue

            state["pubmed_results"] = results
            state["messages"] = list(state.get("messages", [])) + [
                f"Found {len(results)} papers from PubMed"
            ]
            time.sleep(0.5)

        except Exception as e:
            state["pubmed_results"] = []
            state["messages"] = list(state.get("messages", [])) + [
                f"PubMed search error: {str(e)}"
            ]
        return state

    def _synthesize_findings(self, state: ResearchState) -> ResearchState:
        results = state["pubmed_results"]

        if not results:
            state["synthesized_findings"] = (
                "No PubMed papers were retrieved. This may be due to network limitations "
                "or BioPython not being available. Please check https://pubmed.ncbi.nlm.nih.gov/ directly."
            )
            state["key_papers"] = []
            state["total_papers"] = 0
            return state

        papers_text = "\n---\n".join(
            f"Paper {i}:\nTitle: {p['title']}\nAuthors: {p['authors']}\n"
            f"Date: {p['date']}\nAbstract: {p['abstract']}"
            for i, p in enumerate(results, 1)
        )

        prompt = f"""You are a medical research analyst. Synthesize these PubMed papers on: "{state['query']}"

RESEARCH PAPERS:
{papers_text}

Provide a structured synthesis:
1. MAIN FINDINGS: Key discoveries and trends across papers
2. CLINICAL IMPLICATIONS: What these findings mean for practice
3. RESEARCH GAPS: What remains unanswered
4. KEY PAPERS: The 3 most impactful papers and why

Be concise and actionable."""

        response = self.llm.invoke([
            SystemMessage(content="You are an expert medical research analyst."),
            HumanMessage(content=prompt),
        ])

        state["synthesized_findings"] = response.content
        state["key_papers"] = results[:3]
        state["total_papers"] = len(results)
        state["messages"] = list(state.get("messages", [])) + ["Research synthesis complete"]
        return state

    def _build_graph(self):
        workflow = StateGraph(ResearchState)
        workflow.add_node("search", self._search_pubmed)
        workflow.add_node("synthesize", self._synthesize_findings)
        workflow.set_entry_point("search")
        workflow.add_edge("search", "synthesize")
        workflow.add_edge("synthesize", END)
        return workflow.compile()

    def research(self, query: str) -> dict:
        initial_state: ResearchState = {
            "query": query,
            "pubmed_results": [],
            "synthesized_findings": "",
            "key_papers": [],
            "total_papers": 0,
            "messages": [],
        }
        result = self.graph.invoke(initial_state)
        return {
            "query": result["query"],
            "findings": result["synthesized_findings"],
            "key_papers": result["key_papers"],
            "total_papers": result["total_papers"],
            "process_log": list(result["messages"]),
        }