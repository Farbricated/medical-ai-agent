from langchain_groq import ChatGroq
from langchain_core.message import HumanMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence, List, Dict
from langgraph.graph import StateGraph, END
import operator
import os
from dotenv import load_dotenv
from Bio import Entrez
import time

# Import your RAG components
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

load_dotenv()

# Set your email for PubMed (required by NCBI)
Entrez.email = os.getenv("EMAIL", "your-email@example.com")

# State definition for LangGraph
class ResearchState(TypedDict):
    query: str
    pubmed_results: list
    synthesized_findings: str
    key_papers: list
    messages: Annotated[Sequence[str], operator.add]

class MedicalResearchAgent:
    def __init__(self):
        # Initialize Groq LLM
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2000
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _search_pubmed(self, state: ResearchState) -> ResearchState:
        """Search PubMed for relevant research papers"""
        query = state['query']
        
        try:
            # Search PubMed
            print(f"  Searching PubMed for: '{query}'")
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=10,
                sort="relevance",
                reldate=730  # Last 2 years
            )
            record = Entrez.read(handle)
            handle.close()
            
            id_list = record["IdList"]
            
            if not id_list:
                state['pubmed_results'] = []
                state['messages'] = state.get('messages', []) + [
                    "No PubMed results found"
                ]
                return state
            
            # Fetch details for papers
            print(f"  Fetching details for {len(id_list)} papers...")
            handle = Entrez.efetch(
                db="pubmed",
                id=id_list,
                rettype="abstract",
                retmode="xml"
            )
            papers = Entrez.read(handle)
            handle.close()
            
            # Extract paper information
            results = []
            for paper in papers['PubmedArticle']:
                try:
                    article = paper['MedlineCitation']['Article']
                    
                    # Extract title
                    title = article.get('ArticleTitle', 'No title')
                    
                    # Extract abstract
                    abstract = ""
                    if 'Abstract' in article and 'AbstractText' in article['Abstract']:
                        abstract_parts = article['Abstract']['AbstractText']
                        if isinstance(abstract_parts, list):
                            abstract = ' '.join(str(part) for part in abstract_parts)
                        else:
                            abstract = str(abstract_parts)
                    
                    # Extract authors
                    authors = []
                    if 'AuthorList' in article:
                        for author in article['AuthorList'][:3]:  # First 3 authors
                            if 'LastName' in author and 'Initials' in author:
                                authors.append(f"{author['LastName']} {author['Initials']}")
                    
                    # Extract publication date
                    pub_date = ""
                    if 'Journal' in article and 'JournalIssue' in article['Journal']:
                        issue = article['Journal']['JournalIssue']
                        if 'PubDate' in issue:
                            date = issue['PubDate']
                            year = date.get('Year', '')
                            month = date.get('Month', '')
                            pub_date = f"{month} {year}".strip()
                    
                    # Extract PMID
                    pmid = paper['MedlineCitation']['PMID']
                    
                    results.append({
                        'title': title,
                        'abstract': abstract[:500],  # First 500 chars
                        'authors': ', '.join(authors) if authors else 'Unknown',
                        'date': pub_date,
                        'pmid': str(pmid),
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    })
                    
                except Exception as e:
                    print(f"  Error processing paper: {e}")
                    continue
            
            state['pubmed_results'] = results
            state['messages'] = state.get('messages', []) + [
                f"Found {len(results)} relevant papers from PubMed"
            ]
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  PubMed search error: {e}")
            state['pubmed_results'] = []
            state['messages'] = state.get('messages', []) + [
                f"PubMed search failed: {str(e)}"
            ]
        
        return state
    
    def _synthesize_findings(self, state: ResearchState) -> ResearchState:
        """Synthesize findings from research papers using LLM"""
        
        if not state['pubmed_results']:
            state['synthesized_findings'] = "No research papers found for this query."
            state['key_papers'] = []
            return state
        
        # Prepare research context
        papers_text = []
        for i, paper in enumerate(state['pubmed_results'], 1):
            paper_summary = f"""
Paper {i}:
Title: {paper['title']}
Authors: {paper['authors']}
Date: {paper['date']}
Abstract: {paper['abstract']}
PMID: {paper['pmid']}
"""
            papers_text.append(paper_summary)
        
        context = "\n---\n".join(papers_text)
        
        # Create synthesis prompt
        prompt = f"""You are a medical research analyst. Synthesize the following recent research papers related to: "{state['query']}"

RESEARCH PAPERS:
{context}

Provide a comprehensive synthesis that includes:
1. MAIN FINDINGS: What are the key discoveries or trends across these papers?
2. CLINICAL IMPLICATIONS: What do these findings mean for clinical practice?
3. RESEARCH GAPS: What questions remain unanswered?
4. KEY PAPERS: Which 3 papers are most important and why?

Be concise but thorough. Focus on actionable insights."""

        # Get LLM response
        messages = [
            SystemMessage(content="You are an expert medical research analyst."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        synthesis = response.content
        
        # Extract top 3 papers
        key_papers = state['pubmed_results'][:3]
        
        state['synthesized_findings'] = synthesis
        state['key_papers'] = key_papers
        state['messages'] = state.get('messages', []) + [
            "Synthesized research findings"
        ]
        
        return state
    
    def _build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("search", self._search_pubmed)
        workflow.add_node("synthesize", self._synthesize_findings)
        
        # Define edges
        workflow.set_entry_point("search")
        workflow.add_edge("search", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def research(self, query: str) -> dict:
        """Main research method"""
        initial_state = {
            "query": query,
            "pubmed_results": [],
            "synthesized_findings": "",
            "key_papers": [],
            "messages": []
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "query": result["query"],
            "findings": result["synthesized_findings"],
            "key_papers": result["key_papers"],
            "total_papers": len(result["pubmed_results"]),
            "process_log": result["messages"]
        }
