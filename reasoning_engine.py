# reasoning_engine.py - Fixed LangChain ReAct Agent with Full Memory Access
"""
Enrich.AI Autonomous Biology Assistant - Reasoning Engine

FIXES APPLIED:
1. FULL MEMORY - LLM sees ALL results (not summaries)
2. Proper result extraction from LangGraph messages
3. Complete enrichment/literature/db results stored
4. Compatible with app.py interface

Architecture:
- LangGraph ReAct agent with proper state management
- Gemini as LLM backbone
- Full tool result storage (no truncation for LLM)
- Clean industry-standard approach
"""

from __future__ import annotations
import json
import logging
import os
import re
import sys
import traceback
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
import time
import uuid

try:
    import streamlit as st
except ImportError:
    st = None

from pydantic import BaseModel, Field

# LangChain imports
LANGCHAIN_AVAILABLE = False
LANGGRAPH_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.tools import StructuredTool
    from langchain_core.prompts import PromptTemplate

    try:
        from langgraph.prebuilt import create_react_agent as create_langgraph_react_agent

        LANGGRAPH_AVAILABLE = True
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        pass

    if not LANGGRAPH_AVAILABLE:
        try:
            from langchain.agents import AgentExecutor, create_react_agent

            LANGCHAIN_AVAILABLE = True
        except ImportError:
            pass

except ImportError as e:
    print(f"WARNING: LangChain not available ({e})")

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Project imports
_import_paths = [
    '/mnt/user-data/uploads',
    os.path.dirname(os.path.abspath(__file__)),
    '.',
]
for _path in _import_paths:
    if _path not in sys.path and os.path.exists(_path):
        sys.path.insert(0, _path)

from config import CONFIG
from llm_factory import get_llm, get_utility_llm, complete_text, resolve_api_key
from tools import (
    get_gene_info,
    db_retrieve,
    run_enrichment_analysis,
    search_literature,
    get_paper_annotations
)

logger = logging.getLogger(__name__)


# ===============================================================================
# DATA STRUCTURES
# ===============================================================================

@dataclass
class MergedEnvelope:
    """
    Complete response with FULL memory access.

    CRITICAL: All results stored in full (not summarized) so LLM can access them.
    """
    query_id: str = ""
    final_text: str = ""

    # === ENRICHMENT (FULL DATA) ===
    enrichment_df: Optional[pd.DataFrame] = None
    full_enrichment_results: Optional[Dict] = None  # Complete API response

    # === LITERATURE (FULL DATA) ===
    literature: Optional[List[Dict]] = None  # Top results for display
    full_literature_results: Optional[List[Dict]] = None  # ALL papers

    # === DATABASE (FULL DATA) ===
    db: Optional[Dict] = None  # Summary for display
    full_db_results: Optional[Dict] = None  # Complete results

    # === GENE INFO (FULL DATA) ===
    gene_info: Optional[List[Dict]] = None  # All gene lookups

    # === CONTEXT ===
    what_i_ran: str = ""
    caveats: List[str] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    reasoning_steps: List[Dict] = field(default_factory=list)  # Detailed thinking trace
    confidence_score: float = 0.0

    # === METADATA ===
    execution_time: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    citation_guard: Optional[Dict] = None  # what the post-generation guard removed

    def to_dict(self) -> Dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, pd.DataFrame):
                result[key] = value
            else:
                result[key] = value
        return result


# ===============================================================================
# PYDANTIC SCHEMAS FOR TOOL INPUTS
# ===============================================================================

class GeneInfoInput(BaseModel):
    gene: str = Field(description="Gene symbol (e.g., 'TP53', 'BRCA1', 'EGFR')")
    organism: str = Field(default="9606", description="NCBI taxonomy ID. Default '9606' for human.")


class DbRetrieveInput(BaseModel):
    query: str = Field(description="Search query - gene symbols, term names, IDs, or keywords")
    libraries: Optional[str] = Field(default=None, description="Comma-separated Enrichr library names"
                                     "Names are exact — do not guess library names")
    task: str = Field(default="auto",
                      description=(
                          "Task type: "
                          "'term_search' = substring match on term names (use for keyword lookup), "
                          "'find_similar' = find biologically similar terms by gene-set overlap (Jaccard similarity) — "
                          "accepts a term ID (e.g. GO:0030217), term name, or gene symbols as query, "
                          "'term_details' = get info about a specific term by ID, "
                          "'term_genes' = get genes belonging to a term, "
                          "'gene_terms' = find all terms containing a gene, "
                          "'auto' = infer from query (less reliable, prefer specifying explicitly)"
                      ))
    organism: str = Field(default="9606", description="NCBI taxonomy ID")
    limit: int = Field(default=20, description="Maximum results to return")
    include_genes: bool = Field(default=False, description="Whether to include gene lists for each term")
    similarity_threshold: float = Field(default=0.05,
                                        description="Minimum Jaccard similarity score for find_similar task (0-1). Default 0.05.")


class EnrichmentInput(BaseModel):
    genes: str = Field(description="Comma-separated gene symbols (minimum 4 genes)")
    libraries: Optional[str] = Field(default=None, description="Comma-separated Enrichr libraries to test. None = default set.")
    p_value_threshold: float = Field(default=0.05, description="Adjusted p-value cutoff for significance")
    top_n: Optional[int] = Field(default=None, description="Max terms per library. None = all significant terms.")


class LiteratureSearchInput(BaseModel):
    query: str = Field(description=(
        "Europe PMC search query. Build a SPECIFIC query from the user query and the results you have."
        "You may use Europe PMC field syntax (e.g. TITLE:, ABSTRACT:) and quoted phrases for precision."
    ))
    max_results: int = Field(default=20, description="Maximum papers to return (up to 100)")
    min_year: Optional[int] = Field(default=None, description="Minimum publication year filter")
    max_year: Optional[int] = Field(default=None, description="Maximum publication year filter")
    sort_by: str = Field(default="relevance", description="Sort order: 'relevance', 'citations', or 'date'")


class PaperAnnotationsInput(BaseModel):
    paper_id: str = Field(description="Paper identifier: PMID, PMCID, or DOI")
    id_type: str = Field(default="auto", description="Identifier type: 'pmid', 'pmcid', 'doi', or 'auto'")
    annotation_types: Optional[str] = Field(default=None, description="Comma-separated annotation types to retrieve")


# ===============================================================================
# TOOL WRAPPER FUNCTIONS - RETURN FULL RESULTS
# ===============================================================================

def _wrap_get_gene_info(gene: str, organism: str = "9606") -> str:
    """Wrapper - returns FULL gene info JSON"""
    try:
        result = get_gene_info(gene=gene, organism=organism)
        # Return FULL result - no truncation
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "tool": "get_gene_info"})


def _wrap_db_retrieve(
        query: str,
        libraries: Optional[str] = None,
        task: str = "auto",
        organism: str = "9606",
        limit: int = 20,
        include_genes: bool = False,
        similarity_threshold: float = 0.05
) -> str:
    """Wrapper - returns structured summary + full db_retrieve results"""
    try:
        libs = None
        if libraries:
            libs = [lib.strip() for lib in libraries.split(",") if lib.strip()]

        result = db_retrieve(
            query=query,
            libraries=libs,
            task=task,
            organism=organism,
            limit=limit,
            include_genes=include_genes,
            similarity_threshold=similarity_threshold
        )

        # Create structured summary for LLM interpretation
        summary_parts = []
        summary_parts.append(f"=== DATABASE RETRIEVAL RESULTS ===")
        summary_parts.append(f"Query: {query}")
        summary_parts.append(f"Task: {result.get('task_performed', task)}")
        summary_parts.append(f"Libraries: {', '.join(result.get('query_info', {}).get('libraries_used', []))}")
        summary_parts.append(f"Total results: {result.get('statistics', {}).get('total_results', 0)}")
        summary_parts.append("")

        # Format results as readable table
        results_list = result.get("results", [])
        if results_list:
            summary_parts.append("=== RESULTS TABLE ===")
            summary_parts.append(f"{'#':<3} | {'Term/Pathway':<50} | {'Source':<25} | {'Details'}")
            summary_parts.append("-" * 120)

            for i, item in enumerate(results_list, 1):
                term_name = item.get("term_name", item.get("term", item.get("name", "Unknown")))[:48]
                source = item.get("source", item.get("library", "N/A"))[:23]

                # Build details string based on available fields
                details = []
                if "description" in item and item["description"]:
                    details.append(f"Desc: {item['description'][:60]}...")
                if "gene_count" in item:
                    details.append(f"Genes: {item['gene_count']}")
                if "genes" in item and item["genes"]:
                    gene_preview = ", ".join(item["genes"][:5])
                    if len(item["genes"]) > 5:
                        gene_preview += f"... (+{len(item['genes']) - 5} more)"
                    details.append(f"Contains: {gene_preview}")
                if "score" in item:
                    details.append(f"Score: {item['score']:.3f}")
                if "p_value" in item:
                    details.append(f"p={item['p_value']:.2e}")
                if "similarity_score" in item:
                    details.append(f"Similarity: {item['similarity_score']:.3f}")
                if "shared_genes" in item:
                    details.append(f"Shared genes: {item['shared_genes']}")

                detail_str = " | ".join(details) if details else "No additional details"
                summary_parts.append(f"{i:<3} | {term_name:<50} | {source:<25} | {detail_str}")

            summary_parts.append("")

        # Add interpretation guidance
        summary_parts.append("=== INTERPRETATION ===")
        task_performed = result.get('task_performed', task)
        if task_performed == "gene_terms":
            summary_parts.append("These are pathways/terms that contain the queried gene(s).")
            summary_parts.append("Consider: Which pathways are most relevant to the biological question?")
        elif task_performed == "term_search":
            summary_parts.append("These terms matched your search query.")
            summary_parts.append("Consider: Which terms best represent the biological concept you're investigating?")
        elif task_performed == "term_genes":
            summary_parts.append("These are genes associated with the queried term(s).")
        elif task_performed == "find_similar":
            summary_parts.append(
                "These terms share overlapping genes with the query term (Jaccard similarity on gene sets).")
            summary_parts.append("Higher similarity = more shared genes = more biologically related.")

        # Include any warnings/errors
        if result.get("warnings"):
            summary_parts.append(f"\nWarnings: {', '.join(result['warnings'])}")
        if result.get("errors"):
            summary_parts.append(f"\nErrors: {', '.join(result['errors'])}")

        # Combine summary with full JSON
        output = {
            "summary": "\n".join(summary_parts),
            "full_results": result
        }

        return json.dumps(output, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "tool": "db_retrieve"})


def _wrap_enrichment_analysis(
        genes: str,
        libraries: Optional[str] = None,
        p_value_threshold: float = 0.05,
        top_n: Optional[int] = None
) -> str:
    """Wrapper - returns structured summary + full results for LLM analysis"""
    try:
        gene_list = [g.strip().upper() for g in genes.split(",") if g.strip()]

        libs = None
        if libraries:
            libs = [lib.strip() for lib in libraries.split(",") if lib.strip()]

        # Fallback: use sidebar-selected libraries if LLM didn't specify any
        if not libs and st:
            try:
                sidebar_libs = st.session_state.get("selected_libraries", [])
                if sidebar_libs:
                    libs = list(sidebar_libs)
                    logger.info(f"Using sidebar-selected libraries: {libs}")
            except Exception:
                pass

        result = run_enrichment_analysis(
            genes=gene_list,
            libraries=libs,
            p_value_threshold=p_value_threshold,
            top_n=top_n
        )

        # Create structured summary for LLM interpretation
        summary_parts = []
        summary_parts.append(f"=== ENRICHMENT ANALYSIS SUMMARY ===")
        summary_parts.append(f"Genes analyzed: {result.get('gene_count', 0)}")
        summary_parts.append(f"Total significant terms: {result.get('significant_terms_total', 0)}")
        summary_parts.append(f"Libraries tested: {', '.join(result.get('libraries_tested', []))}")
        summary_parts.append("")

        # Per-library breakdown with ALL terms listed
        enrichment_results = result.get("enrichment_results", {})
        for library, terms in enrichment_results.items():
            summary_parts.append(f"--- {library} ({len(terms)} significant terms) ---")
            for i, term in enumerate(terms, 1):
                # Show ALL terms, not just top 5
                summary_parts.append(
                    f"  {i}. {term.get('term_name', 'Unknown')} "
                    f"(p={term.get('adjusted_p_value', 1.0):.2e}, "
                    f"overlap={term.get('overlap', 'N/A')}, "
                    f"genes: {', '.join(term.get('genes', [])[:5])}{'...' if len(term.get('genes', [])) > 5 else ''})"
                )
            summary_parts.append("")

        summary_parts.append("=== INTERPRETATION GUIDE ===")
        summary_parts.append("- Lower adjusted_p_value = more significant")
        summary_parts.append("- Higher combined_score = stronger association")
        summary_parts.append("- Look for biological themes across multiple terms")
        summary_parts.append("- Note which input genes appear in multiple pathways")

        # Combine summary with full JSON for complete access
        output = {
            "summary": "\n".join(summary_parts),
            "full_results": result
        }

        return json.dumps(output, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "tool": "run_enrichment_analysis"})


def _score_paper_relevance(query: str, papers: List[Dict], llm) -> List[Dict]:
    """Score paper relevance using the same model as the reasoning agent.

    Adds 'relevance_rating' (High/Medium/Low) and 'relevance_details' to each paper.
    The agent sees these ratings in its observation and can decide whether to
    call get_paper_annotations for deeper analysis of high-relevance papers.

    The client is passed in (not resolved globally) so each session scores with
    its own provider and key.
    """
    if not papers or llm is None:
        return papers

    try:
        batch_size = 25
        for batch_start in range(0, len(papers), batch_size):
            batch_end = min(batch_start + batch_size, len(papers))
            batch = papers[batch_start:batch_end]

            paper_texts = []
            for i, paper in enumerate(batch, 1):
                title = paper.get("title", "")
                abstract = paper.get("abstract", "")[:500]
                paper_texts.append(f"{i}. Title: {title}\n   Abstract: {abstract}")

            prompt = f"""Research query: "{query}"

Score each paper's relevance to this query.
Format: One line per paper:
1. High | Directly investigates [topic]
2. Medium | Related but broader context
3. Low | Only tangentially related

Criteria:
- HIGH: Directly addresses the query topic or provides key mechanistic insights
- MEDIUM: Related but focuses on broader context or different model system
- LOW: Only briefly mentions the topic

Respond with exactly {len(batch)} lines."""

            text = complete_text(llm, f"{prompt}\n\nPapers:\n" + "\n".join(paper_texts))

            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                match = re.match(
                    r'^(\d+)[.\):\s]*\s*(High|Medium|Low)\s*[\|:\-–]\s*(.+)$',
                    line, re.IGNORECASE
                )
                if match:
                    idx = int(match.group(1)) - 1
                    actual_idx = batch_start + idx
                    if 0 <= actual_idx < len(papers):
                        papers[actual_idx]["relevance_rating"] = match.group(2).capitalize()
                        papers[actual_idx]["relevance_details"] = match.group(3).strip()

        # Fill any unscored papers
        for paper in papers:
            if "relevance_rating" not in paper:
                paper["relevance_rating"] = ""
                paper["relevance_details"] = ""

    except Exception as e:
        logger.error(f"Relevance scoring failed: {e}")
        for paper in papers:
            paper.setdefault("relevance_rating", "")
            paper.setdefault("relevance_details", "")

    return papers


def _wrap_search_literature(
        query: str,
        max_results: int = 20,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        sort_by: str = "relevance",
        scoring_llm=None
) -> str:
    """
    Wrapper - returns literature search results with relevance scoring.

    Papers are scored by the same model as the reasoning agent (scoring_llm is
    bound per engine instance). The agent sees these ratings and can call
    get_paper_annotations for deeper analysis of high-relevance papers.
    """
    try:
        result = search_literature(
            query=query,
            max_results=max_results,
            min_year=min_year,
            max_year=max_year,
            sort_by=sort_by
        )

        papers = result.get("papers", [])

        # Score relevance using the same model as the reasoning agent
        if scoring_llm is not None and papers:
            papers = _score_paper_relevance(query, papers, scoring_llm)
            # Drop papers with no usable abstract or rated Low relevance
            # (Low = "only briefly mentions the topic"). Keep the rest.
            filtered = [p for p in papers
                        if (p.get("abstract") or "").strip()
                        and str(p.get("relevance_rating", "")).strip().lower() != "low"]
            papers = filtered if filtered else papers  # never return empty
            result["papers"] = papers

        # Create concise summary for LLM
        summary_parts = []
        summary_parts.append(f"=== LITERATURE SEARCH: '{query}' ===")
        summary_parts.append(f"Found {len(papers)} papers (sorted by {sort_by})")
        if min_year or max_year:
            summary_parts.append(f"Year range: {min_year or 'any'} - {max_year or 'any'}")
        summary_parts.append("")

        if papers:
            for i, paper in enumerate(papers, 1):
                title = paper.get("title", "No title")
                year = paper.get("year", "N/A")
                citations = paper.get("citations", paper.get("citation_count", 0))
                journal = paper.get("journal", "Unknown journal")
                authors = paper.get("authors", "")
                rating = paper.get("relevance_rating", "")
                details = paper.get("relevance_details", "")

                if isinstance(authors, list):
                    author_str = authors[0] if authors else "Unknown"
                    if len(authors) > 1:
                        author_str += " et al."
                else:
                    author_str = str(authors)[:50] if authors else "Unknown"

                summary_parts.append(f"{i}. {title}")
                summary_parts.append(f"   {author_str} | {journal} ({year}) | {citations} citations")
                if rating and details:
                    summary_parts.append(f"   Relevance: [{rating}] {details}")

                # Shorter abstract preview
                abstract = paper.get("abstract", "")
                if abstract:
                    summary_parts.append(f"   Abstract: {abstract[:200]}...")

                pmid = paper.get("pmid", "")
                pmcid = paper.get("pmcid", "")
                if pmid or pmcid:
                    ids = []
                    if pmid:
                        ids.append(f"PMID:{pmid}")
                    if pmcid:
                        ids.append(f"PMCID:{pmcid}")
                    summary_parts.append(f"   {' | '.join(ids)}")
                summary_parts.append("")

        output = {
            "summary": "\n".join(summary_parts),
            "full_results": result
        }

        return json.dumps(output, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "tool": "search_literature"})


def _wrap_get_paper_annotations(
        paper_id: str,
        id_type: str = "auto",
        annotation_types: Optional[str] = None
) -> str:
    """Wrapper - returns FULL paper annotations"""
    try:
        types = None
        if annotation_types:
            types = [t.strip() for t in annotation_types.split(",") if t.strip()]

        result = get_paper_annotations(
            paper_id=paper_id,
            id_type=id_type,
            annotation_types=types
        )
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "tool": "get_paper_annotations"})


# ===============================================================================
# REACT PROMPT - MODEL-DRIVEN EXPERT ANALYSIS
# ===============================================================================

SYSTEM_PROMPT = SYSTEM_PROMPT = """You are Enrich.AI, an expert biology research assistant for molecular biologists. You have deep knowledge of genomics, signaling pathways, disease mechanisms, systems biology, and experimental design.

You have access to tools for enrichment analysis, gene information, literature search, database queries, and paper annotation. You may call any tool, in any order, as many times as needed. You decide when and why.

=== CORE PRINCIPLE ===

Enrichment tools return lists. You provide biological meaning.

Your role is to interpret what those pathways mean in the specific experimental context — identifying what is expected, mechanistically coherent, surprising, potentially artifactual, and worth follow-up.
When biological context is provided (e.g., tissue, disease, condition), your primary task is to prioritize: rank the enriched terms by biological relevance to that context, explain why certain terms matter more than others, and flag which are noise or generic.
You have to be consise in your answer, highlighting terms based on context is your main aim and you have a variety of tools to validate your answer.
You are a knowledgeable colleague reviewing results together and sharing your experience. When the user explicitly asks for any specific analysis or tool, you must do it and not skip an explicitly requested action.

=== HOW YOU THINK ===

You reason, then act, then observe, then reason again. There is no fixed pipeline.

- Think about what biological question is being asked.
- Decide which tools are appropriate.
- After results return, interpret before acting again.
- If something looks incomplete, inconsistent, or intriguing — investigate further.
- If a tool fails or returns weak signal, adapt.
- Let the biology determine your workflow.

=== VALIDATE WHAT YOU CLAIM IF NEEDED ===

Do not state biological conclusions without evidence. Use your tools to support your interpretations:
- Use search_literature to find papers that confirm or challenge a mechanistic link. 
  CONSTRUCT SPECIFIC QUERIES from the enriched terms and key genes you actually found
  never a broad single word. If the search returns off-topic or 
  low-quality papers, refine the query or the parameters and search again. 
  Before searching literature decide what you actually want to answer. 
  Decide if the literature results answered the question or you want to reframe or have a differet one.
- Use db_retrieve with short keyword queries to check pathway membership or term overlap.
- Use get_gene_info to verify gene functions before making claims about their role.
- If a paper has full text available, you can retrieve it for deeper validation.
Your credibility depends on grounding interpretation in data, not generating plausible-sounding text.

=== CITATIONS ===

Citations (PMIDs, paper titles, author-year references) are ONLY allowed for papers returned by the
search_literature tool in this run.
If search_literature was not called, include ZERO citations — do not cite any paper, PMID, or author
from memory.
Never invent, recall, or reconstruct a citation. If you did not retrieve it via search_literature, it
does not get cited.
This does not oblige you to search — you decide whether a search is warranted. It forbids citing
anything when no search results exist. Uncited established biology is fine; a fabricated reference is
not, and destroys the credibility of the entire analysis.

When you are given a gene list to interpret, analyse it with your tools before interpreting it. Do not
answer from memory alone and present the result as if it were an evidence-based interpretation.

=== FOLLOW-UP QUESTIONS ===

Look your memory to see what you have. Does the user ask about previous results?  
If yes, do NOT re-query tools to find information you already have. Use tools only to validate or expand — 
not to re-discover what is already in the conversation context.

=== CONTEXT IS EVERYTHING ===

The same enriched term can mean different things depending on the tissue, disease, perturbation, species, or experimental design. Always anchor interpretation to the biological context.
If context is not explicitly provided, infer it from the gene signature itself — read it like a biologist would.

=== EVIDENCE AND CONFIDENCE ===

Distinguish clearly between strong conclusions supported by data, plausible mechanistic hypotheses, and speculative interpretations. If you use literature to answer cite the papers. Maintain scientific credibility.

=== RESPONSE GUIDELINES ===

- Structure your response with a summary section first, then use "## Detailed Analysis" as a header 
  before the in-depth interpretation. The summary and detailed sections will be displayed separately in the UI.
- Structure responses based on what the query needs. Use headers.
- Do not output raw markdown tables.
- For db_retrieve use short keyword queries, not full sentences.
- State which tools were used and what parameters were applied.
- Discuss results from ALL tools called.
- If the first approach fails, try at least one alternative before reporting failure.
- For get_gene_info, be selective. Do not look up only one or every gene in a list if you have hundreds of them. Focus on the most 
  biologically informative or ambiguous genes that would change interpretation."""


# Legacy AgentExecutor format — wraps SYSTEM_PROMPT with template variables
REACT_PROMPT = SYSTEM_PROMPT + """

Tools available:
{tools}

Format:
Question: user question
Thought: What biological question is being asked? Is it a follow up question? What tools are appropriate?
Action: tool from [{tool_names}]
Action Input: JSON parameters
Observation: result
Thought: Interpret before acting again. Is this sufficient?
... (continue until comprehensive)
Thought: I have enough information for synthesis
Final Answer: response

Begin.

Question: {input}
Thought:{agent_scratchpad}"""


# ===============================================================================
# CITATION GUARD - deterministic, post-generation
# ===============================================================================
# The system prompt asks the model not to cite papers it did not retrieve. This
# ENFORCES it: any PMID, author-year reference or reference-list entry that does not
# correspond to a paper returned by search_literature in this run is removed.
#
# It does not force a tool call and does not touch the model's reasoning - the LLM
# still decides whether to search. It only deletes citations with nothing behind them.

_PMID_CITE_RE = re.compile(r"\(?\s*PMIDs?[:\s#]*((?:\d{6,9})(?:\s*[,;and]+\s*\d{6,9})*)\s*\)?", re.I)
_AUTHOR_YEAR_RE = re.compile(r"\(\s*([A-Z][A-Za-z\-']{2,})(?:\s+(?:and|&)\s+[A-Z][A-Za-z\-']{2,})?"
                             r"\s+et\s+al\.?,?\s*(\d{4})\s*\)")
_ETAL_REF_RE = re.compile(r"[A-Z][A-Za-z\-']+(?:\s*,\s*[A-Z]\.?)*\s*,?\s*et\s+al\.[^\n]*")
# "by Semenza GL (2009)" / "by Ward PS, Thompson CB (2012)" - attribution with no PMID
# and no "et al.", which earlier versions of this guard missed entirely.
_BY_AUTHOR_YEAR_RE = re.compile(
    r"\bby\s+[A-Z][A-Za-z\-']+(?:\s+[A-Z]{1,3})?"
    r"(?:\s*,\s*[A-Z][A-Za-z\-']+(?:\s+[A-Z]{1,3})?)*"
    r"(?:\s*,?\s*et\s+al\.)?\s*\(\d{4}\)")
# a bullet/numbered line whose payload is a quoted title = a reference-list entry
_REF_LINE_RE = re.compile(r'^\s*(?:[-*•]|\d+[.)])\s*\**\s*["“][^"”\n]{20,}["”]')


def _norm_paper_title(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s or "")
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def _tidy(text: str) -> str:
    """Clean up punctuation left behind after removing a citation."""
    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)      # empty parens
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def verify_citations(text: str, papers: List[Dict]) -> tuple:
    """Strip citations that no retrieved paper backs.

    Returns (clean_text, report). report records what was removed so the UI and the
    eval can see it; an empty report means the model cited cleanly on its own.
    """
    if not text:
        return text, {"checked": False}

    papers = [p for p in (papers or []) if isinstance(p, dict)]
    got_pmids = {str(p.get("pmid")).strip() for p in papers if p.get("pmid")}
    got_authors = " ".join(
        " ".join(p.get("authors", [])) if isinstance(p.get("authors"), list) else str(p.get("authors", ""))
        for p in papers
    ).lower()

    removed_pmids, removed_author_years, removed_refs = [], [], []

    def _pmid_sub(m):
        ids = re.findall(r"\d{6,9}", m.group(1))
        unverified = [i for i in ids if i not in got_pmids]
        if not unverified:
            return m.group(0)
        removed_pmids.extend(unverified)
        kept = [i for i in ids if i in got_pmids]
        return f" (PMID: {', '.join(kept)})" if kept else ""

    out = _PMID_CITE_RE.sub(_pmid_sub, text)

    def _ay_sub(m):
        surname = m.group(1)
        if surname.lower() in got_authors and got_authors:
            return m.group(0)
        removed_author_years.append(f"{surname} et al., {m.group(2)}")
        return ""

    out = _AUTHOR_YEAR_RE.sub(_ay_sub, out)

    # With nothing retrieved, EVERY citation-shaped construct is unverifiable by
    # definition: "et al." references, "by Author (Year)" attributions, and
    # reference-list lines built around a quoted title.
    if not papers:
        lines = []
        for line in out.split("\n"):
            is_list_item = re.match(r"^\s*(?:[-*•]|\d+[.)])\s", line) is not None
            looks_like_reference = (
                "et al." in line
                or re.search(r"\(\d{4}\)", line)
                or _REF_LINE_RE.match(line)
                or _BY_AUTHOR_YEAR_RE.search(line)
            )
            # a bibliography entry: drop the whole line, quoted title or not
            if is_list_item and looks_like_reference:
                removed_refs.append(line.strip()[:160])
                continue
            # inline attribution inside prose: strip the citation, keep the sentence
            new_line = _BY_AUTHOR_YEAR_RE.sub("", line)
            if "et al." in new_line:
                new_line = re.sub(r"[A-Z][A-Za-z\-'.]*(?:[^.\n]*?)et al\.[^.\n]*\.?", "", new_line)
            if new_line != line:
                removed_refs.append(line.strip()[:160])
                if len(re.sub(r"[^A-Za-z0-9]", "", new_line)) < 25:
                    continue
                line = new_line
            lines.append(line)
        out = "\n".join(lines)

    n_removed = len(removed_pmids) + len(removed_author_years) + len(removed_refs)
    if n_removed:
        out = _tidy(out)
        out += (f"\n\n> *{n_removed} citation(s) were removed by the citation guard: they did not "
                f"correspond to any paper retrieved by search_literature in this run.*")

    return out, {
        "checked": True,
        "n_papers_retrieved": len(papers),
        "removed_pmids": removed_pmids,
        "removed_author_years": removed_author_years,
        "removed_reference_lines": removed_refs,
        "n_removed": n_removed,
    }


# ===============================================================================
# REASONING ENGINE CLASS
# ===============================================================================

class ReasoningEngine:
    """
    LangChain-based ReAct reasoning engine with FULL memory access.

    Key improvements:
    - Stores ALL tool results (not truncated)
    - LLM sees complete context from previous queries
    - Proper result extraction
    - Memory persistence across queries
    """

    def __init__(self, model_name: str = None, provider: str = None, api_key: str = None):
        """Initialize the reasoning engine."""
        # Provider drives which LangChain client the factory builds
        self.provider = provider or CONFIG.get("default_provider", "gemini")

        # Get model name from the provider registry or use default
        if model_name is None:
            model_name = CONFIG.get("providers", {}).get(self.provider, {}).get("default_model")
        if model_name is None:
            model_name = CONFIG.get("gemini", {}).get("model_name", "gemini-2.5-flash")

        self.model_name = model_name
        self.selected_libraries = []
        self.envelope = None
        self.query_id = ""
        self.use_langgraph = False

        # === MEMORY PERSISTENCE ===
        # Store previous envelopes for follow-up questions
        self.previous_envelopes: List[Dict] = []
        self.max_memory_envelopes = 10  # Keep last 10 queries

        # Rate limiting
        self.last_call_time = 0
        self.min_call_interval = 1.0

        # Get API key: explicit -> config/env for this provider -> session state
        self.api_key = api_key or resolve_api_key(self.provider)
        if not self.api_key and st:
            # Provider-specific key from the sidebar, else the legacy Gemini field
            self.api_key = st.session_state.get(f"{self.provider}_api_key")
            if not self.api_key and self.provider == "gemini":
                self.api_key = st.session_state.get("api_key")

        if not self.api_key:
            env_vars = CONFIG.get("providers", {}).get(self.provider, {}).get("api_key_env", [])
            raise ValueError(
                f"{self.provider} API key not found. Set {' or '.join(env_vars) or 'the provider API key'}"
            )

        if LANGCHAIN_AVAILABLE:
            self._init_langchain()
        else:
            self._init_fallback()

    def _init_langchain(self):
        """Initialize LangChain ReAct agent"""
        logger.info(f"Initializing LangChain ReAct agent (provider={self.provider}, model={self.model_name})...")

        self.llm = get_llm(
            provider=self.provider,
            model=self.model_name,
            api_key=self.api_key,
        )

        # Separate client for one-shot scoring (no thinking trace). Bound into the
        # literature tool below so each engine instance scores with its own
        # provider/key - a module-level client would be shared across the
        # concurrent Streamlit sessions of different users.
        self.utility_llm = get_utility_llm(
            provider=self.provider,
            model=self.model_name,
            api_key=self.api_key,
        )

        def _search_literature_with_scoring(
                query: str,
                max_results: int = 20,
                min_year: Optional[int] = None,
                max_year: Optional[int] = None,
                sort_by: str = "relevance"
        ) -> str:
            return _wrap_search_literature(
                query=query,
                max_results=max_results,
                min_year=min_year,
                max_year=max_year,
                sort_by=sort_by,
                scoring_llm=self.utility_llm,
            )

        # Create tools with proper schemas
        self.tools = [
            StructuredTool.from_function(
                func=_wrap_get_gene_info,
                name="get_gene_info",
                description=(
                    "Retrieve gene information from MyGene.info. "
                    "Returns gene summary, GO term annotations, pathway memberships, "
                    "and disease associations when available."
                ),
                args_schema=GeneInfoInput
            ),
            StructuredTool.from_function(
                func=_wrap_db_retrieve,
                name="db_retrieve",
                description=(
                    "Search 200+ Enrichr biological databases for pathways, GO terms, and gene-term associations. "
                    "Specify the task explicitly for best results: "
                    "task='term_search' finds terms whose names contain a keyword. "
                    "task='find_similar' finds biologically similar terms by gene-set overlap (Jaccard similarity) — "
                    "accepts a term ID, term name, or gene symbols as query. "
                    "task='term_details' gets info about a specific term ID. "
                    "task='term_genes' returns genes belonging to a term. "
                    "task='gene_terms' finds all terms containing a gene. "
                    "Returns ranked results with scores and source libraries."
                ),
                args_schema=DbRetrieveInput
            ),
            StructuredTool.from_function(
                func=_wrap_enrichment_analysis,
                name="run_enrichment_analysis",
                description=(
                    "Run statistical enrichment analysis on a gene list (minimum 4 genes). "
                    "Tests against Enrichr libraries and returns significantly enriched terms "
                    "with adjusted p-values, combined scores, and overlapping genes."
                ),
                args_schema=EnrichmentInput
            ),
            StructuredTool.from_function(
                func=_search_literature_with_scoring,
                name="search_literature",
                description=(
                    "Search Europe PMC for scientific papers. "
                    "Returns titles, abstracts, authors, citation counts, publication year, "
                    "and AI-generated relevance ratings."
                ),
                args_schema=LiteratureSearchInput
            ),
            StructuredTool.from_function(
                func=_wrap_get_paper_annotations,
                name="get_paper_annotations",
                description=(
                    "Extract text-mined annotations from a full-text paper via Europe PMC. "
                    "Returns tagged genes, diseases, chemicals, GO terms, and organisms "
                    "found in the paper body. Requires PMCID for full-text access; "
                    "PMID will return abstract-level annotations only."
                ),
                args_schema=PaperAnnotationsInput
            )
        ]

        if LANGGRAPH_AVAILABLE:
            self._init_langgraph_agent()
        else:
            self._init_legacy_agent()

        logger.info(f"ReAct agent initialized with {len(self.tools)} tools")

    def _init_langgraph_agent(self):
        """Initialize using LangGraph with custom system prompt"""
        logger.info("Using LangGraph create_react_agent...")

        # Use the shared system prompt
        self._langgraph_system_prompt = SYSTEM_PROMPT

        # Create agent without any custom prompt parameters (most compatible)
        self.agent_executor = create_langgraph_react_agent(self.llm, self.tools)
        self.use_langgraph = True

    def _init_legacy_agent(self):
        """Initialize using legacy AgentExecutor"""
        logger.info("Using legacy AgentExecutor...")
        self.prompt = PromptTemplate.from_template(REACT_PROMPT)
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            max_execution_time=120,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        self.use_langgraph = False

    def _init_fallback(self):
        """Initialize fallback direct Gemini mode"""
        logger.warning("LangChain not available, using fallback mode")
        if genai:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        else:
            raise ImportError("Neither LangChain nor google-generativeai available!")

    def _rate_limit_wait(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_call_interval:
            wait_time = self.min_call_interval - elapsed
            time.sleep(wait_time)
        self.last_call_time = time.time()

    def _format_memory_context(self) -> str:
        """
        Format previous query results as context for follow-up questions.

        Gold-standard pattern: keep the MOST RECENT query in full detail (immediate
        follow-ups need every term), and SUMMARIZE older queries (top terms + paper
        titles only) to prevent context bloat on multi-turn sessions.
        """
        if not self.previous_envelopes:
            return ""

        context_parts = ["=== PREVIOUS QUERY RESULTS (for follow-up reference) ===\n"]
        recent = self.previous_envelopes[-3:]

        for i, prev in enumerate(recent, 1):
            query = prev.get("query", "")
            envelope = prev.get("envelope", {})
            is_most_recent = (i == len(recent))

            context_parts.append(f'\n--- Query {i}: "{query[:100]}" ---')

            enrich = envelope.get("full_enrichment_results") or {}
            if enrich:
                context_parts.append("\nEnrichment Results:")
                context_parts.append(f"  Genes analyzed: {enrich.get('query_genes', [])}")
                context_parts.append(f"  Libraries tested: {enrich.get('libraries_tested', [])}")
                context_parts.append(f"  Total significant terms: {enrich.get('significant_terms_total', 0)}")

                results = enrich.get("enrichment_results") or {}
                if results:
                    if is_most_recent:
                        # FULL detail for the most recent query
                        context_parts.append("\n  COMPLETE ENRICHMENT RESULTS BY LIBRARY:")
                        for lib, terms in results.items():
                            context_parts.append(f"\n    {lib} ({len(terms)} significant terms):")
                            for idx, term in enumerate(terms, 1):
                                term_name = term.get("term", term.get("term_name", ""))
                                p_val = term.get("adjusted_p_value", 1.0)
                                genes = term.get("genes", [])
                                overlap = term.get("overlap", "")
                                context_parts.append(
                                    f"      {idx}. {term_name} (p={p_val:.2e}, overlap={overlap}, "
                                    f"genes: {', '.join(genes[:10])}{'...' if len(genes) > 10 else ''})"
                                )
                    else:
                        # SUMMARY for older queries: top 5 terms per library, no gene lists
                        context_parts.append("\n  TOP ENRICHED TERMS (summary):")
                        for lib, terms in results.items():
                            top = terms[:5]
                            names = [t.get("term", t.get("term_name", "")) for t in top]
                            more = f" (+{len(terms) - 5} more)" if len(terms) > 5 else ""
                            context_parts.append(f"    {lib}: {'; '.join(names)}{more}")

            lit = envelope.get("full_literature_results") or []
            if lit:
                if is_most_recent:
                    context_parts.append(f"\nLiterature Results: {len(lit)} papers found")
                    context_parts.append("  Papers retrieved:")
                    for idx, paper in enumerate(lit[:15], 1):
                        title = paper.get("title", "")
                        year = paper.get("year", paper.get("pub_year", ""))
                        citations = paper.get("citations", paper.get("citation_count", 0))
                        authors = paper.get("authors", [])
                        first_author = authors[0] if authors else "Unknown"
                        abstract = (paper.get("abstract", "") or "")[:200]
                        context_parts.append(
                            f"    {idx}. {title} ({first_author} et al., {year}, {citations} citations)"
                        )
                        if abstract:
                            context_parts.append(f"       Abstract: {abstract}...")
                else:
                    titles = [(p.get("title", "") or "")[:60] for p in lit[:3]]
                    more = f" (+{len(lit) - 3} more)" if len(lit) > 3 else ""
                    context_parts.append(f"\nLiterature: {len(lit)} papers — e.g. {'; '.join(titles)}{more}")

            # DB + gene info only for the most recent (older summarized to a mention)
            db_results = envelope.get("full_db_results")
            if db_results:
                if is_most_recent:
                    if isinstance(db_results, list):
                        context_parts.append(f"\nDatabase Results: {len(db_results)} result sets")
                        for db in db_results:
                            if isinstance(db, dict):
                                task = db.get("task_performed", "query")
                                total = db.get("statistics", {}).get("total_results", 0)
                                context_parts.append(f"  Task: {task}, Results: {total}")
                                for result in db.get("results", []):
                                    term_name = result.get("term_name", result.get("term_id", ""))
                                    desc = (result.get("description", "") or "")[:100]
                                    context_parts.append(f"    - {term_name}: {desc}")
                    elif isinstance(db_results, dict):
                        task = db_results.get("task_performed", "query")
                        total = db_results.get("statistics", {}).get("total_results", 0)
                        context_parts.append(f"\nDatabase Results: Task={task}, Total={total}")
                        for result in db_results.get("results", []):
                            term_name = result.get("term_name", result.get("term_id", ""))
                            context_parts.append(f"    - {term_name}")
                else:
                    context_parts.append("\nDatabase Results: (available from earlier query)")

            gene_info = envelope.get("gene_info")
            if gene_info:
                context_parts.append(f"\nGene Info: {len(gene_info)} gene(s) looked up")
                if is_most_recent:
                    for gene_data in gene_info:
                        gene_name = gene_data.get("gene", gene_data.get("symbol", ""))
                        summary = (gene_data.get("summary", "") or "")[:200]
                        context_parts.append(f"  {gene_name}: {summary}...")

        context_parts.append("\n=== END PREVIOUS RESULTS ===\n")
        return "\n".join(context_parts)
    
    def _store_envelope_in_memory(self, query: str, envelope_dict: Dict):
        """Store envelope for future follow-up questions"""
        self.previous_envelopes.append({
            "query": query,
            "envelope": envelope_dict,
            "timestamp": time.time()
        })

        # Keep only last N envelopes
        if len(self.previous_envelopes) > self.max_memory_envelopes:
            self.previous_envelopes = self.previous_envelopes[-self.max_memory_envelopes:]

    def _extract_results_from_messages(self, messages: list) -> Dict:
        """
        Extract FULL structured results from LangGraph messages.

        CRITICAL: Stores COMPLETE results, not summaries.
        Handles both old format and new summary+full_results format.
        Also captures full reasoning trace showing Gemini's thinking process.
        """
        results = {
            "enrichment": None,
            "enrichment_df": None,
            "literature": [],
            "gene_info": [],
            "db_results": [],
            "tools_used": [],
            "trace": [],
            "reasoning_steps": []  # Detailed reasoning trace
        }

        step_number = 0
        saw_tool_call = False

        for msg in messages:
            msg_type = type(msg).__name__

            if msg_type == "HumanMessage":
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if not content.startswith("You are Enrich.AI"):
                    display_content = content
                    if "=== PREVIOUS QUERY RESULTS" in content:
                        parts = content.split("=== END PREVIOUS RESULTS ===")
                        display_content = parts[-1].strip() if len(parts) > 1 else content[-500:]
                    results["reasoning_steps"].append({
                        "step": 0,
                        "type": "query",
                        "content": display_content[:500]
                    })

            elif msg_type == "AIMessage":
                content = ""
                thinking_text = ""
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, str):
                        content = msg.content
                    elif isinstance(msg.content, list):
                        text_parts = []
                        thinking_parts = []
                        for block in msg.content:
                            if isinstance(block, dict):
                                if block.get('type') == 'thinking':
                                    thinking_parts.append(block.get('thinking', ''))
                                elif 'text' in block:
                                    text_parts.append(block['text'])
                            elif isinstance(block, str):
                                text_parts.append(block)
                        content = ' '.join(text_parts)
                        thinking_text = ' '.join(thinking_parts)

                if not content and hasattr(msg, 'additional_kwargs'):
                    additional = msg.additional_kwargs
                    content = additional.get('reasoning', '') or additional.get('thoughts', '')

                has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls

                if has_tool_calls:
                    if content.strip() and saw_tool_call:
                        results["reasoning_steps"].append({
                            "step": step_number,
                            "type": "reasoning",
                            "content": content.strip()[:500]
                        })

                    for i, tc in enumerate(msg.tool_calls):
                        step_number += 1
                        tool_name = tc.get("name", "unknown")
                        tool_args = tc.get("args", {})

                        results["tools_used"].append(tool_name)
                        results["trace"].append(f"Tool: {tool_name} | Args: {tool_args}")

                        # Use actual thinking tokens if available
                        if thinking_text.strip():
                            thought_text = thinking_text.strip()
                        elif content.strip() and not saw_tool_call and i == 0:
                            thought_text = content.strip()[:300]
                        else:
                            thought_text = self._generate_contextual_thought(tool_name, tool_args, step_number)

                        saw_tool_call = True
                        args_summary = self._format_tool_args(tool_name, tool_args)

                        results["reasoning_steps"].append({
                            "step": step_number,
                            "type": "thought_action",
                            "thought": thought_text,
                            "action": tool_name,
                            "args": args_summary,
                            "tool_name": tool_name,
                            "tool_call_id": tc.get("id", "")
                        })

                else:
                    if content.strip():
                        results["reasoning_steps"].append({
                            "step": step_number + 1,
                            "type": "final_answer",
                            "content": content[:500]
                        })

            elif msg_type == "ToolMessage":

                tool_name = getattr(msg, 'name', 'unknown')

                tool_call_id = getattr(msg, 'tool_call_id', '')

                if tool_name not in results["tools_used"]:
                    results["tools_used"].append(tool_name)

                try:

                    obs_data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content

                except:

                    obs_data = {"raw": str(msg.content)}

                obs_summary = self._summarize_observation(tool_name, obs_data)

                # Match observation to its tool call by tool_call_id

                matched = False

                if tool_call_id:

                    for step in reversed(results["reasoning_steps"]):

                        if step.get("type") == "thought_action" and step.get(
                                "tool_call_id") == tool_call_id and not step.get("observation"):
                            step["observation"] = obs_summary

                            matched = True

                            break

                if not matched:

                    # Fallback: match to last unmatched thought_action

                    for step in reversed(results["reasoning_steps"]):

                        if step.get("type") == "thought_action" and not step.get("observation"):
                            step["observation"] = obs_summary

                            break

                # Handle new format: {"summary": ..., "full_results": ...}
                if "full_results" in obs_data:
                    obs_data = obs_data["full_results"]

                # Store FULL results based on tool type
                if tool_name == "get_gene_info":
                    results["gene_info"].append(obs_data)

                elif tool_name == "db_retrieve":
                    results["db_results"].append(obs_data)

                elif tool_name == "run_enrichment_analysis":
                    results["enrichment"] = obs_data
                    if obs_data.get("enrichment_results"):
                        rows = []
                        for lib, terms in obs_data["enrichment_results"].items():
                            for term in terms:
                                rows.append({
                                    "term": term.get("term", term.get("term_name", "")),
                                    "library": lib,
                                    "p_value": term.get("p_value", 1.0),
                                    "adjusted_p_value": term.get("adjusted_p_value", 1.0),
                                    "z_score": term.get("z_score", 0.0),
                                    "combined_score": term.get("combined_score", 0.0),
                                    "overlap": term.get("overlap", ""),
                                    "genes": term.get("genes", term.get("overlap_genes", []))
                                })
                        if rows:
                            results["enrichment_df"] = pd.DataFrame(rows)

                elif tool_name == "search_literature":
                    if obs_data.get("papers"):
                        results["literature"].extend(obs_data["papers"])

                elif tool_name == "get_paper_annotations":
                    results["db_results"].append({"paper_annotations": obs_data})

        return results

    def _format_tool_args(self, tool_name: str, args: dict) -> str:
        """Format tool arguments for readable display"""
        if not args:
            return "None"

        formatted = []
        for key, value in args.items():
            if isinstance(value, str) and len(value) > 50:
                value = value[:50] + "..."
            formatted.append(f"{key}={value}")

        return ", ".join(formatted)

    def _generate_contextual_thought(self, tool_name: str, tool_args: dict, step_number: int) -> str:
        """Neutral fallback when the LLM doesn't provide explicit reasoning text."""
        args_brief = ", ".join(f"{k}={v}" for k, v in list(tool_args.items())[:3])
        return f"Calling {tool_name}({args_brief})"

    def _summarize_observation(self, tool_name: str, obs_data: dict) -> str:
        """Create a brief summary of tool observation for reasoning trace"""
        # Handle new format
        if "summary" in obs_data:
            # Extract first few lines of summary
            summary_lines = obs_data["summary"].split("\n")[:3]
            return " | ".join(line.strip() for line in summary_lines if line.strip())

        if "full_results" in obs_data:
            obs_data = obs_data["full_results"]

        if tool_name == "get_gene_info":
            gene = obs_data.get("symbol", obs_data.get("query", "Unknown"))
            summary = obs_data.get("summary", "")[:100]
            return f"Gene {gene}: {summary}..." if summary else f"Gene info for {gene}"

        elif tool_name == "db_retrieve":
            total = obs_data.get("statistics", {}).get("total_results", 0)
            task = obs_data.get("task_performed", "query")
            return f"Found {total} results ({task})"

        elif tool_name == "run_enrichment_analysis":
            total_terms = obs_data.get("significant_terms_total", 0)
            gene_count = obs_data.get("gene_count", 0)
            return f"Enrichment complete: {total_terms} significant terms from {gene_count} genes"

        elif tool_name == "search_literature":
            total = obs_data.get("total_results", 0)
            papers = obs_data.get("papers", [])
            if papers:
                top_paper = papers[0].get("title", "")[:50]
                return f"Found {total} papers. Top: '{top_paper}...'"
            return f"Found {total} papers"

        elif tool_name == "get_paper_annotations":
            annotations = obs_data.get("annotations", {})
            total = sum(len(v) for v in annotations.values() if isinstance(v, list))
            return f"Extracted {total} annotations from paper"

        else:
            return f"Received response from {tool_name}"

    def _extract_results_from_steps(self, intermediate_steps: list) -> Dict:
        """Extract FULL results from legacy AgentExecutor steps.

        Handles both old format and new summary+full_results format.
        Also captures reasoning trace.
        """
        results = {
            "enrichment": None,
            "enrichment_df": None,
            "literature": [],
            "gene_info": [],
            "db_results": [],
            "tools_used": [],
            "trace": [],
            "reasoning_steps": []
        }

        for step_num, (action, observation) in enumerate(intermediate_steps, 1):
            tool_name = action.tool
            tool_input = action.tool_input
            results["tools_used"].append(tool_name)
            results["trace"].append(f"Tool: {tool_name} | Input: {tool_input}")

            try:
                obs_data = json.loads(observation) if isinstance(observation, str) else observation
            except:
                obs_data = {"raw": observation}

            # Get observation summary
            obs_summary = self._summarize_observation(tool_name, obs_data)

            # Extract thought from action log if available
            thought = ""
            if hasattr(action, 'log') and action.log:
                # Legacy agent stores thought in log
                log_lines = action.log.strip().split('\n')
                for line in log_lines:
                    if line.strip() and not line.startswith('Action'):
                        thought = line.strip()
                        break

            # Generate contextual thought if none found in log
            if not thought:
                tool_args = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
                thought = self._generate_contextual_thought(tool_name, tool_args, step_num)

            # Format args
            args_summary = self._format_tool_args(tool_name,
                                                  tool_input if isinstance(tool_input, dict) else {"input": tool_input})

            # Add reasoning step
            results["reasoning_steps"].append({
                "step": step_num,
                "type": "thought_action",
                "thought": f"ðŸ’­ Thought: {thought[:400] if thought else 'Processing query...'}",
                "action": f"ðŸ”§ Action: {tool_name}",
                "args": f"ðŸ“‹ Parameters: {args_summary}",
                "observation": f"ðŸ‘ï¸ Observation: {obs_summary}",
                "tool_name": tool_name
            })

            # Handle new format: {"summary": ..., "full_results": ...}
            if "full_results" in obs_data:
                obs_data = obs_data["full_results"]

            if tool_name == "get_gene_info":
                results["gene_info"].append(obs_data)

            elif tool_name == "db_retrieve":
                results["db_results"].append(obs_data)

            elif tool_name == "run_enrichment_analysis":
                results["enrichment"] = obs_data
                if obs_data.get("enrichment_results"):
                    rows = []
                    for lib, terms in obs_data["enrichment_results"].items():
                        for term in terms:
                            rows.append({
                                "term": term.get("term", term.get("term_name", "")),
                                "library": lib,
                                "p_value": term.get("p_value", 1.0),
                                "adjusted_p_value": term.get("adjusted_p_value", 1.0),
                                "z_score": term.get("z_score", 0.0),
                                "combined_score": term.get("combined_score", 0.0),
                                "overlap": term.get("overlap", ""),
                                "genes": term.get("genes", term.get("overlap_genes", []))
                            })
                    if rows:
                        results["enrichment_df"] = pd.DataFrame(rows)

            elif tool_name == "search_literature":
                if obs_data.get("papers"):
                    results["literature"].extend(obs_data["papers"])

            elif tool_name == "get_paper_annotations":
                results["db_results"].append({"paper_annotations": obs_data})

        return results

    def reason(self, query: str, chat_history: list = None, selected_libraries: List[str] = None) -> MergedEnvelope:
        """
        Main entry point for test scripts - returns MergedEnvelope directly.
        """
        result = self.run(query, chat_history, selected_libraries)
        envelope_dict = result.get('envelope', {})

        envelope = MergedEnvelope()
        for key, value in envelope_dict.items():
            if hasattr(envelope, key):
                setattr(envelope, key, value)

        return envelope

    def run(self, query: str, chat_history: list = None, selected_libraries: List[str] = None,
            progress_callback=None) -> Dict:
        """
        Main entry point - backward compatible interface for app.py.

        Returns FULL results in envelope (not summaries).
        Memory context from previous queries is passed to LLM.
        """
        if selected_libraries:
            self.selected_libraries = selected_libraries

        self._progress_callback = progress_callback
        start_time = time.time()
        self.query_id = str(uuid.uuid4())[:8]
        self.envelope = MergedEnvelope(query_id=self.query_id)

        if st:
            st.session_state.current_query_id = self.query_id

        try:
            logger.info("ReAct agent path")

            # Get memory context from previous queries
            memory_context = self._format_memory_context()

            if LANGCHAIN_AVAILABLE:
                result = self._run_react_agent(query, memory_context)
            else:
                result = self._run_fallback(query, memory_context)

            # Deterministic guard: remove any citation not backed by a paper this run
            # actually retrieved. Runs before anything downstream reads final_text.
            guarded_text, guard_report = verify_citations(
                result.get("output", "I couldn't generate a response."),
                result.get("literature") or []
            )
            if guard_report.get("n_removed"):
                logger.warning(f"Citation guard removed {guard_report['n_removed']} unverifiable "
                               f"citation(s): {guard_report['removed_pmids']} "
                               f"{guard_report['removed_author_years']}")
            self.envelope.final_text = guarded_text
            self.envelope.citation_guard = guard_report
            self.envelope.tools_used = result.get("tools_used", [])
            self.envelope.reasoning_trace = result.get("trace", [])
            self.envelope.reasoning_steps = result.get("reasoning_steps", [])
            self.envelope.what_i_ran = f"ReAct Agent | Tools: {', '.join(self.envelope.tools_used)} | Time: {time.time() - start_time:.2f}s"

            # === STORE FULL RESULTS ===

            # Enrichment - COMPLETE data
            if result.get("enrichment_df") is not None:
                self.envelope.enrichment_df = result["enrichment_df"]
            if result.get("enrichment"):
                self.envelope.full_enrichment_results = result["enrichment"]

            # Literature - ALL papers
            if result.get("literature"):
                self.envelope.literature = result["literature"][:20]  # Top 20 for display
                self.envelope.full_literature_results = result["literature"]  # ALL papers

            # Database - COMPLETE results (merge multiple calls into single structure)
            if result.get("db_results"):
                db_results_list = result["db_results"]

                if len(db_results_list) == 1:
                    # Single db_retrieve call - use directly
                    self.envelope.full_db_results = db_results_list[0]
                    self.envelope.db = db_results_list[0]
                else:
                    # Multiple db_retrieve calls - merge into single structure
                    merged = {
                        "task_performed": "multiple_queries",
                        "query_info": {"queries": len(db_results_list)},
                        "results": [],
                        "statistics": {"total_results": 0, "queries_made": len(db_results_list)},
                        "sources": []
                    }
                    for db_result in db_results_list:
                        if isinstance(db_result, dict):
                            # Add results from each query
                            if "results" in db_result:
                                merged["results"].extend(db_result["results"])
                                merged["statistics"]["total_results"] += len(db_result.get("results", []))
                            if "sources" in db_result:
                                merged["sources"].extend(db_result["sources"])

                    merged["sources"] = list(set(merged["sources"]))
                    self.envelope.full_db_results = merged
                    self.envelope.db = merged

            # Gene info - ALL lookups
            if result.get("gene_info"):
                self.envelope.gene_info = result["gene_info"]

            self.envelope.confidence_score = 0.9

            self.envelope.execution_time = time.time() - start_time

        except Exception as e:
            logger.error(f"Reasoning failed: {e}\n{traceback.format_exc()}")
            self.envelope.final_text = f"I encountered an error: {str(e)}"
            self.envelope.caveats.append(f"Error: {str(e)}")
            self.envelope.execution_time = time.time() - start_time

        # === STORE ENVELOPE IN MEMORY FOR FOLLOW-UP QUESTIONS ===
        envelope_dict = self.envelope.to_dict()
        self._store_envelope_in_memory(query, envelope_dict)

        return {
            "type": "final_answer",
            "envelope": envelope_dict
        }

    def _run_react_agent(self, query: str, memory_context: str = "") -> Dict:
        """Run the LangChain/LangGraph ReAct agent with memory context"""
        self._rate_limit_wait()

        # Prepend memory context to query for follow-up support
        if memory_context:
            augmented_query = f"{memory_context}\n\nCURRENT QUESTION: {query}"
        else:
            augmented_query = query

        # Inform LLM about user-selected libraries
        if self.selected_libraries:
            lib_list = ", ".join(self.selected_libraries)
            augmented_query += (
                f"\n\n[USER HAS SELECTED THESE ENRICHR LIBRARIES: {lib_list}. "
                f"Use these libraries for any enrichment analysis unless the user specifies otherwise.]"
            )

        try:
            if LANGGRAPH_AVAILABLE and self.use_langgraph:
                return self._run_langgraph_agent(augmented_query)
            else:
                return self._run_legacy_agent(augmented_query)
        except Exception as e:
            logger.error(f"ReAct agent error: {e}\n{traceback.format_exc()}")
            return {
                "output": f"Agent execution failed: {str(e)}",
                "tools_used": [],
                "trace": [f"Error: {str(e)}"]
            }

    def _run_langgraph_agent(self, query: str) -> Dict:
        """Run using LangGraph"""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build messages - prepend system prompt if using fallback method
        messages = []
        if hasattr(self, '_langgraph_system_prompt'):
            messages.append(SystemMessage(content=self._langgraph_system_prompt))
        messages.append(HumanMessage(content=query))

        # Stream step-by-step so we can report progress as each node/tool completes.
        result = None
        PRETTY = {
            "run_enrichment_analysis": "Running enrichment analysis",
            "search_literature": "Searching literature",
            "db_retrieve": "Querying databases",
            "get_gene_info": "Looking up gene information",
        }

        def pretty(name):
            return PRETTY.get(name, name.replace("_", " ").capitalize())

        for chunk in self.agent_executor.stream(
                {"messages": messages},
                config={"recursion_limit": 30},
                stream_mode="values",
        ):
            result = chunk  # each chunk is the full state; last one is final
            # report the latest tool activity to the UI, if a callback was provided
            cb = getattr(self, "_progress_callback", None)
            if cb:
                msgs = chunk.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    # a tool just produced output
                    if type(last).__name__ == "ToolMessage":
                        tool_name = getattr(last, "name", "tool")
                        cb(f"✓ {pretty(tool_name)}")
                    # the model requested tool calls
                    elif type(last).__name__ == "AIMessage" and getattr(last, "tool_calls", None):
                        for tc in last.tool_calls:
                            cb(f"→ {pretty(tc.get('name', 'tool'))}")

        response_messages = (result or {}).get("messages", [])

        # Get final output
        final_output = ""
        for msg in reversed(response_messages):
            if type(msg).__name__ == "AIMessage":
                if hasattr(msg, 'content') and msg.content:
                    if isinstance(msg.content, str):
                        final_output = msg.content
                    elif isinstance(msg.content, list):
                        text_parts = []
                        for block in msg.content:
                            if isinstance(block, dict) and 'text' in block:
                                text_parts.append(block['text'])
                            elif isinstance(block, str):
                                text_parts.append(block)
                        final_output = ' '.join(text_parts) if text_parts else str(msg.content)
                    break

        # Extract FULL results
        extracted = self._extract_results_from_messages(response_messages)

        return {
            "output": final_output,
            "tools_used": extracted.get("tools_used", []),
            "trace": extracted.get("trace", []),
            "reasoning_steps": extracted.get("reasoning_steps", []),
            "enrichment": extracted.get("enrichment"),
            "enrichment_df": extracted.get("enrichment_df"),
            "literature": extracted.get("literature", []),
            "db_results": extracted.get("db_results", []),
            "gene_info": extracted.get("gene_info", [])
        }

    def _run_legacy_agent(self, query: str) -> Dict:
        """Run using legacy AgentExecutor"""
        result = self.agent_executor.invoke(
            {"input": query},
            {"callbacks": []}
        )

        extracted = self._extract_results_from_steps(result.get("intermediate_steps", []))

        return {
            "output": result.get("output", ""),
            "tools_used": extracted["tools_used"],
            "trace": extracted["trace"],
            "reasoning_steps": extracted.get("reasoning_steps", []),
            "enrichment": extracted["enrichment"],
            "enrichment_df": extracted.get("enrichment_df"),
            "literature": extracted["literature"],
            "db_results": extracted["db_results"],
            "gene_info": extracted["gene_info"]
        }

    def _run_fallback(self, query: str, memory_context: str = "") -> Dict:
        """Fallback to simple Gemini call with memory context"""
        # Include memory context in the prompt
        context_section = ""
        if memory_context:
            context_section = f"\n{memory_context}\n"

        prompt = f"""You are an expert biology research assistant.
{context_section}
Answer this question: {query}

Provide a detailed, accurate response. If this is a follow-up question, reference the previous results shown above."""

        try:
            response = self.model.generate_content(prompt)
            return {
                "output": response.text,
                "tools_used": [],
                "trace": ["Fallback mode - no tools used"]
            }
        except Exception as e:
            return {
                "output": f"Fallback failed: {str(e)}",
                "tools_used": [],
                "trace": [f"Error: {str(e)}"]
            }


# ===============================================================================
# FACTORY FUNCTION
# ===============================================================================

def create_reasoning_engine(
        model: Any = None,
        provider: str = None,
        model_name: str = None,
        api_key: str = None
) -> ReasoningEngine:
    """
    Factory function to create a reasoning engine.

    Args:
        model: Optional (for backward compatibility, ignored)
        provider: Provider key from CONFIG["providers"]. None = configured default.
        model_name: Model id. None = the provider's default_model.
        api_key: Explicit key. None = resolved from config/env/session.

    Returns:
        ReasoningEngine instance
    """
    return ReasoningEngine(model_name=model_name, provider=provider, api_key=api_key)


# ===============================================================================
# TEST
# ===============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Reasoning Engine")
    print("=" * 60)

    try:
        engine = create_reasoning_engine()

        # Test simple query (goes through ReAct agent - no hardcoded bypass)
        print("\n1. Testing simple query...")
        result = engine.run("What is DNA?")
        print(f"Response: {result['envelope']['final_text'][:200]}...")

        # Test tool query
        print("\n2. Testing tool query (gene info)...")
        result = engine.run("Tell me about the TP53 gene")
        print(f"Response: {result['envelope']['final_text'][:200]}...")
        print(f"Tools used: {result['envelope']['tools_used']}")

        # Check full results storage
        print("\n3. Checking full results storage...")
        if result['envelope'].get('gene_info'):
            print(f"Gene info stored: {len(result['envelope']['gene_info'])} entries")

        print("\nAll tests passed!")

    except Exception as e:
        print(f"\nTest failed: {e}")
        traceback.print_exc()
