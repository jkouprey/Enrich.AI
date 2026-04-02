# tools.py - Production-Ready Functions for Autonomous Biology Assistant
"""
Complete function set for Enrich.AI:
- get_gene_info: Gene information from MyGene.info (simplified, no hidden calls)
- db_retrieve: Flexible database search (Enrichr, 200+ libraries)
- run_enrichment_analysis: Statistical enrichment analysis
- search_literature: Europe PMC search (abstracts, citations)
- get_paper_annotations: Full-text annotations from Europe PMC

VERSION: 3.0 - Europe PMC Integration
"""

from __future__ import annotations
import requests
import pandas as pd
import time
import random
import logging
from collections import OrderedDict, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from functools import lru_cache

# Logging - uses root logger configured by app.py
logger = logging.getLogger(__name__)


# ===============================================================================
# CORE UTILITIES
# ===============================================================================

class SmartCache:
    """Simple caching with TTL and LRU eviction"""

    def __init__(self, max_size: int = 1000, ttl: int = 600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            item, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.cache.move_to_end(key)
                return item
            del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        self.cache[key] = (value, time.time())
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)


cache = SmartCache()


def _retry_request(
        url: str,
        params: dict = None,
        json_data: dict = None,
        files: dict = None,
        data: dict = None,
        method: str = "GET",
        timeout: int = 15,
        max_retries: int = 5
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """Universal retry logic with exponential backoff."""
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = min(30, (2 ** attempt)) + random.uniform(0, 1)
                logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {delay:.2f}s")
                time.sleep(delay)

            if method.upper() == "GET":
                r = requests.get(url, params=params, timeout=timeout)
            elif method.upper() == "POST":
                if files:
                    r = requests.post(url, files=files, data=data, timeout=timeout)
                elif data:
                    r = requests.post(url, data=data, timeout=timeout)
                else:
                    r = requests.post(url, json=json_data, timeout=timeout)
            else:
                return False, None, f"Unsupported method: {method}"

            if r.status_code == 200:
                try:
                    return True, r.json(), None
                except Exception:
                    return True, {"text": r.text}, None

            elif r.status_code in RETRYABLE_STATUS_CODES:
                logger.warning(f"HTTP {r.status_code}, retrying...")
                continue
            else:
                return False, None, f"HTTP {r.status_code}: {r.text[:200]}"

        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                return False, None, f"Timeout after {max_retries} attempts"
            continue
        except requests.exceptions.ConnectionError:
            if attempt == max_retries - 1:
                return False, None, f"Connection error after {max_retries} attempts"
            continue
        except Exception as e:
            if attempt == max_retries - 1:
                return False, None, str(e)
            continue

    return False, None, "Max retries exceeded"


# ===============================================================================
# ENRICHR LIBRARY DISCOVERY & MANAGEMENT
# ===============================================================================

@lru_cache(maxsize=1)
def get_all_enrichr_libraries() -> Dict[str, Dict[str, Any]]:
    """Discover ALL available Enrichr libraries dynamically."""
    logger.info("Fetching all Enrichr libraries...")
    cache_key = "enrichr_all_libraries_v2"
    cached = cache.get(cache_key)
    if cached:
        return cached

    libraries = {}
    enrichr_base = "https://maayanlab.cloud/Enrichr"

    try:
        success, data, error = _retry_request(f"{enrichr_base}/datasetStatistics", timeout=10)

        if not success or not data:
            logger.error(f"Failed to fetch Enrichr libraries: {error}")
            return {}

        stats = data.get("statistics", [])
        for lib_stat in stats:
            lib_name = lib_stat.get("libraryName")
            if lib_name:
                libraries[lib_name] = {
                    "name": lib_name,
                    "num_terms": lib_stat.get("numTerms", 0),
                    "genes_per_term": lib_stat.get("genesPerTerm", 0),
                    "category": lib_name,
                    "ontology_type": lib_name
                }

        cache.set(cache_key, libraries)
        logger.info(f"Discovered {len(libraries)} Enrichr libraries")
        return libraries

    except Exception as e:
        logger.error(f"Error discovering libraries: {e}")
        return {}




def _fetch_enrichr_library_data(library_name: str) -> Dict[str, Dict]:
    """Fetch all data for a specific Enrichr library"""
    cache_key = f"enrichr_lib_{library_name}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    enrichr_base = "https://maayanlab.cloud/Enrichr"

    try:
        url = f"{enrichr_base}/geneSetLibrary?mode=text&libraryName={library_name}"
        success, data, error = _retry_request(url, timeout=30)

        if not success or not data:
            logger.error(f"Failed to fetch library {library_name}: {error}")
            return {}

        library_data = {}
        text_data = data.get("text", "") if isinstance(data, dict) else str(data)
        lines = text_data.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            term_name = parts[0].strip()
            potential_desc = parts[1].strip()

            if len(potential_desc) < 15 and potential_desc.isupper():
                description = ""
                genes = [g.strip() for g in parts[1:] if g.strip()]
            else:
                description = potential_desc
                genes = [g.strip() for g in parts[2:] if g.strip()]

            library_data[term_name] = {
                "id": term_name,
                "description": description,
                "genes": genes
            }

        cache.set(cache_key, library_data)
        logger.info(f"Loaded {len(library_data)} terms from {library_name}")
        return library_data

    except Exception as e:
        logger.error(f"Error fetching library {library_name}: {e}")
        return {}


# ===============================================================================
# FUNCTION 1: GET GENE INFO (SIMPLIFIED - MyGene.info ONLY)
# ===============================================================================

def get_gene_info(
        gene: str,
        organism: str = "9606"
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Get basic information about a gene from MyGene.info.

    Use db_retrieve separately if you need detailed pathway information.

    Parameters:
    -----------
    gene : str
        Gene symbol or comma-separated list (e.g., "TP53" or "TP53,BRCA1,CD4")
    organism : str
        NCBI taxonomy ID (default "9606" for human)

    Returns:
    --------
    Dict with gene information including name, summary, GO terms
    """
    if not gene or not isinstance(gene, str):
        return {"error": "Gene symbol is required", "found": False}

        # Support comma-separated batch queries
    gene_list = [g.strip().upper() for g in gene.split(',') if g.strip()]
    if len(gene_list) > 1:
        results = []
        for g in gene_list[:20]:  # Cap at 20
            results.append(get_gene_info(gene=g, organism=organism))
        return results

    gene = gene_list[0]
    logger.info(f"Getting gene info for: {gene}")

    result = {
        "gene": gene,
        "organism": organism,
        "found": False
    }

    try:
        url = "https://mygene.info/v3/query"
        params = {
            "q": f"symbol:{gene} AND taxid:{organism}",
            "fields": "name,summary,symbol,alias,go,pathway,generif",
            "size": 1
        }

        success, data, error = _retry_request(url, params=params, timeout=10)

        if not success or not data:
            result["error"] = error
            return result

        hits = data.get("hits", [])
        if not hits:
            result["error"] = f"Gene {gene} not found"
            return result

        gene_data = hits[0]
        result["found"] = True
        result["name"] = gene_data.get("name", "")
        result["summary"] = gene_data.get("summary", "")
        result["aliases"] = gene_data.get("alias", [])
        if isinstance(result["aliases"], str):
            result["aliases"] = [result["aliases"]]

        # Extract GO terms directly from API response
        go_data = gene_data.get("go", {})
        result["go_terms"] = {
            "biological_process": _extract_go_terms(go_data.get("BP", [])),
            "molecular_function": _extract_go_terms(go_data.get("MF", [])),
            "cellular_component": _extract_go_terms(go_data.get("CC", []))
        }

        # Extract pathway info from MyGene (counts only, no Enrichr calls)
        pathway_data = gene_data.get("pathway", {})
        result["pathways_summary"] = {
            "kegg_count": len(_ensure_list(pathway_data.get("kegg", []))),
            "reactome_count": len(_ensure_list(pathway_data.get("reactome", []))),
            "wikipathways_count": len(_ensure_list(pathway_data.get("wikipathways", [])))
        }

        # Include pathway names (lightweight, from MyGene only)
        result["kegg_pathways"] = [
            p.get("name", "") for p in _ensure_list(pathway_data.get("kegg", []))
        ][:10]
        result["reactome_pathways"] = [
            p.get("name", "") for p in _ensure_list(pathway_data.get("reactome", []))
        ][:10]

        logger.info(f"Successfully retrieved info for gene {gene}")
        return result

    except Exception as e:
        logger.error(f"Error getting gene info: {e}")
        result["error"] = str(e)
        return result


def _extract_go_terms(go_list: Any) -> List[Dict[str, str]]:
    """Extract GO terms from MyGene response"""
    if not go_list:
        return []
    if isinstance(go_list, dict):
        go_list = [go_list]
    return [
        {"id": t.get("id", ""), "term": t.get("term", "")}
        for t in go_list[:10]
    ]


def _ensure_list(val: Any) -> List:
    """Ensure value is a list"""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


# ===============================================================================
# FUNCTION 2: DB_RETRIEVE (FLEXIBLE DATABASE SEARCH)
# ===============================================================================

def db_retrieve(
        query: Union[str, List[str], Dict],
        libraries: Optional[Union[str, List[str]]] = None,
        task: str = "auto",
        organism: str = "9606",
        limit: int = 20,
        include_genes: bool = False,
        include_descriptions: bool = True,
        similarity_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Universal database retrieval function for biological data.

    Parameters:
    -----------
    Search query (gene symbols, term names, IDs, or keywords).
        Use short specific terms for best results.
        For complex queries, call this tool multiple times with different focused terms.

    libraries : str or List[str], optional
        Specific Enrichr library/libraries to search.
        If None, automatically selects appropriate libraries.

    task : str
        Task to perform. Options:
        - "auto": Automatically infer task from query (default)
        - "term_search": Search for terms by name/keyword
        - "term_details": Get detailed info about specific term(s)
        - "term_genes": Get genes for specific term(s)
        - "gene_terms": Get all terms for gene(s)
        - "library_browse": Browse all terms in library
        - "find_similar": Find terms similar to query. instead of term_search we use this to get relevant terms based on common genes.

    organism : str
        NCBI taxonomy ID. Default "9606" (human)

    limit : int
        Maximum number of results to return

    include_genes : bool
        Whether to fetch gene lists for terms

    include_descriptions : bool
        Include term descriptions/definitions

    similarity_threshold : float
        Minimum similarity score for "find_similar" task (0-1)

    Returns:
    --------
    Dict with results, metadata, and statistics
    """
    logger.info("=" * 80)
    logger.info(f"DB_RETRIEVE called with query: {query}, task: {task}")
    logger.info("=" * 80)

    response = {
        "task_performed": task,
        "query_info": {"original_query": query, "libraries_used": [], "organism": organism},
        "results": [],
        "metadata": {},
        "sources": [],
        "statistics": {"total_results": 0, "libraries_queried": 0, "execution_time_seconds": 0},
        "errors": [],
        "warnings": []
    }

    start_time = time.time()

    try:
        # Step 1: Normalize query
        # Step 1: Normalize query
        query_normalized = _normalize_query(query)
        # If task is explicitly term_search, force text type regardless of normalization
        if task == "term_search" and query_normalized.get("type") == "genes":
            query_normalized = {"type": "text", "value": " ".join(query_normalized["value"])}
        response["query_info"]["normalized_query"] = query_normalized

        # Step 2: Determine task if auto
        if task == "auto":
            task = _infer_task(query_normalized, libraries)
            response["task_performed"] = task
            logger.info(f"Inferred task: {task}")

        # Step 3: Select libraries if not specified
        if libraries is None:
            libraries = _select_libraries_for_query(query_normalized, task)
            response["warnings"].append("Libraries auto-selected based on query")
        elif isinstance(libraries, str):
            libraries = [libraries]

        response["query_info"]["libraries_used"] = libraries
        response["statistics"]["libraries_queried"] = len(libraries)

        # Step 4: Execute appropriate retrieval strategy
        logger.info(f"Executing task '{task}' on {len(libraries)} libraries...")

        if task == "term_search":
            results = _search_terms_in_libraries(query_normalized, libraries, limit, include_descriptions)
        elif task == "term_details":
            results = _get_term_details(query_normalized, libraries, include_genes)
        elif task == "term_genes":
            results = _get_genes_for_terms(query_normalized, libraries, organism, limit)
        elif task == "gene_terms":
            results = _get_terms_for_genes(query_normalized, libraries, organism, limit)
        elif task == "library_browse":
            results = _browse_library_terms(libraries, limit, include_descriptions, include_genes)
        elif task == "find_similar":
            results = _find_similar_terms(query_normalized, libraries, similarity_threshold, limit)
        else:
            response["errors"].append(f"Unknown task: {task}")
            return response

        response["results"] = results
        response["statistics"]["total_results"] = len(results)
        response["statistics"]["execution_time_seconds"] = round(time.time() - start_time, 2)
        response["sources"] = list(set([r.get("source") for r in results if r.get("source")]))

        logger.info(f"Task completed: {len(results)} results")
        return response

    except Exception as e:
        logger.error(f"Error in db_retrieve: {e}", exc_info=True)
        response["errors"].append(str(e))
        response["statistics"]["execution_time_seconds"] = round(time.time() - start_time, 2)
        return response


def _normalize_query(query: Union[str, List[str], Dict]) -> Dict[str, Any]:
    """Normalize various query formats into standard structure.

    Handles:
    - Single gene: "TP53" → {"type": "genes", "value": ["TP53"]}
    - Comma-separated genes: "TP53, BRCA1" → {"type": "genes", "value": ["TP53", "BRCA1"]}
    - Term IDs: "GO:0006915" → {"type": "term_id", "value": "GO:0006915"}
    - Free text: "apoptosis regulation" → {"type": "text", "value": "apoptosis regulation"}
    """
    if isinstance(query, str):
        query_stripped = query.strip()
        query_upper = query_stripped.upper()

        # Check for term IDs first
        if any(prefix in query_upper for prefix in ["GO:", "KEGG:", "HSA", "R-HSA", "WP", "HALLMARK"]):
            return {"type": "term_id", "value": query_upper}

        # Check if this looks like comma-separated or space-separated gene symbols
        # Split by comma first, then by spaces if no commas
        if ',' in query_stripped:
            tokens = [t.strip() for t in query_stripped.split(',') if t.strip()]
        else:
            tokens = query_stripped.split()

        # Detect if tokens look like gene symbols:
        # Gene symbols are typically short (1-15 chars), alphanumeric (may contain hyphens/dots),
        # and are UPPERCASE or mixed case. Pure lowercase multi-word queries are text searches.
        if len(tokens) >= 1:
            gene_like = all(
                len(t) <= 15 and t.replace('-', '').replace('.', '').replace('_', '').isalnum()
                for t in tokens
            )
            # If any token is a lowercase word > 3 chars, it's likely text search, not genes
            # Gene symbols are almost always uppercase (TP53, BRCA1, IL-6, etc.)
            has_lowercase_words = any(
                t.islower() and len(t) > 3 for t in tokens
            )
            # Single long lowercase word is text search
            single_lowercase = len(tokens) == 1 and query_stripped.islower() and len(query_stripped) > 3

            if gene_like and not has_lowercase_words and not single_lowercase:
                return {"type": "genes", "value": [t.upper() for t in tokens]}

        return {"type": "text", "value": query_stripped}
    elif isinstance(query, list):
        normalized_list = [str(item).strip().upper() for item in query if item]
        if any(item.startswith("GO:") or item.startswith("KEGG:") for item in normalized_list):
            return {"type": "term_ids", "value": normalized_list}
        return {"type": "genes", "value": normalized_list}
    elif isinstance(query, dict):
        return query
    return {"type": "text", "value": str(query)}


def _infer_task(query_normalized: Dict, libraries: Optional[List[str]]) -> str:
    """Infer task from query data type only. LLM should specify task explicitly for best results."""
    q_type = query_normalized.get("type")
    # Simple inference based on data type only
    if q_type == "term_id" or q_type == "term_ids":
        return "term_details"
    elif q_type == "genes":
        return "gene_terms"
    # Default - LLM should specify task
    return "term_search"


def _select_libraries_for_query(query_normalized: Dict, task: str) -> List[str]:
    """Return default libraries. LLM should specify libraries explicitly for best results."""
    # Simple defaults - LLM overrides with specific libraries
    return [
        "GO_Biological_Process_2023",
        "GO_Molecular_Function_2023",
        "GO_Cellular_Component_2023",
        "KEGG_2021_Human",
        "Reactome_2022"
    ]


def _search_terms_in_libraries(query_normalized: Dict, libraries: List[str], limit: int, include_descriptions: bool) -> \
        List[Dict]:
    """Search for terms across multiple Enrichr libraries"""
    results = []
    raw_value = query_normalized.get("value", "")
    if isinstance(raw_value, list):
        search_text = " ".join(raw_value).lower()
    else:
        search_text = str(raw_value).lower()
    search_words = [w for w in search_text.split() if len(w) > 2]

    for library in libraries:
        try:
            library_data = _fetch_enrichr_library_data(library)
            if not library_data:
                continue

            for term_name, term_info in library_data.items():
                term_lower = term_name.lower()
                term_words = set(term_lower.split())
                matched = sum(
                    1 for w in search_words if w in term_words or w.rstrip('s') in term_words or w + 's' in term_words)
                if matched >= max(1, len(search_words) // 2):
                    result = {
                        "term_id": term_info.get("id", term_name),
                        "term_name": term_name,
                        "library": library,
                        "source": "Enrichr",
                        "num_genes": len(term_info.get("genes", []))
                    }
                    if include_descriptions:
                        result["description"] = term_info.get("description", "")
                    results.append(result)
                    if len(results) >= limit:
                        break

            if len(results) >= limit:
                break
        except Exception as e:
            logger.error(f"Error searching {library}: {e}")

    return results[:limit]


def _get_term_details(query_normalized: Dict, libraries: List[str], include_genes: bool) -> List[Dict]:
    """Get detailed information about specific terms"""
    results = []
    term_ids = query_normalized.get("value")
    if isinstance(term_ids, str):
        term_ids = [term_ids]

    for library in libraries:
        try:
            library_data = _fetch_enrichr_library_data(library)
            if not library_data:
                continue

            for term_id in term_ids:
                for term_name, term_info in library_data.items():
                    if term_id.upper() in term_name.upper():
                        result = {
                            "term_id": term_info.get("id", term_name),
                            "term_name": term_name,
                            "library": library,
                            "source": "Enrichr",
                            "description": term_info.get("description", ""),
                            "num_genes": len(term_info.get("genes", []))
                        }
                        if include_genes:
                            result["genes"] = term_info.get("genes", [])
                        results.append(result)
                        break
        except Exception as e:
            logger.error(f"Error getting term details from {library}: {e}")

    return results


def _get_genes_for_terms(query_normalized: Dict, libraries: List[str], organism: str, limit: int) -> List[Dict]:
    """Get all genes associated with specific terms"""
    results = []
    term_ids = query_normalized.get("value")
    if isinstance(term_ids, str):
        term_ids = [term_ids]

    for library in libraries:
        try:
            library_data = _fetch_enrichr_library_data(library)
            if not library_data:
                continue

            for term_id in term_ids:
                for term_name, term_info in library_data.items():
                    if term_id.upper() in term_name.upper():
                        genes = term_info.get("genes", [])
                        results.append({
                            "term_id": term_info.get("id", term_name),
                            "term_name": term_name,
                            "library": library,
                            "genes": genes[:limit],
                            "total_genes": len(genes),
                            "source": "Enrichr"
                        })
                        break
        except Exception as e:
            logger.error(f"Error getting genes from {library}: {e}")

    return results


def _get_terms_for_genes(query_normalized: Dict, libraries: List[str], organism: str, limit: int) -> List[Dict]:
    """Get all terms associated with specific genes"""
    results = []
    genes = query_normalized.get("value")
    if isinstance(genes, str):
        genes = [genes]
    genes = [g.upper() for g in genes]

    for library in libraries:
        try:
            library_data = _fetch_enrichr_library_data(library)
            if not library_data:
                continue

            for term_name, term_info in library_data.items():
                term_genes = [g.upper() for g in term_info.get("genes", [])]
                overlap = set(genes).intersection(set(term_genes))
                if overlap:
                    results.append({
                        "term_id": term_info.get("id", term_name),
                        "term_name": term_name,
                        "library": library,
                        "overlap_genes": list(overlap),
                        "overlap_count": len(overlap),
                        "total_genes": len(term_genes),
                        "source": "Enrichr"
                    })
        except Exception as e:
            logger.error(f"Error getting terms from {library}: {e}")

    results.sort(key=lambda x: x.get("overlap_count", 0), reverse=True)
    return results[:limit]


def _browse_library_terms(libraries: List[str], limit: int, include_descriptions: bool, include_genes: bool = False) -> \
        List[Dict]:
    """Browse terms in a library"""
    results = []
    for library in libraries:
        try:
            library_data = _fetch_enrichr_library_data(library)
            if not library_data:
                continue

            for term_name, term_info in list(library_data.items())[:limit]:
                result = {
                    "term_id": term_info.get("id", term_name),
                    "term_name": term_name,
                    "library": library,
                    "num_genes": len(term_info.get("genes", [])),
                    "source": "Enrichr"
                }
                if include_descriptions:
                    result["description"] = term_info.get("description", "")
                if include_genes:
                    result["genes"] = term_info.get("genes", [])
                results.append(result)

            if len(results) >= limit:
                break
        except Exception as e:
            logger.error(f"Error browsing {library}: {e}")

    return results[:limit]


def _find_similar_terms(query_normalized: Dict, libraries: List[str], similarity_threshold: float, limit: int) -> List[
    Dict]:
    """Find biologically similar terms using gene-set Jaccard similarity.

    Accepts either gene symbols or a term ID as input.
    If a term ID is provided, resolves its genes first, then computes Jaccard.
    Works across all Enrichr libraries (GO, KEGG, Reactome, etc.).
    """
    q_type = query_normalized.get("type", "")
    query_value = query_normalized.get("value", "")
    reference_genes = None
    reference_name = None

    # Resolve reference genes based on input type
    if q_type in ("term_id", "term_ids"):
        term_id = query_value if isinstance(query_value, str) else query_value[0]
        for library in libraries:
            try:
                library_data = _fetch_enrichr_library_data(library)
                if not library_data:
                    continue
                for term_name, term_info in library_data.items():
                    if term_id.upper() in term_name.upper():
                        reference_genes = set(g.upper() for g in term_info.get("genes", []))
                        reference_name = term_name
                        break
                if reference_genes:
                    break
            except Exception as e:
                logger.error(f"Error resolving term genes in {library}: {e}")

        if not reference_genes:
            logger.warning(f"Could not resolve genes for term: {term_id}")
            return []
        logger.info(f"Resolved {len(reference_genes)} genes from term {term_id}")

    elif q_type == "genes":
        reference_genes = set(g.upper() for g in query_value) if isinstance(query_value, list) else set(
            g.strip().upper() for g in str(query_value).split(",") if g.strip())

    else:
        search_text = str(query_value).lower()
        for library in libraries:
            try:
                library_data = _fetch_enrichr_library_data(library)
                if not library_data:
                    continue
                for term_name, term_info in library_data.items():
                    if search_text in term_name.lower():
                        candidate = set(g.upper() for g in term_info.get("genes", []))
                        if candidate:
                            reference_genes = candidate
                            reference_name = term_name
                            break
                if reference_genes:
                    break
            except Exception as e:
                logger.error(f"Error resolving term in {library}: {e}")

        if not reference_genes:
            logger.warning(f"Could not resolve genes for: {query_value}")
            return []

    if not reference_genes:
        logger.warning("find_similar requires gene symbols or a valid term as input")
        return []

    logger.info(f"Finding similar terms for {len(reference_genes)} reference genes")

    # Compute Jaccard similarity against all terms
    results = []
    for library in libraries:
        try:
            library_data = _fetch_enrichr_library_data(library)
            if not library_data:
                continue

            for term_name, term_info in library_data.items():
                if term_name == reference_name:
                    continue
                term_genes = set(g.upper() for g in term_info.get("genes", []))
                if not term_genes:
                    continue

                intersection = len(reference_genes & term_genes)
                if intersection == 0:
                    continue
                union = len(reference_genes | term_genes)
                jaccard = intersection / union if union > 0 else 0

                if jaccard >= similarity_threshold:
                    results.append({
                        "term_id": term_info.get("id", term_name),
                        "term_name": term_name,
                        "library": library,
                        "similarity_score": round(jaccard, 3),
                        "shared_genes": intersection,
                        "num_genes": len(term_genes),
                        "source": "Enrichr"
                    })
        except Exception as e:
            logger.error(f"Error finding similar in {library}: {e}")

    results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
    return results[:limit]


# ===============================================================================
# ENRICHMENT HELPER: GET LIBRARY GMT DATA FOR PATHWAY SIZES
# ===============================================================================

_gmt_cache = {}  # Simple cache for GMT data


def _get_library_gmt_cache(library_name: str) -> Dict[str, int]:
    """
    Get pathway sizes from Enrichr library GMT file.
    Returns dict: {term_name: total_gene_count}
    """
    if library_name in _gmt_cache:
        return _gmt_cache[library_name]

    try:
        enrichr_base = "https://maayanlab.cloud/Enrichr"
        url = f"{enrichr_base}/geneSetLibrary?mode=text&libraryName={library_name}"
        success, data, error = _retry_request(url, timeout=30)

        if not success:
            logger.warning(f"Could not fetch GMT for {library_name}: {error}")
            return {}

        # _retry_request returns {"text": ...} for non-JSON responses
        text = data.get("text", "") if isinstance(data, dict) else str(data)

        pathway_sizes = {}
        for line in text.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                term_name = parts[0]
                # Count genes (skip term name and description)
                gene_count = len([g for g in parts[2:] if g.strip()])
                pathway_sizes[term_name] = gene_count

        _gmt_cache[library_name] = pathway_sizes
        logger.info(f"Cached GMT for {library_name}: {len(pathway_sizes)} terms")
        return pathway_sizes

    except Exception as e:
        logger.warning(f"Could not fetch GMT for {library_name}: {e}")
        return {}


# ===============================================================================
# FUNCTION 3: RUN ENRICHMENT ANALYSIS
# ===============================================================================

def run_enrichment_analysis(
        genes: List[str],
        libraries: Optional[List[str]] = None,
        organism: str = "9606",
        p_value_threshold: float = 0.05,
        top_n: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run statistical enrichment analysis on a gene list using Enrichr.

    REQUIRES: Minimum 4 genes for statistical validity.

    Parameters:
    -----------
    genes : List[str]
        List of gene symbols (minimum 4 required)
    libraries : List[str], optional
        Enrichr libraries to test
    p_value_threshold : float
        Adjusted p-value threshold for significance
    top_n : int, optional
        Maximum number of terms per library. If None (default), returns ALL significant terms.

    Returns:
    --------
    Dict with enrichment results
    """
    if not genes or not isinstance(genes, list):
        return {"error": "Gene list is required", "enrichment_results": {}}

    genes = [g.strip().upper() for g in genes if g and isinstance(g, str)]

    if len(genes) < 4:
        return {
            "error": f"Minimum 4 genes required for enrichment. You provided {len(genes)}.",
            "suggestion": "Use db_retrieve with task='gene_terms' for single gene lookup.",
            "enrichment_results": {}
        }

    logger.info(f"Running enrichment analysis on {len(genes)} genes")
    start_time = time.time()

    results = {
        "query_genes": genes,
        "gene_count": len(genes),
        "libraries_tested": [],
        "enrichment_results": {},
        "significant_terms_total": 0,
        "execution_time": 0
    }

    try:
        if libraries is None:
            libraries = [
                "GO_Biological_Process_2023",
                "GO_Molecular_Function_2023",
                "GO_Cellular_Component_2023",
                "KEGG_2021_Human",
                "Reactome_2022",
                "WikiPathways_2021_Human",
                "MSigDB_Hallmark_2020"
            ]

        results["libraries_tested"] = libraries

        # Submit gene list to Enrichr
        enrichr_base = "https://maayanlab.cloud/Enrichr"
        gene_list_str = "\n".join(genes)

        success, data, error = _retry_request(
            f"{enrichr_base}/addList",
            files={'list': (None, gene_list_str)},
            data={'description': ''},
            method="POST",
            timeout=30
        )

        if not success or not data:
            results["error"] = f"Failed to submit gene list: {error}"
            return results

        user_list_id = data.get("userListId")
        if not user_list_id:
            results["error"] = "No userListId returned from Enrichr"
            return results

        logger.info(f"Gene list submitted. ID: {user_list_id}")

        # Query each library
        for library in libraries:
            try:
                url = f"{enrichr_base}/enrich"
                params = {"userListId": user_list_id, "backgroundType": library}

                success, data, error = _retry_request(url, params=params, timeout=30)

                if not success or not data:
                    continue

                enrichment_data = data.get(library, [])
                if not enrichment_data:
                    continue

                # Get pathway sizes for overlap ratio
                library_gmt = _get_library_gmt_cache(library)

                # First, collect ALL significant terms (filter by p-value threshold)
                significant_terms = []
                for term_data in enrichment_data:
                    adjusted_pval = term_data[6] if len(term_data) > 6 else 1.0

                    if adjusted_pval <= p_value_threshold:
                        term_name = term_data[1] if len(term_data) > 1 else ""
                        genes_list = term_data[5] if len(term_data) > 5 else []

                        # VERIFIED from R code: c("Rank", "Term", "p.value", "z.score",
                        #                          "combined.score", "Genes", "adj.p.value",
                        #                          "old.p.value", "old.adj.p.value")
                        # [0]=rank, [1]=term, [2]=p_value, [3]=z_score, [4]=combined_score,
                        # [5]=genes, [6]=adj_p, [7]=old_p, [8]=old_adj_p

                        # Calculate overlap ratio (e.g., "4/150")
                        overlap_count = len(genes_list)
                        total_in_pathway = library_gmt.get(term_name, overlap_count)  # Fallback to overlap if not found
                        overlap_ratio = f"{overlap_count}/{total_in_pathway}"

                        significant_terms.append({
                            "rank": term_data[0] if len(term_data) > 0 else 0,
                            "term_name": term_name,
                            "term": term_name,  # Alias for compatibility
                            "overlap": overlap_ratio,  # e.g., "4/150"
                            "p_value": term_data[2] if len(term_data) > 2 else 1.0,
                            "z_score": term_data[3] if len(term_data) > 3 else 0.0,  # Confirmed: z_score (can be 500+!)
                            "combined_score": term_data[4] if len(term_data) > 4 else 0.0,
                            "overlap_genes": genes_list,
                            "genes": genes_list,  # Alias for compatibility
                            "adjusted_p_value": adjusted_pval
                        })

                if significant_terms:
                    # Sort by adjusted p-value (most significant first)
                    significant_terms.sort(key=lambda x: x.get("adjusted_p_value", 1.0))

                    # Apply top_n limit if specified
                    if top_n is not None:
                        significant_terms = significant_terms[:top_n]

                    results["enrichment_results"][library] = significant_terms
                    results["significant_terms_total"] += len(significant_terms)

            except Exception as e:
                logger.warning(f"Error with library {library}: {e}")
                continue

        results["execution_time"] = round(time.time() - start_time, 2)

        # Create DataFrame
        if results["enrichment_results"]:
            all_terms = []
            for library, terms in results["enrichment_results"].items():
                for term in terms:
                    term_row = term.copy()
                    term_row["library"] = library
                    all_terms.append(term_row)

            if all_terms:
                results["dataframe"] = pd.DataFrame(all_terms).sort_values(
                    "combined_score", ascending=False
                )

        logger.info(f"Enrichment complete: {results['significant_terms_total']} significant terms")
        return results

    except Exception as e:
        logger.error(f"Error in enrichment analysis: {e}")
        results["error"] = str(e)
        return results


# ===============================================================================
# FUNCTION 4: SEARCH LITERATURE (Europe PMC - Abstracts)
# ===============================================================================

def search_literature(
        query: str,
        max_results: int = 20,
        search_full_text: bool = True,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        sort_by: str = "relevance"
) -> Dict[str, Any]:
    """
    Search for scientific papers using Europe PMC.

    Returns papers with titles, abstracts, citations, and metadata.
    Use this for general literature search and relevance checking.

    Parameters:
    -----------
    query : str
        Search query (e.g., "TP53 cancer therapy")
    max_results : int
        Maximum papers to return (default 20, max 100)
    search_full_text : bool
        Include full-text articles from PMC (default True)
    min_year : int, optional
        Minimum publication year
    max_year : int, optional
        Maximum publication year
    sort_by : str
        "relevance" (default), "citations", or "date"

    Returns:
    --------
    Dict with papers including title, abstract, citations, authors, year
    """
    if not query or not isinstance(query, str):
        return {"error": "Search query is required", "papers": []}

    query = query.strip()
    if not query:
        return {"error": "Search query cannot be empty", "papers": []}

    logger.info(f"Literature search: {query}")
    start_time = time.time()

    # Build Europe PMC query
    search_query = query

    # Add year filters
    if min_year and max_year:
        search_query += f" AND (PUB_YEAR:[{min_year} TO {max_year}])"
    elif min_year:
        current_year = datetime.now().year
        search_query += f" AND (PUB_YEAR:[{min_year} TO {current_year}])"
    elif max_year:
        search_query += f" AND (PUB_YEAR:[1900 TO {max_year}])"

    # Determine sort order
    sort_param = ""
    if sort_by == "citations":
        sort_param = "CITED desc"
    elif sort_by == "date":
        sort_param = "P_PDATE_D desc"
    # relevance is default, no sort param needed

    results = {
        "query": query,
        "total_found": 0,
        "papers": [],
        "execution_time": 0
    }

    try:
        # Europe PMC REST API
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query": search_query,
            "format": "json",
            "pageSize": min(max_results, 100),
            "resultType": "core"  # Get full metadata
        }

        if sort_param:
            params["sort"] = sort_param

        success, data, error = _retry_request(url, params=params, timeout=30)

        if not success or not data:
            results["error"] = f"Search failed: {error}"
            return results

        result_list = data.get("resultList", {}).get("result", [])
        results["total_found"] = data.get("hitCount", 0)

        for paper in result_list:
            # Convert year to int
            year_val = paper.get("pubYear", 0)
            try:
                year_val = int(year_val)
            except Exception:
                year_val = 0

            # Get citation count
            cit_val = paper.get("citedByCount", 0)
            try:
                cit_val = int(cit_val)
            except Exception:
                cit_val = 0

            # Build paper URL (prefer DOI > PMID > PMCID)
            doi = paper.get("doi", "")
            pmid = paper.get("pmid", "")
            pmcid = paper.get("pmcid", "")

            if doi:
                paper_url = f"https://doi.org/{doi}"
            elif pmid:
                paper_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
            elif pmcid:
                paper_url = f"https://europepmc.org/article/PMC/{pmcid}"
            else:
                paper_url = ""

            paper_info = {
                "pmid": pmid,
                "pmcid": pmcid,
                "doi": doi,
                "url": paper_url,  # Direct link to paper
                "title": paper.get("title", ""),
                "abstract": paper.get("abstractText", ""),
                "authors": paper.get("authorString", ""),
                "journal": paper.get("journalTitle", ""),
                "journalTitle": paper.get("journalTitle", ""),  # Alias
                "year": year_val,
                "pub_year": year_val,  # Alias for compatibility
                "citations": cit_val,
                "citation_count": cit_val,  # Alias for compatibility
                "is_open_access": paper.get("isOpenAccess", "N") == "Y",
                "source": paper.get("source", ""),
                "has_full_text": paper.get("hasFullText", "N") == "Y",
                # Additional useful fields
                "issue": paper.get("issue", ""),
                "journal_volume": paper.get("journalVolume", ""),
                "page_info": paper.get("pageInfo", ""),
                "pub_type": paper.get("pubType", ""),
                "has_pdf": paper.get("hasPDF", "N") == "Y",
                "has_suppl": paper.get("hasSuppl", "N") == "Y",
                "first_publication_date": paper.get("firstPublicationDate", "")
            }

            results["papers"].append(paper_info)

        results["execution_time"] = round(time.time() - start_time, 2)

        logger.info(f"Found {len(results['papers'])} papers in {results['execution_time']}s")
        return results

    except Exception as e:
        logger.error(f"Literature search error: {e}")
        results["error"] = str(e)
        return results


# ===============================================================================
# FUNCTION 5: GET PAPER ANNOTATIONS (Europe PMC - Full Text Mining)
# ===============================================================================

def get_paper_annotations(
        paper_id: str,
        id_type: str = "auto",
        annotation_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get text-mined annotations from a paper's full text using Europe PMC.

    Returns entities extracted from the FULL TEXT (not just abstract):
    - Genes/Proteins
    - GO Terms
    - Diseases
    - Chemicals
    - Organisms

    Parameters:
    -----------
    paper_id : str
        Paper identifier (PMID, PMCID, or DOI)
    id_type : str
        Type of ID: "pmid", "pmcid", "doi", or "auto" (detect automatically)
    annotation_types : List[str], optional
        Types to retrieve. Options: "Gene_Proteins", "Diseases", "Chemicals",
        "GO_Terms", "Organisms", "Accession_Numbers"
        If None, returns all types.

    Returns:
    --------
    Dict with annotations organized by type, including text location
    """
    if not paper_id or not isinstance(paper_id, str):
        return {"error": "Paper ID is required", "annotations": {}}

    paper_id = paper_id.strip()
    logger.info(f"Getting annotations for paper: {paper_id}")

    # Auto-detect ID type
    if id_type == "auto":
        if paper_id.upper().startswith("PMC"):
            id_type = "pmcid"
        elif paper_id.startswith("10."):
            id_type = "doi"
        else:
            id_type = "pmid"

    # Format the article ID for Europe PMC
    if id_type == "pmid":
        article_id = f"MED:{paper_id}"
    elif id_type == "pmcid":
        # Remove PMC prefix if present
        pmcid = paper_id.upper().replace("PMC", "")
        article_id = f"PMC:{pmcid}"
    elif id_type == "doi":
        article_id = f"DOI:{paper_id}"
    else:
        article_id = paper_id

    # Default annotation types
    if annotation_types is None:
        annotation_types = [
            "Gene_Proteins",
            "Diseases",
            "Chemicals",
            "GO_Terms",
            "Organisms"
        ]

    results = {
        "paper_id": paper_id,
        "id_type": id_type,
        "annotations": {},
        "annotation_counts": {},
        "execution_time": 0
    }

    start_time = time.time()

    try:
        # Europe PMC Annotations API
        url = "https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds"
        params = {
            "articleIds": article_id,
            "format": "JSON"
        }

        # Add type filter if specific types requested
        if annotation_types:
            params["type"] = ",".join(annotation_types)

        success, data, error = _retry_request(url, params=params, timeout=30)

        if not success or not data:
            results["error"] = f"Failed to get annotations: {error}"
            return results

        # Parse annotations
        annotations_list = data if isinstance(data, list) else data.get("annotations", [])

        # Organize by type
        by_type = defaultdict(list)

        for ann_group in annotations_list:
            # Each group contains annotations for one article
            for annotation in ann_group.get("annotations", []):
                ann_type = annotation.get("type", "Unknown")

                ann_info = {
                    "text": annotation.get("exact", ""),
                    "section": annotation.get("section", ""),
                    "prefix": annotation.get("prefix", ""),
                    "postfix": annotation.get("postfix", ""),
                    "tags": annotation.get("tags", [])
                }

                # Extract database IDs from tags
                for tag in annotation.get("tags", []):
                    if tag.get("name") == "go_id":
                        ann_info["go_id"] = tag.get("uri", "")
                    elif tag.get("name") == "gene_id":
                        ann_info["gene_id"] = tag.get("uri", "")

                by_type[ann_type].append(ann_info)

        # Deduplicate and count
        for ann_type, annotations in by_type.items():
            # Get unique terms
            seen_texts = set()
            unique_annotations = []
            for ann in annotations:
                text_lower = ann["text"].lower()
                if text_lower not in seen_texts:
                    seen_texts.add(text_lower)
                    unique_annotations.append(ann)

            results["annotations"][ann_type] = unique_annotations
            results["annotation_counts"][ann_type] = len(unique_annotations)

        results["execution_time"] = round(time.time() - start_time, 2)

        total_annotations = sum(results["annotation_counts"].values())
        logger.info(f"Found {total_annotations} unique annotations in {results['execution_time']}s")

        return results

    except Exception as e:
        logger.error(f"Annotation retrieval error: {e}")
        results["error"] = str(e)
        return results



# ===============================================================================
# MAIN
# ===============================================================================

if __name__ == "__main__":
    print("Biology Assistant Tools - Version 3.0")
    print(f"Available Enrichr Libraries: {len(get_all_enrichr_libraries())}")
    print("\nFunctions:")
    print("  1. get_gene_info(gene)")
    print("  2. db_retrieve(query, task, libraries)")
    print("  3. run_enrichment_analysis(genes)")
    print("  4. search_literature(query)")
    print("  5. get_paper_annotations(paper_id)")