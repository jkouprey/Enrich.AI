# Enrich.AI

An autonomous biology research assistant for gene set enrichment analysis and biological interpretation. Powered by Google Gemini 2.5 Flash and orchestrated via LangGraph, Enrich.AI uses a ReAct (Reason → Act → Observe) reasoning loop to autonomously select tools, interpret results, and synthesize biological insights — with zero hardcoded decision trees.

## What it does

Give Enrich.AI a biological question in natural language. The model decides which tools to call, in what order, and with what parameters.

**5 autonomous tools:**

- **Enrichment Analysis** — Statistical enrichment (Fisher's exact test) against 222 Enrichr gene set libraries (GO, KEGG, Reactome, MSigDB, and more)
- **Database Query** — Search Enrichr libraries for pathways, GO terms, and gene-term associations. Supports keyword search, term lookup, gene-set Jaccard similarity, and more
- **Gene Information** — Gene function, GO annotations, pathway memberships, and disease associations via MyGene.info
- **Literature Search** — Scientific paper search via Europe PMC with AI-scored relevance ratings
- **Paper Annotations** — Text-mined genes, diseases, drugs, and GO terms extracted from full-text papers via Europe PMC

**6 visualization types:** Bar plot, bubble chart, UpSet plot, hierarchical dendrogram, similarity heatmap with clustering, and concept network (cnetplot). All include interactive controls and optional AI interpretation.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/jkouprey/EnrichAI.git
cd EnrichAI
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate enrichai
```

### 2. Get a Gemini API key

Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Launch

```bash
streamlit run app.py
```

Enter your API key in the sidebar when prompted.

## Architecture

Enrich.AI uses a **ReAct agent** pattern:

1. **Think** — Analyze the biological question, decide which tools and parameters to use
2. **Act** — Call biological tools (enrichment, literature, gene info, database queries)
3. **Observe** — Interpret results, evaluate relevance, identify gaps
4. **Iterate** — Continue until the query is comprehensively answered, then synthesize

No fixed pipelines. If enrichment reveals cancer pathways, the model may look up gene functions next. If literature reveals a key paper, it may fetch full-text annotations. Pure LLM reasoning drives every decision.

## Tech Stack

- **Reasoning:** LangGraph ReAct agent with Gemini 2.5 Flash
- **Orchestration:** LangChain StructuredTools with Pydantic schemas
- **Biological APIs:** Enrichr (222 libraries), MyGene.info, Europe PMC
- **Visualizations:** Plotly, SciPy (hierarchical clustering), NetworkX
- **Interface:** Streamlit with human-in-the-loop feedback

## Project Structure

```
EnrichAI/
├── app.py                 # Streamlit UI and main application
├── reasoning_engine.py    # LangGraph ReAct agent and tool wrappers
├── tools.py               # Biological tool implementations (Enrichr, MyGene, Europe PMC)
├── visualizer.py          # 6 plot types with AI interpretation
├── config.py              # Model and logging configuration
├── tool_registry.py       # Enrichr library discovery
├── requirements.txt       # Python dependencies
└── assets/                # Icons, logos, figures
```

## License

MIT
