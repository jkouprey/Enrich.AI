# visualizer.py - Enhanced visualization with improved memory and error handling
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import networkx as nx
from itertools import combinations
import logging
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
from typing import Dict, List, Optional, Tuple

from config import CONFIG

logger = logging.getLogger(__name__)


# -------------------- PDF Export Utility --------------------

def export_figure_to_pdf(fig: go.Figure, filename: str = "plot.pdf", width: int = 1200, height: int = 800) -> Optional[
    bytes]:
    """
    Export a Plotly figure to PDF format.

    Parameters:
    -----------
    fig : go.Figure
        The Plotly figure to export
    filename : str
        Suggested filename for the download
    width : int
        Width of the exported image in pixels
    height : int
        Height of the exported image in pixels

    Returns:
    --------
    bytes or None
        PDF bytes if successful, None if export failed
    """
    try:
        # Try to use kaleido for PDF export
        pdf_bytes = fig.to_image(format="pdf", width=width, height=height, scale=2)
        return pdf_bytes
    except Exception as e:
        logger.warning(f"PDF export failed: {e}")
        return None


def add_pdf_download_button(fig: go.Figure, filename: str, key: str, width: int = None, height: int = None):
    """
    Add a PDF download button for a Plotly figure.

    Parameters:
    -----------
    fig : go.Figure
        The Plotly figure to export
    filename : str
        Filename for the downloaded PDF
    key : str
        Unique key for the Streamlit button
    width : int, optional
        Width override (uses figure's width if not specified)
    height : int, optional
        Height override (uses figure's height if not specified)
    """
    # Get dimensions from figure if not specified
    if width is None:
        width = fig.layout.width or 1200
    if height is None:
        height = fig.layout.height or 800

    pdf_bytes = export_figure_to_pdf(fig, filename, width=width, height=height)

    if pdf_bytes:
        st.download_button(
            label="📄️ Download as PDF",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            key=key
        )
    else:
        st.caption(" PDF export requires 'kaleido' package. Install with: `pip install kaleido`")



# -------------------- Utilities --------------------


@st.cache_data
def get_gemini_interpretation(_gemini_model, prompt: str, payload_json: str) -> str:
    """Get Gemini interpretation with caching"""
    if not _gemini_model:
        return "Gemini model not available."

    gemini_config = CONFIG.get("gemini", {})
    gen_config = {
        "temperature": gemini_config.get("temperature", 0.4),
        "max_output_tokens": gemini_config.get("max_output_tokens", 4096),
    }

    try:
        response = _gemini_model.generate_content(
            f"{prompt}\n\nDATA (JSON):\n{payload_json}",
            generation_config=gen_config
        )
        if hasattr(response, 'text'):
            return response.text
        return "Could not generate interpretation."
    except Exception as e:
        logger.error(f"Gemini interpretation failed: {e}")
        return f"Interpretation failed: {str(e)}"


# -------------------- Main Visualizer Class --------------------

class Visualizer:
    """Enhanced visualizer with improved memory and error handling"""

    def __init__(self, gemini_model=None):
        self.gemini_model = gemini_model
        # Color palettes
        self.COLOR_PALETTES = {
            "Viridis": px.colors.sequential.Viridis,
            "Plasma": px.colors.sequential.Plasma,
            "Inferno": px.colors.sequential.Inferno,
            "Magma": px.colors.sequential.Magma,
            "Cividis": px.colors.sequential.Cividis,
            "Blues": px.colors.sequential.Blues,
            "Reds": px.colors.sequential.Reds,
            "Greens": px.colors.sequential.Greens,
            "Rainbow": px.colors.sequential.Rainbow,
            "Turbo": px.colors.sequential.Turbo
        }

    # ── Plot interpretation ──

    PLOT_DESCRIPTIONS = {
        "bar": "Bar Plot showing -log10(p-value) significance. Longer bars = more significant enrichment.",
        "bubble": "Bubble Chart: X-axis is significance, bubble size is gene count. Larger rightward bubbles are key terms.",
        "dendrogram": "Hierarchical Clustering groups similar terms by gene overlap. Close branches share genes.",
        "upset": "UpSet Plot shows gene set intersections. Bars indicate shared genes between term combinations.",
        "similarity": "Similarity Clusters group terms by semantic and gene overlap similarity.",
        "cnetplot": "Concept Network shows term-gene relationships. Terms connected to their member genes.",
    }

    def get_plot_interpretation(self, plot_type: str, df: pd.DataFrame) -> str:
        """Generate AI interpretation for a specific plot type."""
        if not self.gemini_model:
            return "Enable Gemini API for interpretation."

        top_terms = (
            df.nsmallest(8, 'adjusted_p_value')['term'].tolist()
            if 'adjusted_p_value' in df.columns
            else df['term'].head(8).tolist()
        )

        desc = self.PLOT_DESCRIPTIONS.get(plot_type, "visualization")
        prompt = f"""This is a {desc}.

For these enrichment results with top terms: {top_terms[:5]}

Provide 2-3 sentences explaining what this plot reveals about this specific data."""

        gemini_config = CONFIG.get("gemini", {})
        gen_config = {
            "temperature": gemini_config.get("temperature", 0.4),
            "max_output_tokens": gemini_config.get("max_output_tokens", 4096),
        }

        try:
            response = self.gemini_model.generate_content(prompt, generation_config=gen_config)
            return response.text if hasattr(response, 'text') else "Interpretation unavailable."
        except Exception:
            return "Interpretation unavailable."


    def _compute_term_similarity_matrix(self, df: pd.DataFrame, n_terms: int = 30) -> Tuple[
        np.ndarray, List[str], pd.DataFrame]:
        """Compute Jaccard similarity matrix between terms based on gene overlap"""
        if 'adjusted_p_value' in df.columns:
            df_top = df.nsmallest(n_terms, 'adjusted_p_value').copy()
        else:
            df_top = df.head(n_terms).copy()

        terms = df_top['term'].tolist()
        n = len(terms)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            genes_i = set(df_top.iloc[i].get('genes', []) or [])
            for j in range(n):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    genes_j = set(df_top.iloc[j].get('genes', []) or [])
                    if genes_i and genes_j:
                        intersection = len(genes_i & genes_j)
                        union = len(genes_i | genes_j)
                        sim_matrix[i, j] = intersection / union if union > 0 else 0

        return sim_matrix, terms, df_top

    def _extract_keywords(self, terms: List[str], n_keywords: int = 10) -> List[Tuple[str, int]]:
        """Extract most frequent keywords from term names"""
        stopwords = {
            'of', 'the', 'to', 'in', 'and', 'a', 'an', 'by', 'for', 'with', 'from',
            'on', 'at', 'as', 'is', 'are', 'or', 'via', 'go', 'kegg', 'process',
            'pathway', 'activity', 'positive', 'negative', 'regulation', 'involved',
            'term', 'terms', 'bp', 'mf', 'cc', 'homo', 'sapiens', 'human'
        }

        word_freq = {}
        for term in terms:
            words = re.sub(r'[^\w\s]', ' ', term.lower()).split()
            for word in words:
                if len(word) > 2 and word not in stopwords and not word.isdigit():
                    word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:n_keywords]


    def render_dendrogram(self, df: pd.DataFrame, n_terms: int = 25, method: str = "ward",
                          font_size: int = 10, tree_scale: float = 0.3,
                          dot_size: int = 12) -> go.Figure:
        """
        Hierarchical Clustering Dendrogram with AUTOMATIC cluster detection.

        Clusters are determined by the hierarchical structure using distance threshold,
        NOT by a user-specified number.

        Layout (left to right):
        1. Tree (compressed via tree_scale)
        2. Colored cluster backgrounds
        3. Dots (colored by p-value, sized by gene count)
        4. Term labels
        5. Cluster name labels (right side)
        6. Legends (far right)

        Full interactivity enabled (zoom, pan, hover, etc.)
        """
        # Prepare data
        if 'adjusted_p_value' in df.columns:
            df_subset = df.nsmallest(n_terms, 'adjusted_p_value').copy()
        else:
            df_subset = df.head(n_terms).copy()

        if 'genes' not in df_subset.columns:
            return None

        df_subset['gene_count'] = df_subset['genes'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df_subset['-log10p'] = -np.log10(df_subset['adjusted_p_value'].replace(0, np.finfo(float).tiny))

        # Compute similarity matrix
        sim_matrix, terms, df_top = self._compute_term_similarity_matrix(df_subset, len(df_subset))

        if len(terms) < 3:
            return None

        n_terms_plot = len(terms)

        # Convert to distance and do hierarchical clustering
        dist_matrix = 1 - sim_matrix
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        condensed = squareform(dist_matrix, checks=False)
        condensed = np.nan_to_num(condensed, nan=0.5)
        condensed = np.clip(condensed, 0, 1)

        Z = linkage(condensed, method=method)

        # Get leaf order from dendrogram
        order = leaves_list(Z)

        # === CLUSTER BASED ON MAXCLUST - gives balanced leaf-based clusters ===
        # Automatically determine number of clusters based on data size
        # Target: ~5 terms per cluster on average
        target_clusters = max(2, min(8, n_terms_plot // 5))

        # Use fcluster with maxclust criterion - this cuts the tree to get exactly N clusters
        # This groups terms by their hierarchical similarity, giving balanced clusters
        clusters = fcluster(Z, t=target_clusters, criterion='maxclust')
        n_clusters_auto = len(set(clusters))

        # Reorder everything by dendrogram leaf order
        terms_ordered = [terms[i] for i in order]
        clusters_ordered = [clusters[i] for i in order]

        term_to_data = {row['term']: row for _, row in df_top.iterrows()}
        gene_counts = [term_to_data.get(t, {}).get('gene_count', 5) for t in terms_ordered]
        log_pvals = [term_to_data.get(t, {}).get('-log10p', 1) for t in terms_ordered]

        # Cluster colors (pastel)
        cluster_colors = [
            '#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFDFBA',
            '#E0BBE4', '#FEC8D8', '#D4F0F0', '#CCE2CB', '#B6CFB6',
            '#F0E68C', '#DDA0DD', '#87CEEB', '#98FB98', '#FFA07A'
        ]

        # Build cluster ranges - track CONTIGUOUS segments for each cluster
        # A cluster may have multiple non-contiguous segments due to dendrogram ordering
        cluster_segments = []  # List of (cluster_id, start_y, end_y, indices)

        current_cluster = clusters_ordered[0]
        segment_start = 0
        segment_indices = [0]

        for i in range(1, len(clusters_ordered)):
            if clusters_ordered[i] == current_cluster:
                # Same cluster, extend segment
                segment_indices.append(i)
            else:
                # Different cluster, save current segment and start new one
                cluster_segments.append({
                    'cluster_id': current_cluster,
                    'min_y': segment_start,
                    'max_y': i - 1,
                    'indices': segment_indices
                })
                current_cluster = clusters_ordered[i]
                segment_start = i
                segment_indices = [i]

        # Save final segment
        cluster_segments.append({
            'cluster_id': current_cluster,
            'min_y': segment_start,
            'max_y': len(clusters_ordered) - 1,
            'indices': segment_indices
        })

        # Count unique visual segments (this is what user sees)
        n_visual_clusters = len(cluster_segments)

        # === FIXED X POSITIONS (proportional to plot width) ===
        TREE_END = 25 * tree_scale  # Tree takes 0 to TREE_END
        DOT_X = 30  # Dots at fixed position
        LABEL_X = 35  # Labels start here
        CLUSTER_LABEL_X = 90  # Cluster names on right

        # Create figure
        fig = go.Figure()

        # === 1. CLUSTER BACKGROUND RECTANGLES (using segments) ===
        for seg in cluster_segments:
            c = seg['cluster_id']
            color = cluster_colors[(c - 1) % len(cluster_colors)]
            fig.add_shape(
                type='rect',
                x0=0, x1=CLUSTER_LABEL_X - 2,
                y0=seg['min_y'] - 0.45, y1=seg['max_y'] + 0.45,
                fillcolor=color,
                opacity=0.3,
                line=dict(width=1, color=color),
                layer='below'
            )

        # === 2. DENDROGRAM TREE ===
        dendro = dendrogram(Z, orientation='left', no_plot=True, labels=list(range(len(terms))))

        icoord = np.array(dendro['icoord'])
        dcoord = np.array(dendro['dcoord'])

        if len(icoord) > 0:
            # Scale y coordinates to match term positions (0 to n_terms-1)
            y_min, y_max = icoord.min(), icoord.max()
            y_range = y_max - y_min if y_max > y_min else 1

            # Scale x (distance) coordinates
            d_max = dcoord.max() if dcoord.max() > 0 else 1

            for i in range(len(icoord)):
                y_coords = [(y - y_min) / y_range * (n_terms_plot - 1) for y in icoord[i]]
                x_coords = [TREE_END * (1 - d / d_max) for d in dcoord[i]]

                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(color='#333333', width=1.2),
                    hoverinfo='skip',
                    showlegend=False
                ))

        # === 3. DOTS (colored by p-value, sized by gene count) ===
        max_genes = max(gene_counts) if gene_counts else 1
        min_genes = min(gene_counts) if gene_counts else 1

        node_sizes = [dot_size + 8 * ((g - min_genes) / max(1, max_genes - min_genes)) for g in gene_counts]

        fig.add_trace(go.Scatter(
            x=[DOT_X] * n_terms_plot,
            y=list(range(n_terms_plot)),
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=log_pvals,
                colorscale='Reds',
                cmin=min(log_pvals) if log_pvals else 0,
                cmax=max(log_pvals) if log_pvals else 5,
                colorbar=dict(
                    title=dict(text='-log10(p)', font=dict(size=font_size, color='black')),
                    x=1.02,
                    len=0.4,
                    y=0.8,
                    thickness=12,
                    tickfont=dict(size=font_size - 1, color='black')
                ),
                line=dict(width=0.5, color='black')
            ),
            hovertext=[f"{t}<br>Genes: {g}<br>-log10(p): {p:.2f}"
                       for t, g, p in zip(terms_ordered, gene_counts, log_pvals)],
            hoverinfo='text',
            showlegend=False
        ))

        # === 4. TERM LABELS ===
        for i, term in enumerate(terms_ordered):
            label = term[:50] + '...' if len(term) > 50 else term
            fig.add_annotation(
                x=LABEL_X, y=i,
                text=label,
                showarrow=False,
                font=dict(size=font_size, color='black'),
                xanchor='left',
                yanchor='middle'
            )

        # === 5. CLUSTER NAME LABELS (right side with connecting line) ===
        for seg in cluster_segments:
            c = seg['cluster_id']
            color = cluster_colors[(c - 1) % len(cluster_colors)]
            mid_y = (seg['min_y'] + seg['max_y']) / 2

            # Extract keywords for cluster name
            cluster_terms = [terms_ordered[i] for i in seg['indices']]
            keywords = self._extract_keywords(cluster_terms, 3)
            name = ' / '.join([w[0].title() for w in keywords[:2]]) if keywords else f"Cluster {c}"

            # Vertical line marking cluster extent
            fig.add_shape(
                type='line',
                x0=CLUSTER_LABEL_X - 5, x1=CLUSTER_LABEL_X - 5,
                y0=seg['min_y'] - 0.3, y1=seg['max_y'] + 0.3,
                line=dict(color=color, width=3)
            )

            # Cluster name label - ALL TEXT BLACK
            fig.add_annotation(
                x=CLUSTER_LABEL_X, y=mid_y,
                text=f"<b>{name}</b>",
                showarrow=False,
                font=dict(size=font_size, color='black'),
                bgcolor=color,
                bordercolor='black',
                borderwidth=1,
                borderpad=3,
                xanchor='left',
                yanchor='middle'
            )

        # === 6. SIZE LEGEND (Gene Count) - ALL TEXT BLACK ===
        legend_y_base = -2

        fig.add_annotation(
            x=DOT_X, y=legend_y_base,
            text="<b>Gene Count:</b>",
            showarrow=False,
            font=dict(size=font_size, color='black'),
            xanchor='left'
        )

        legend_values = [min_genes, (min_genes + max_genes) // 2, max_genes]
        legend_sizes_display = [dot_size, dot_size + 4, dot_size + 8]

        for i, (val, sz) in enumerate(zip(legend_values, legend_sizes_display)):
            x_pos = DOT_X + 15 + i * 12
            fig.add_trace(go.Scatter(
                x=[x_pos], y=[legend_y_base],
                mode='markers+text',
                marker=dict(size=sz, color='#d9534f', line=dict(width=0.5, color='black')),
                text=[str(val)],
                textposition='bottom center',
                textfont=dict(size=font_size - 1, color='black'),
                hoverinfo='skip',
                showlegend=False
            ))

        # === LAYOUT - FULL INTERACTIVITY ENABLED ===
        fig.update_layout(
            title=dict(
                text=f"Hierarchical Clustering Dendrogram ({n_visual_clusters} clusters shown)",
                font=dict(size=14, color='black')
            ),
            height=max(450, n_terms_plot * 22 + 80),
            width=950,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            margin=dict(l=20, r=120, t=50, b=60),
            # Full interactivity - removed fixedrange
            xaxis=dict(
                range=[-2, 105],
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False
            ),
            yaxis=dict(
                range=[-3.5, n_terms_plot - 0.5],
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False
            ),
            # Enable modebar with all tools
            modebar=dict(
                bgcolor='rgba(255,255,255,0.8)',
                color='black',
                activecolor='#2563eb'
            )
        )

        # Enable all interactive features
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'scrollZoom': True
        }

        return fig

    def render_dendrogram_with_clusters(self, df: pd.DataFrame, n_terms: int = 25, method: str = "ward",
                                        font_size: int = 10, tree_scale: float = 0.3,
                                        dot_size: int = 12) -> Tuple[Optional[go.Figure], Dict, Dict]:
        """
        Wrapper that returns both figure and cluster data for expanders.
        Clusters are automatically detected from hierarchical structure.
        """
        # Prepare data
        if 'adjusted_p_value' in df.columns:
            df_subset = df.nsmallest(n_terms, 'adjusted_p_value').copy()
        else:
            df_subset = df.head(n_terms).copy()

        if 'genes' not in df_subset.columns or len(df_subset) < 3:
            return None, {}, {}

        df_subset['gene_count'] = df_subset['genes'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df_subset['-log10p'] = -np.log10(df_subset['adjusted_p_value'].replace(0, np.finfo(float).tiny))

        # Compute similarity matrix
        sim_matrix, terms, df_top = self._compute_term_similarity_matrix(df_subset, len(df_subset))

        if len(terms) < 3:
            return None, {}, {}

        # Convert to distance
        dist_matrix = 1 - sim_matrix
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        condensed = squareform(dist_matrix, checks=False)
        condensed = np.nan_to_num(condensed, nan=0.5)
        condensed = np.clip(condensed, 0, 1)

        # Linkage and AUTOMATIC clustering
        Z = linkage(condensed, method=method)
        order = leaves_list(Z)

        # Automatic cluster detection using distance threshold
        max_dist = Z[:, 2].max() if len(Z) > 0 else 1
        threshold = max_dist * 0.7
        clusters = fcluster(Z, t=threshold, criterion='distance')

        # Reorder data according to dendrogram leaves
        terms_ordered = [terms[i] for i in order]
        clusters_ordered = [clusters[i] for i in order]

        # Build cluster_ranges
        cluster_ranges = {}
        for i, c in enumerate(clusters_ordered):
            if c not in cluster_ranges:
                cluster_ranges[c] = {'indices': []}
            cluster_ranges[c]['indices'].append(i)

        # Build cluster data for expanders with descriptive names
        cluster_term_data = {}
        cluster_names = {}
        for c, info in cluster_ranges.items():
            term_list = [terms_ordered[i] for i in info['indices']]
            cluster_term_data[c] = term_list

            # Generate descriptive name for cluster
            keywords = self._extract_keywords(term_list, n_keywords=3)
            if keywords:
                cluster_names[c] = " / ".join([kw[0].title() for kw in keywords[:2]])
            else:
                cluster_names[c] = f"Cluster {c}"

        # Render the figure using the main method
        fig = self.render_dendrogram(df, n_terms=n_terms, method=method,
                                     font_size=font_size, tree_scale=tree_scale, dot_size=dot_size)

        return fig, cluster_term_data, cluster_names

    # ========================================================================
    # FIXED ENRICHMENT MAP - Smaller nodes, smaller hulls, category names above
    # ========================================================================


    def render_upset_plot(self, df: pd.DataFrame, n_terms: int = 8,
                          bar_color: str = "#1a1a2e", dot_color: str = "#16213e",
                          line_color: str = "#888888", font_size: int = 10) -> go.Figure:
        """
        FIXED UpSet Plot:
        - Proper intersection calculation (more terms = more potential intersections)
        - Much smaller bars
        - Main emphasis on dot matrix
        - Increased figure height
        - All black letters

        Parameters:
        -----------
        line_color : str
            Color for connecting lines between dots (default grey "#888888")
        font_size : int
            Base font size for labels (default 10)
        """
        if 'adjusted_p_value' in df.columns:
            df_top = df.nsmallest(n_terms, 'adjusted_p_value').copy()
        else:
            df_top = df.head(n_terms).copy()

        # Get gene sets
        term_genes = {}
        for _, row in df_top.iterrows():
            genes = row.get('genes', []) or []
            if genes:
                term_name = row['term']
                # Wrap long names after ~20 characters
                if len(term_name) > 20:
                    words = term_name.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line) + len(word) + 1 <= 20:
                            current_line = current_line + " " + word if current_line else word
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)
                    term_name = "<br>".join(lines)
                term_genes[term_name] = set(genes)

        if len(term_genes) < 2:
            return None

        terms = list(term_genes.keys())

        # Compute ALL intersections properly
        # For each gene, find which terms contain it
        all_genes = set()
        for genes in term_genes.values():
            all_genes.update(genes)

        gene_memberships = {}
        for gene in all_genes:
            membership = frozenset([t for t in terms if gene in term_genes[t]])
            if membership not in gene_memberships:
                gene_memberships[membership] = []
            gene_memberships[membership].append(gene)

        # Convert to intersections list
        intersections = []
        for membership, genes in gene_memberships.items():
            if membership:  # Skip empty membership
                intersections.append({
                    'terms': membership,
                    'count': len(genes),
                    'genes': genes
                })

        # Sort by count and take top 25
        intersections.sort(key=lambda x: x['count'], reverse=True)
        intersections = intersections[:25]

        if not intersections:
            return None

        # Create figure with subplots - EMPHASIS ON DOTPLOT (larger ratio)
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.35, 0.65],  # Smaller bars, larger dotplot
            vertical_spacing=0.05,
            shared_xaxes=True
        )

        # === BAR CHART - SMALLER BARS ===
        x_pos = list(range(len(intersections)))
        counts = [inter['count'] for inter in intersections]

        fig.add_trace(
            go.Bar(
                x=x_pos,
                y=counts,
                marker_color=bar_color,
                width=0.3,  # Smaller bars
                hovertemplate='%{y} genes<extra></extra>'
            ),
            row=1, col=1
        )

        # === DOT MATRIX ===
        dot_inactive = '#e0e0e0'
        dot_size_active = 14
        dot_size_inactive = 5
        # Use the line_color parameter for connecting lines

        for i, inter in enumerate(intersections):
            for j, term in enumerate(terms):
                y_pos = j * 3.0  # MORE spacing between term rows

                if term in inter['terms']:
                    # Active dot
                    fig.add_trace(go.Scatter(
                        x=[i], y=[y_pos],
                        mode='markers',
                        marker=dict(size=dot_size_active, color=dot_color, line=dict(width=1, color='black')),
                        showlegend=False, hoverinfo='skip'
                    ), row=2, col=1)
                else:
                    # Inactive dot
                    fig.add_trace(go.Scatter(
                        x=[i], y=[y_pos],
                        mode='markers',
                        marker=dict(size=dot_size_inactive, color=dot_inactive),
                        showlegend=False, hoverinfo='skip'
                    ), row=2, col=1)

            # Connect active dots with GREY lines
            terms_in_inter = [t for t in terms if t in inter['terms']]
            if len(terms_in_inter) > 1:
                y_vals = sorted([terms.index(t) * 3.0 for t in terms_in_inter])
                fig.add_trace(go.Scatter(
                    x=[i, i],
                    y=[min(y_vals), max(y_vals)],
                    mode='lines',
                    line=dict(color=line_color, width=2),  # Customizable line color
                    showlegend=False, hoverinfo='skip'
                ), row=2, col=1)

        # Calculate dynamic height - INCREASED for more spacing
        matrix_height = max(400, len(terms) * 70)  # More height for spacing
        total_height = 250 + matrix_height

        fig.update_layout(
            title=dict(text="Gene Set Intersections (UpSet Plot)", font=dict(color='black', size=font_size + 4)),
            height=total_height,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=font_size),
            margin=dict(l=180)  # More space for term names
        )

        fig.update_yaxes(
            title_text="Intersection Size",
            row=1, col=1,
            tickfont=dict(color='black', size=font_size),
            title_font=dict(color='black', size=font_size)
        )
        fig.update_yaxes(
            ticktext=terms,
            tickvals=[j * 3.0 for j in range(len(terms))],  # Match new spacing
            tickfont=dict(size=font_size, color='black'),
            title_text="",
            row=2, col=1,
            range=[-1.5, (len(terms) - 1) * 3.0 + 1.5]  # Adjusted range
        )
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=2, col=1)

        return fig

    # ========================================================================
    # FIXED SIMILARITY CLUSTERS - SimplifyEnrichment style with word clouds
    # ========================================================================

    def render_similarity_clusters(self, df: pd.DataFrame, n_terms: int = 50, n_clusters: int = 8,
                                   width: int = 1200, height: int = 650,
                                   word_font_size: int = 12,
                                   heatmap_palette: str = "Reds") -> go.Figure:
        """
        FIXED Similarity Clusters (SimplifyEnrichment style):
        - Smaller heatmap on left
        - Colored cluster bar
        - Linking shapes to word cloud boxes
        - Word cloud boxes with grey background and sized keywords

        Parameters:
        -----------
        word_font_size : int
            Base font size for keywords inside word cloud boxes (default 12)
        heatmap_palette : str
            Color palette for the similarity heatmap (default "Reds")
            Options: "Viridis", "Reds", "Blues", "Greens", "YlOrRd", "RdBu", "Plasma", etc.
        """
        sim_matrix, terms, df_top = self._compute_term_similarity_matrix(df, n_terms)

        if len(terms) < 3:
            return None

        # Cluster terms
        dist = 1 - sim_matrix
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2
        condensed = squareform(dist, checks=False)
        condensed = np.nan_to_num(condensed, nan=0.5)
        condensed = np.clip(condensed, 0, 1)

        Z = linkage(condensed, method='ward')
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
        order = leaves_list(Z)

        # Reorder matrix and data
        sim_ordered = sim_matrix[order][:, order]
        terms_ordered = [terms[i] for i in order]
        clusters_ordered = [clusters[i] for i in order]

        # Muted colors for clusters
        muted_colors = [
            '#c44e52', '#55a868', '#4c72b0', '#8172b3', '#ccb974',
            '#64b5cd', '#dd8452', '#da8bc3', '#8c8c8c', '#937860',
            '#6acc64', '#d65f5f', '#6b8ba4', '#a079bf', '#b5a14a'
        ]

        # Group terms by cluster with their indices
        cluster_term_map = {}
        for i, (term, c) in enumerate(zip(terms_ordered, clusters_ordered)):
            if c not in cluster_term_map:
                cluster_term_map[c] = {'terms': [], 'indices': [], 'pvalues': [], 'original_indices': []}
            cluster_term_map[c]['terms'].append(term)
            cluster_term_map[c]['indices'].append(i)
            cluster_term_map[c]['original_indices'].append(i)  # Keep original for colorbar/trapezoid
            # Get p-value for this term
            pval = df_top[df_top['term'] == term]['adjusted_p_value'].values
            cluster_term_map[c]['pvalues'].append(pval[0] if len(pval) > 0 else 1.0)

        # Store original cluster boundaries BEFORE filtering
        cluster_boundaries = {}
        for c, data in cluster_term_map.items():
            cluster_boundaries[c] = {
                'y_min': min(data['original_indices']) - 0.5,
                'y_max': max(data['original_indices']) + 0.5
            }

        # LIMIT CLUSTERS TO TOP 10 TERMS BY P-VALUE (for word cloud only)
        for c in cluster_term_map:
            data = cluster_term_map[c]
            if len(data['terms']) > 10:
                # Sort by p-value and keep top 10
                sorted_indices = np.argsort(data['pvalues'])[:10]
                data['terms'] = [data['terms'][i] for i in sorted_indices]
                data['indices'] = [data['indices'][i] for i in sorted_indices]
                data['pvalues'] = [data['pvalues'][i] for i in sorted_indices]

        n = len(terms_ordered)

        # Create figure with custom domain layout
        fig = go.Figure()

        # Define layout regions (in paper coordinates 0-1)
        heatmap_left = 0.0
        heatmap_right = 0.35  # Smaller heatmap
        colorbar_left = 0.36
        colorbar_right = 0.38
        wordcloud_left = 0.35
        wordcloud_right = 0.98

        # === HEATMAP ===
        hover_text = [
            [f"{terms_ordered[i][:30]}<br>{terms_ordered[j][:30]}<br>Similarity: {sim_ordered[i, j]:.3f}"
             for j in range(n)] for i in range(n)]

        fig.add_trace(
            go.Heatmap(
                z=sim_ordered,
                x=list(range(n)),
                y=list(range(n)),
                colorscale=heatmap_palette,
                showscale=True,
                colorbar=dict(
                    title=dict(text='Similarity', font=dict(size=12, color='black')),
                    x=1.02,
                    len=0.3,
                    y=0.85,
                    thickness=10,
                    tickfont=dict(color='black', size=10)
                ),
                hovertemplate='%{text}<extra></extra>',
                text=hover_text,
                xaxis='x',
                yaxis='y'
            )
        )

        # Add cluster boundary lines on heatmap
        prev_cluster = clusters_ordered[0]
        for i, c in enumerate(clusters_ordered):
            if c != prev_cluster:
                fig.add_shape(
                    type='line',
                    x0=-0.5, x1=n - 0.5,
                    y0=i - 0.5, y1=i - 0.5,
                    line=dict(color='black', width=1),
                    xref='x', yref='y'
                )
                fig.add_shape(
                    type='line',
                    x0=i - 0.5, x1=i - 0.5,
                    y0=-0.5, y1=n - 0.5,
                    line=dict(color='black', width=1),
                    xref='x', yref='y'
                )
                prev_cluster = c

        # === CLUSTER COLOR BAR (right of heatmap) ===
        for c, data in cluster_term_map.items():
            color = muted_colors[(c - 1) % len(muted_colors)]

            # Use ORIGINAL cluster boundaries (before top-10 filtering)
            y_min = cluster_boundaries[c]['y_min']
            y_max = cluster_boundaries[c]['y_max']

            # Colored rectangle for this cluster
            fig.add_shape(
                type='rect',
                x0=0, x1=1,
                y0=y_min, y1=y_max,
                fillcolor=color,
                line=dict(color=color, width=0),
                xref='x2', yref='y2'
            )

        # === WORD CLOUD BOXES WITH LINKS (SimplifyEnrichment style) ===
        # Sort clusters by their vertical position for better layout
        sorted_clusters = sorted(cluster_term_map.items(), key=lambda x: np.mean(x[1]['indices']))

        def hex_to_rgb(hex_color):
            """Convert hex to RGB tuple"""
            h = hex_color.lstrip('#')
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

        def darken_color(hex_color, factor):
            """Darken a color by factor (0=black, 1=original)"""
            r, g, b = hex_to_rgb(hex_color)
            new_r = int(r * factor)
            new_g = int(g * factor)
            new_b = int(b * factor)
            return f'rgb({new_r},{new_g},{new_b})'

        for c, data in sorted_clusters:
            color = muted_colors[(c - 1) % len(muted_colors)]

            # Use ORIGINAL cluster boundaries (before top-10 filtering)
            y_min_idx = cluster_boundaries[c]['y_min'] + 0.5  # Convert back from -0.5 adjustment
            y_max_idx = cluster_boundaries[c]['y_max'] - 0.5  # Convert back from +0.5 adjustment
            y_center = (y_min_idx + y_max_idx) / 2
            cluster_height = y_max_idx - y_min_idx + 1

            # Word cloud box position - bigger, positioned right
            box_y_center = y_center
            box_height = max(cluster_height * 0.7, 3.0)  # Even bigger boxes
            box_y_min = box_y_center - box_height / 2
            box_y_max = box_y_center + box_height / 2

            # Box x position - a bit more to the right
            box_x_start = 0.12
            box_x_end = 0.52

            # Extract keywords with frequencies
            all_words = {}
            for term in data['terms']:
                words = self._tokenize_term(term)
                for word in words:
                    if len(word) > 2:
                        all_words[word] = all_words.get(word, 0) + 1

            sorted_words = sorted(all_words.items(), key=lambda x: x[1], reverse=True)[:5]

            if not sorted_words:
                continue

            max_freq = sorted_words[0][1] if sorted_words else 1
            min_freq = sorted_words[-1][1] if sorted_words else 1

            # Draw grey background box
            fig.add_shape(
                type='rect',
                x0=box_x_start, x1=box_x_end,
                y0=box_y_min, y1=box_y_max,
                fillcolor='#DDDDDD',
                line=dict(color='#AAAAAA', width=1),
                xref='x3', yref='y3'
            )

            # Draw connecting trapezoid - starts MORE TO THE RIGHT and matches cluster height exactly
            # Left side (near heatmap): full cluster height
            # Right side (at box): tapered to small point
            trap_x_left = 0.08  # More to the right to avoid overlapping colorbar
            trap_x_right = box_x_start  # Connect to box edge

            # Left side matches ORIGINAL cluster height exactly
            trap_y_left_min = cluster_boundaries[c]['y_min']
            trap_y_left_max = cluster_boundaries[c]['y_max']

            # Right side tapered to center
            taper_factor = 0.15
            trap_half_height = cluster_height * taper_factor / 2
            trap_y_right_min = y_center - trap_half_height
            trap_y_right_max = y_center + trap_half_height

            link_path = f"M {trap_x_left},{trap_y_left_min} L {trap_x_right},{trap_y_right_min} L {trap_x_right},{trap_y_right_max} L {trap_x_left},{trap_y_left_max} Z"
            fig.add_shape(
                type='path',
                path=link_path,
                fillcolor='#DDDDDD',
                line=dict(color='#AAAAAA', width=1),
                xref='x3', yref='y3'
            )

            # Add colored line at the LEFT edge of trapezoid - matches ORIGINAL cluster height exactly
            fig.add_shape(
                type='line',
                x0=trap_x_left, x1=trap_x_left,
                y0=cluster_boundaries[c]['y_min'], y1=cluster_boundaries[c]['y_max'],
                line=dict(color=color, width=4),
                xref='x3', yref='y3'
            )

            # Add words inside the box with varying sizes AND colors
            n_words = len(sorted_words)
            if n_words > 0:
                # Arrange words in compact layout
                rows_needed = min(n_words, 3)
                y_spacing = (box_y_max - box_y_min) / (rows_needed + 1)

                word_idx = 0
                for row in range(rows_needed):
                    # Alternate between 1 and 2 words per row
                    words_in_row = 2 if row % 2 == 0 and word_idx + 1 < n_words else 1

                    for col in range(words_in_row):
                        if word_idx >= n_words:
                            break

                        word, freq = sorted_words[word_idx]
                        word_idx += 1

                        # Position within smaller box
                        box_center_x = (box_x_start + box_x_end) / 2
                        box_width = box_x_end - box_x_start
                        if words_in_row == 1:
                            word_x = box_center_x
                        else:
                            word_x = box_center_x - box_width * 0.22 if col == 0 else box_center_x + box_width * 0.22

                        word_y = box_y_max - (row + 1) * y_spacing

                        # Calculate darkness factor (0.4-1.0 range)
                        # Highest freq = 1.0 (full cluster color), lowest = 0.4 (darker version)
                        if max_freq > min_freq:
                            darkness = 0.4 + 0.6 * ((freq - min_freq) / (max_freq - min_freq))
                        else:
                            darkness = 1.0

                        # Font size based on frequency (scaled from word_font_size parameter)
                        # Range: word_font_size to word_font_size + 6
                        font_size_offset = int(6 * ((freq - min_freq) / max(1, max_freq - min_freq)))
                        font_size = word_font_size + font_size_offset

                        # Color: use cluster color directly, darken for less frequent
                        word_color = darken_color(color, darkness)

                        fig.add_annotation(
                            x=word_x,
                            y=word_y,
                            xref='x3',
                            yref='y3',
                            text=f"<b>{word}</b>",
                            showarrow=False,
                            font=dict(size=font_size, color=word_color),
                            xanchor='center',
                            yanchor='middle'
                        )

        # Layout with multiple axes
        fig.update_layout(
            title=dict(
                text=f"Semantic Similarity Clustering ({n} terms, {n_clusters} clusters)",
                font=dict(color='black', size=16),
                x=0.5
            ),
            height=height,
            width=width,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            showlegend=False,
            margin=dict(t=60, b=30, l=20, r=20),  # Reduced bottom margin

            # Heatmap axis (left)
            xaxis=dict(
                domain=[heatmap_left, heatmap_right],
                showticklabels=False, showgrid=False, zeroline=False
            ),
            yaxis=dict(
                domain=[0.05, 0.95],
                showticklabels=False, showgrid=False, zeroline=False,
                autorange='reversed'
            ),

            # Color bar axis (thin strip)
            xaxis2=dict(
                domain=[colorbar_left, colorbar_right],
                showticklabels=False, showgrid=False, zeroline=False,
                range=[0, 1], anchor='y2'
            ),
            yaxis2=dict(
                domain=[0.05, 0.95],
                showticklabels=False, showgrid=False, zeroline=False,
                range=[-0.5, n - 0.5], autorange='reversed', anchor='x2'
            ),

            # Word cloud panel axis
            xaxis3=dict(
                domain=[wordcloud_left, wordcloud_right],
                showticklabels=False, showgrid=False, zeroline=False,
                range=[0, 1], anchor='y3'
            ),
            yaxis3=dict(
                domain=[0.05, 0.95],
                showticklabels=False, showgrid=False, zeroline=False,
                range=[-0.5, n - 0.5], autorange='reversed', anchor='x3'
            )
        )

        return fig

    def render_similarity_clusters_with_data(self, df: pd.DataFrame, n_terms: int = 50, n_clusters: int = 8,
                                             width: int = 1100, height: int = 900,
                                             word_font_size: int = 12,
                                             heatmap_palette: str = "Reds") -> Tuple[Optional[go.Figure], Dict]:
        """
        Wrapper that returns both figure and cluster data for expanders.

        Parameters:
        -----------
        word_font_size : int
            Base font size for keywords inside word cloud boxes (default 12)
        heatmap_palette : str
            Color palette for similarity heatmap (default "Reds")
        """
        sim_matrix, terms, df_top = self._compute_term_similarity_matrix(df, n_terms)

        if len(terms) < 3:
            return None, {}

        # Cluster terms
        dist = 1 - sim_matrix
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2
        condensed = squareform(dist, checks=False)
        condensed = np.nan_to_num(condensed, nan=0.5)
        condensed = np.clip(condensed, 0, 1)

        Z = linkage(condensed, method='ward')
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
        order = leaves_list(Z)

        terms_ordered = [terms[i] for i in order]
        clusters_ordered = [clusters[i] for i in order]

        # Build cluster data for expanders with descriptive names
        cluster_term_data = {}
        cluster_names = {}
        for i, (term, c) in enumerate(zip(terms_ordered, clusters_ordered)):
            if c not in cluster_term_data:
                cluster_term_data[c] = []
            cluster_term_data[c].append(term)

        # Generate descriptive names for each cluster
        for c, term_list in cluster_term_data.items():
            keywords = self._extract_keywords(term_list, n_keywords=3)
            if keywords:
                cluster_names[c] = " / ".join([kw[0].title() for kw in keywords[:2]])
            else:
                cluster_names[c] = f"Cluster {c}"

        # Render the figure with heatmap_palette
        fig = self.render_similarity_clusters(df, n_terms=n_terms, n_clusters=n_clusters,
                                              width=width, height=height, word_font_size=word_font_size,
                                              heatmap_palette=heatmap_palette)

        return fig, cluster_term_data, cluster_names

    def _tokenize_term(self, term: str) -> List[str]:
        """Tokenize a GO/pathway term into meaningful words"""
        import re
        # Remove GO IDs, parenthetical notes, etc.
        term = re.sub(r'\(GO:\d+\)', '', term)
        term = re.sub(r'GO:\d+', '', term)
        term = re.sub(r'\([^)]*\)', '', term)

        # Split and clean
        words = re.split(r'[\s,_\-/]+', term.lower())

        # Filter out stop words and short words
        stop_words = {'of', 'the', 'and', 'or', 'in', 'to', 'by', 'a', 'an', 'is', 'are',
                      'with', 'from', 'for', 'on', 'at', 'as', 'via', 'into', 'etc',
                      'process', 'regulation', 'positive', 'negative', 'activity', 'involved'}

        return [w for w in words if w and len(w) > 2 and w not in stop_words]

    # ========================================================================
    # FIXED CNETPLOT - Black letters, triangles for clusters, grouped colors
    # ========================================================================


    def render_cnetplot(self, df: pd.DataFrame, n_terms: int = 25, n_clusters: int = 5,
                        show_genes: bool = True, max_genes: int = 50, font_size: int = 10,
                        show_labels: bool = True) -> go.Figure:
        """
        FIXED Concept Network Plot:
        - All black letters
        - Triangle markers for cluster nodes (not circles)
        - Same color nodes grouped closer together
        - max_genes: maximum genes to show per cluster
        """
        sim_matrix, terms, df_top = self._compute_term_similarity_matrix(df, n_terms)

        if len(terms) < 3:
            return None

        # Cluster terms
        dist = 1 - sim_matrix
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2
        condensed = squareform(dist, checks=False)
        condensed = np.nan_to_num(condensed, nan=0.5)
        Z = linkage(condensed, method='ward')
        clusters = fcluster(Z, n_clusters, criterion='maxclust')

        # Build cluster info
        cluster_info = {}
        term_to_cluster = {}
        term_to_data = {row['term']: row for _, row in df_top.iterrows()}

        for term, c in zip(terms, clusters):
            if c not in cluster_info:
                cluster_info[c] = {'terms': [], 'all_genes': set()}
            cluster_info[c]['terms'].append(term)
            term_to_cluster[term] = c

            if term in term_to_data:
                genes = term_to_data[term].get('genes', []) or []
                cluster_info[c]['all_genes'].update(genes)

        # Get keywords for cluster names
        # Limit genes per cluster based on max_genes
        genes_per_cluster = max(1, max_genes // max(len(cluster_info), 1))
        for c in cluster_info:
            keywords = self._extract_keywords(cluster_info[c]['terms'], 3)
            cluster_info[c]['name'] = ' '.join([w for w, _ in keywords]) if keywords else f"Cluster {c}"
            cluster_info[c]['top_genes'] = list(cluster_info[c]['all_genes'])[:genes_per_cluster]

        # Create network
        G = nx.Graph()

        cluster_colors = px.colors.qualitative.Set1

        # Add cluster center nodes
        for c, info in cluster_info.items():
            G.add_node(
                f"cluster_{c}",
                node_type='cluster',
                name=info['name'],
                size=50,
                color=cluster_colors[(c - 1) % len(cluster_colors)],
                cluster=c,
                n_terms=len(info['terms'])
            )

        # Add term nodes
        for term in terms:
            c = term_to_cluster[term]
            row = term_to_data.get(term, {})
            pval = row.get('adjusted_p_value', 0.01) if isinstance(row, dict) else 0.01
            gene_count = len(row.get('genes', []) or []) if isinstance(row, dict) else 0

            G.add_node(
                term,
                node_type='term',
                cluster=c,
                pval=pval,
                gene_count=gene_count,
                color=cluster_colors[(c - 1) % len(cluster_colors)]
            )
            G.add_edge(f"cluster_{c}", term)

        # Add gene nodes
        if show_genes:
            for c, info in cluster_info.items():
                for gene in info['top_genes']:
                    G.add_node(
                        f"gene_{gene}",
                        node_type='gene',
                        name=gene,
                        cluster=c,
                        color=cluster_colors[(c - 1) % len(cluster_colors)]
                    )
                    G.add_edge(f"cluster_{c}", f"gene_{gene}")

        # Custom layout - group same-color nodes together
        # First, position cluster centers
        n_clusters_actual = len(cluster_info)
        cluster_positions = {}
        for i, c in enumerate(cluster_info.keys()):
            angle = 2 * np.pi * i / n_clusters_actual
            cluster_positions[f"cluster_{c}"] = (2 * np.cos(angle), 2 * np.sin(angle))

        # Position nodes around their cluster
        pos = {}
        for c, info in cluster_info.items():
            center = cluster_positions[f"cluster_{c}"]
            pos[f"cluster_{c}"] = center

            # Position terms around cluster
            n_nodes = len(info['terms']) + len(info['top_genes']) if show_genes else len(info['terms'])
            for j, term in enumerate(info['terms']):
                angle = 2 * np.pi * j / max(n_nodes, 1)
                radius = 0.8
                pos[term] = (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))

            # Position genes
            if show_genes:
                for k, gene in enumerate(info['top_genes']):
                    angle = 2 * np.pi * (len(info['terms']) + k) / max(n_nodes, 1)
                    radius = 1.0
                    pos[f"gene_{gene}"] = (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))

        # Apply spring layout to fine-tune but keep initial positions
        pos = nx.spring_layout(G, pos=pos, k=0.5, iterations=30, seed=42)

        fig = go.Figure()

        # Draw edges
        edge_x, edge_y = [], []
        for e in G.edges():
            x0, y0 = pos[e[0]]
            x1, y1 = pos[e[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='rgba(150,150,150,0.3)'),
            hoverinfo='skip',
            showlegend=False
        ))

        # Draw nodes by type
        # Gene nodes
        gene_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'gene']
        if gene_nodes:
            fig.add_trace(go.Scatter(
                x=[pos[n][0] for n in gene_nodes],
                y=[pos[n][1] for n in gene_nodes],
                mode='markers+text' if show_labels else 'markers',
                marker=dict(
                    size=12,
                    color=[G.nodes[n]['color'] for n in gene_nodes],
                    symbol='circle',
                    line=dict(width=1, color='white')
                ),
                text=[G.nodes[n]['name'] for n in gene_nodes],
                textposition='top center',
                textfont=dict(size=font_size, color='black'),
                hovertemplate='<b>%{text}</b><extra></extra>',
                name='Genes',
                showlegend=True
            ))

        # Term nodes - RECTANGLES (squares)
        term_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'term']
        if term_nodes:
            max_gc = max([G.nodes[n].get('gene_count', 1) for n in term_nodes]) or 1
            term_sizes = [max(10, min(25, 10 + 15 * (G.nodes[n].get('gene_count', 1) / max_gc))) for n in term_nodes]

            fig.add_trace(go.Scatter(
                x=[pos[n][0] for n in term_nodes],
                y=[pos[n][1] for n in term_nodes],
                mode='markers+text' if show_labels else 'markers',
                marker=dict(
                    size=term_sizes,
                    color=[G.nodes[n]['color'] for n in term_nodes],
                    symbol='square',  # RECTANGLES (squares)
                    line=dict(width=1, color='white')
                ),
                text=[n[:25] + '...' if len(n) > 25 else n for n in term_nodes] if show_labels else None,
                textposition='bottom center',
                textfont=dict(size=font_size - 1, color='black'),
                hovertemplate='<b>%{customdata}</b><extra></extra>',
                customdata=[n[:30] for n in term_nodes],
                name='Terms',
                showlegend=True
            ))

        # Cluster nodes - TRIANGLES
        cluster_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'cluster']
        if cluster_nodes:
            fig.add_trace(go.Scatter(
                x=[pos[n][0] for n in cluster_nodes],
                y=[pos[n][1] for n in cluster_nodes],
                mode='markers+text',
                marker=dict(
                    size=35,
                    color=[G.nodes[n]['color'] for n in cluster_nodes],
                    symbol='triangle-up',  # TRIANGLES
                    line=dict(width=2, color='black')
                ),
                text=[G.nodes[n]['name'] for n in cluster_nodes],
                textposition='middle center',
                textfont=dict(size=font_size + 1, color='black'),  # BLACK text, scaled by font_size
                hovertemplate='<b>%{text}</b><br>%{customdata} terms<extra></extra>',
                customdata=[G.nodes[n]['n_terms'] for n in cluster_nodes],
                name='Clusters',
                showlegend=True
            ))

        fig.update_layout(
            title=dict(text="Concept Network", font=dict(color='black')),
            height=700,
            showlegend=True,
            legend=dict(x=1.02, y=1, font=dict(color='black')),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black')
        )

        return fig

    def render_cnetplot_with_clusters(self, df: pd.DataFrame, n_terms: int = 25, n_clusters: int = 5,
                                      max_genes: int = 50, font_size: int = 10,
                                      show_labels: bool = True) -> Tuple[Optional[go.Figure], Dict]:
        """
        Wrapper that returns both figure and cluster data for expanders.
        """
        sim_matrix, terms, df_top = self._compute_term_similarity_matrix(df, n_terms)

        if len(terms) < 3:
            return None, {}

        # Cluster terms
        dist = 1 - sim_matrix
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2
        condensed = squareform(dist, checks=False)
        condensed = np.nan_to_num(condensed, nan=0.5)
        Z = linkage(condensed, method='ward')
        clusters = fcluster(Z, n_clusters, criterion='maxclust')

        # Build cluster data for expanders with descriptive names
        cluster_term_data = {}
        cluster_names = {}
        for term, c in zip(terms, clusters):
            if c not in cluster_term_data:
                cluster_term_data[c] = []
            cluster_term_data[c].append(term)

        # Generate descriptive names for each cluster
        for c, term_list in cluster_term_data.items():
            keywords = self._extract_keywords(term_list, n_keywords=3)
            if keywords:
                cluster_names[c] = " / ".join([kw[0].title() for kw in keywords[:2]])
            else:
                cluster_names[c] = f"Cluster {c}"

        # Render the figure
        fig = self.render_cnetplot(df, n_terms=n_terms, n_clusters=n_clusters, show_genes=True,
                                   max_genes=max_genes, font_size=font_size, show_labels=show_labels)

        return fig, cluster_term_data, cluster_names
