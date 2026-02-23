"""
Interaction Graph Builder
Constructs directed graphs from interaction traces and extracts features
for phishing detection via graph-based analysis.
"""

import json
import math
import logging
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class GraphFeatures:
    """Features extracted from interaction graph for classification."""
    # Topological features
    num_nodes: int = 0
    num_edges: int = 0
    avg_degree: float = 0.0
    max_degree: int = 0
    graph_density: float = 0.0
    clustering_coefficient: float = 0.0
    num_connected_components: int = 0
    avg_path_length: float = 0.0
    diameter: int = 0
    
    # Behavioral features
    num_forms: int = 0
    num_redirects: int = 0
    has_password_input: bool = False
    has_email_input: bool = False
    external_redirects: int = 0
    num_input_fields: int = 0
    num_buttons: int = 0
    num_links: int = 0
    max_redirect_depth: int = 0
    form_to_node_ratio: float = 0.0
    
    # Temporal features
    total_interaction_time: float = 0.0
    avg_time_between_events: float = 0.0
    event_frequency: float = 0.0
    
    # Advanced topological
    betweenness_centrality_max: float = 0.0
    closeness_centrality_max: float = 0.0
    pagerank_max: float = 0.0
    in_degree_max: int = 0
    out_degree_max: int = 0
    
    # Sensitive data indicators
    requests_sensitive_data: bool = False
    credential_fields_count: int = 0
    financial_fields_count: int = 0


class InteractionGraphBuilder:
    """Builds and analyzes interaction graphs from traces."""

    def __init__(self):
        self.graph: Optional[nx.DiGraph] = None

    def build_graph_from_trace(self, trace_path: str) -> nx.DiGraph:
        """Build directed graph from trace JSON file."""
        with open(trace_path, 'r', encoding='utf-8') as f:
            trace_data = json.load(f)
        return self.build_graph_from_dict(trace_data)

    @staticmethod
    def filter_interaction_events(events: list) -> list:
        """Filter out screenshot/page events, keeping only real interactions."""
        return [
            e for e in events
            if e and e.get('event_type') != 'screenshot'
            and e.get('element_tag') != 'page'
        ]

    def build_graph_from_dict(self, trace_data: Dict) -> nx.DiGraph:
        """Build directed graph from trace dictionary.
        Screenshot/page events are filtered out (they are visual bookmarks,
        not interactions)."""
        G = nx.DiGraph()

        root_url = trace_data.get('url', 'unknown')
        G.add_node(root_url, node_type='url', is_root=True, label='root')

        previous_node = root_url
        raw_events = trace_data.get('events', [])
        events = self.filter_interaction_events(raw_events)
        
        for idx, event in enumerate(events):
            if event is None:
                continue
            event_type = event.get('event_type', 'unknown')
            el_tag = event.get('element_tag', '')
            el_id = event.get('element_id', '')
            el_class = event.get('element_class', '')
            url_before = event.get('url_before', '')
            url_after = event.get('url_after', '')
            timestamp = event.get('timestamp', 0)
            form_data = event.get('form_data', {})

            # Create event node
            node_id = f"{event_type}_{idx}"
            node_attrs = {
                'node_type': event_type,
                'element_tag': el_tag,
                'element_id': el_id,
                'element_class': el_class,
                'timestamp': timestamp,
                'label': f"{event_type}:{el_tag}",
            }

            # Add form data info if present
            if form_data:
                field_names = list(form_data.keys()) if isinstance(form_data, dict) else []
                node_attrs['form_fields'] = field_names
                node_attrs['has_sensitive'] = any(
                    k in ' '.join(field_names).lower()
                    for k in ['password', 'pass', 'pwd', 'email', 'ssn', 'card', 'credit']
                )

            G.add_node(node_id, **node_attrs)
            G.add_edge(previous_node, node_id, edge_type='sequence', weight=1.0)

            # URL transition edge
            if url_after and url_before and url_after != url_before:
                if url_after not in G.nodes():
                    G.add_node(url_after, node_type='url', is_root=False, label='redirect')
                G.add_edge(node_id, url_after, edge_type='navigation', weight=2.0)
                previous_node = url_after
            else:
                previous_node = node_id

        # Add redirect chain from trace
        redirects = trace_data.get('redirects', [])
        for i in range(len(redirects) - 1):
            if redirects[i] not in G:
                G.add_node(redirects[i], node_type='url', is_root=False, label='redirect')
            if redirects[i + 1] not in G:
                G.add_node(redirects[i + 1], node_type='url', is_root=False, label='redirect')
            G.add_edge(redirects[i], redirects[i + 1], edge_type='redirect', weight=3.0)

        self.graph = G
        return G

    def extract_features(self, graph: Optional[nx.DiGraph] = None) -> GraphFeatures:
        """Extract comprehensive features from interaction graph."""
        if graph is None:
            graph = self.graph
        if graph is None or graph.number_of_nodes() == 0:
            return GraphFeatures()

        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        
        # Basic topology
        degrees = [d for _, d in graph.degree()]
        in_degrees = [d for _, d in graph.in_degree()]
        out_degrees = [d for _, d in graph.out_degree()]
        avg_degree = sum(degrees) / n if n > 0 else 0
        max_deg = max(degrees) if degrees else 0
        density = nx.density(graph)

        # Clustering (on undirected version)
        undirected = graph.to_undirected()
        clustering = nx.average_clustering(undirected) if n > 2 else 0

        # Connected components
        n_components = nx.number_weakly_connected_components(graph)

        # Path metrics (on largest weakly connected component)
        avg_path = 0.0
        diameter = 0
        try:
            largest_cc = max(nx.weakly_connected_components(graph), key=len)
            sg = graph.subgraph(largest_cc)
            if sg.number_of_nodes() > 1:
                # Use undirected for path metrics
                usg = sg.to_undirected()
                if nx.is_connected(usg):
                    avg_path = nx.average_shortest_path_length(usg)
                    diameter = nx.diameter(usg)
        except (nx.NetworkXError, ValueError) as e:
            logger.debug(f"Path metrics computation skipped: {e}")

        # Centrality measures
        bc = nx.betweenness_centrality(graph) if n > 2 else {}
        cc = nx.closeness_centrality(graph) if n > 0 else {}
        pr = nx.pagerank(graph) if n > 0 else {}
        bc_max = max(bc.values()) if bc else 0
        cc_max = max(cc.values()) if cc else 0
        pr_max = max(pr.values()) if pr else 0

        # Behavioral counts
        event_types = Counter()
        has_password = False
        has_email = False
        num_input_fields = 0
        num_buttons = 0
        num_links = 0
        credential_count = 0
        financial_count = 0

        for node, data in graph.nodes(data=True):
            ntype = data.get('node_type', '')
            el_tag = data.get('element_tag', '').lower()
            el_id = str(data.get('element_id', '')).lower()
            el_class = str(data.get('element_class', '')).lower()
            combined = f"{el_tag} {el_id} {el_class}"
            
            event_types[ntype] += 1
            
            if 'password' in combined or 'pwd' in combined:
                has_password = True
                credential_count += 1
            if 'email' in combined or 'mail' in combined:
                has_email = True
                credential_count += 1
            if 'ssn' in combined or 'social' in combined:
                credential_count += 1
            if 'card' in combined or 'credit' in combined or 'cvv' in combined:
                financial_count += 1
            
            if ntype in ('input', 'form_input'):
                num_input_fields += 1
            elif ntype in ('button_detected',):
                num_buttons += 1
            elif ntype in ('link_detected',):
                num_links += 1

        # Count all form submission types (submit, js_form_submit, dual_submit_*)
        num_forms = sum(v for k, v in event_types.items()
                        if 'submit' in k)
        url_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'url']
        num_redirects = len(url_nodes) - 1 if len(url_nodes) > 0 else 0
        
        # External redirects
        from urllib.parse import urlparse
        domains = set()
        for u in url_nodes:
            try:
                d = urlparse(str(u)).netloc
                if d:
                    domains.add(d)
            except (ValueError, AttributeError) as e:
                logger.debug(f"Failed to parse URL node {u}: {e}")
        external_redirects = max(0, len(domains) - 1)

        # Max redirect depth (longest chain of URL nodes)
        max_redirect_depth = 0
        for u in url_nodes:
            try:
                paths = nx.single_source_shortest_path_length(graph, u)
                depth = max(paths.values()) if paths else 0
                max_redirect_depth = max(max_redirect_depth, depth)
            except (nx.NetworkXError, ValueError) as e:
                logger.debug(f"Redirect depth computation failed for {u}: {e}")

        form_to_node_ratio = num_forms / n if n > 0 else 0

        # Temporal features
        timestamps = []
        for _, data in graph.nodes(data=True):
            ts = data.get('timestamp', 0)
            if ts > 0:
                timestamps.append(ts)
        timestamps.sort()
        
        total_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        avg_time_between = total_time / (len(timestamps) - 1) if len(timestamps) > 1 else 0
        event_freq = len(timestamps) / total_time if total_time > 0 else 0

        requests_sensitive = has_password or has_email or credential_count > 0 or financial_count > 0

        return GraphFeatures(
            num_nodes=n, num_edges=m,
            avg_degree=round(avg_degree, 4), max_degree=max_deg,
            graph_density=round(density, 6), clustering_coefficient=round(clustering, 6),
            num_connected_components=n_components,
            avg_path_length=round(avg_path, 4), diameter=diameter,
            num_forms=num_forms, num_redirects=num_redirects,
            has_password_input=has_password, has_email_input=has_email,
            external_redirects=external_redirects,
            num_input_fields=num_input_fields, num_buttons=num_buttons,
            num_links=num_links, max_redirect_depth=max_redirect_depth,
            form_to_node_ratio=round(form_to_node_ratio, 6),
            total_interaction_time=round(total_time, 4),
            avg_time_between_events=round(avg_time_between, 4),
            event_frequency=round(event_freq, 4),
            betweenness_centrality_max=round(bc_max, 6),
            closeness_centrality_max=round(cc_max, 6),
            pagerank_max=round(pr_max, 6),
            in_degree_max=max(in_degrees) if in_degrees else 0,
            out_degree_max=max(out_degrees) if out_degrees else 0,
            requests_sensitive_data=requests_sensitive,
            credential_fields_count=credential_count,
            financial_fields_count=financial_count,
        )

    def detect_phishing_patterns(self, features: GraphFeatures) -> Dict:
        """Detect common phishing patterns from graph features."""
        patterns = {
            'credential_harvesting': False,
            'multiple_redirects': False,
            'suspicious_form_count': False,
            'financial_data_theft': False,
            'excessive_inputs': False,
            'low_complexity': False,
            'risk_score': 0.0,
            'risk_factors': [],
        }

        risk_score = 0.0
        factors = []

        if features.has_password_input and features.has_email_input:
            patterns['credential_harvesting'] = True
            risk_score += 0.25
            factors.append('Requests both email and password')

        if features.num_redirects > 2:
            patterns['multiple_redirects'] = True
            risk_score += 0.15
            factors.append(f'{features.num_redirects} redirects detected')

        if features.external_redirects > 1:
            risk_score += 0.15
            factors.append(f'{features.external_redirects} external domain redirects')

        if features.num_forms >= 2:
            patterns['suspicious_form_count'] = True
            risk_score += 0.1
            factors.append(f'{features.num_forms} form submissions')

        if features.financial_fields_count > 0:
            patterns['financial_data_theft'] = True
            risk_score += 0.2
            factors.append(f'{features.financial_fields_count} financial data fields')

        if features.credential_fields_count >= 3:
            patterns['excessive_inputs'] = True
            risk_score += 0.1
            factors.append(f'{features.credential_fields_count} credential fields')

        if features.clustering_coefficient < 0.1 and features.num_nodes > 3:
            patterns['low_complexity'] = True
            risk_score += 0.05
            factors.append('Low graph complexity')

        if features.form_to_node_ratio > 0.15:
            risk_score += 0.1
            factors.append(f'High form-to-node ratio: {features.form_to_node_ratio:.2f}')

        patterns['risk_score'] = min(round(risk_score, 4), 1.0)
        patterns['risk_factors'] = factors
        return patterns

    def visualize_graph(self, output_path: str = 'interaction_graph.png',
                        graph: Optional[nx.DiGraph] = None):
        """Visualize interaction graph."""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available for visualization")
            return

        if graph is None:
            graph = self.graph
        if graph is None:
            raise ValueError("No graph available")

        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)

        node_colors = []
        for _, data in graph.nodes(data=True):
            ntype = data.get('node_type', 'unknown')
            if ntype == 'url':
                node_colors.append('#3498db')
            elif ntype == 'submit':
                node_colors.append('#e74c3c')
            elif ntype in ('input', 'form_input'):
                node_colors.append('#f39c12')
            elif ntype == 'link_detected':
                node_colors.append('#2ecc71')
            else:
                node_colors.append('#95a5a6')

        nx.draw(graph, pos, node_color=node_colors, node_size=400,
                with_labels=False, arrows=True, edge_color='#bdc3c7',
                arrowsize=10, alpha=0.8, width=1.5)

        legend_elements = [
            Patch(facecolor='#3498db', label='URL/Page'),
            Patch(facecolor='#e74c3c', label='Form Submit'),
            Patch(facecolor='#f39c12', label='Input Field'),
            Patch(facecolor='#2ecc71', label='Link'),
            Patch(facecolor='#95a5a6', label='Other'),
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        plt.title('Website Interaction Graph', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graph saved to {output_path}")


def main():
    builder = InteractionGraphBuilder()
    trace_file = 'interaction_trace.json'
    try:
        graph = builder.build_graph_from_trace(trace_file)
        features = builder.extract_features()
        print(f"Nodes: {features.num_nodes}, Edges: {features.num_edges}")
        print(f"Forms: {features.num_forms}, Redirects: {features.num_redirects}")
        patterns = builder.detect_phishing_patterns(features)
        print(f"Risk Score: {patterns['risk_score']}")
        for f in patterns['risk_factors']:
            print(f"  - {f}")
    except FileNotFoundError:
        print(f"Trace file '{trace_file}' not found.")

if __name__ == "__main__":
    main()
