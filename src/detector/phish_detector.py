"""
PhishTrace Detector - Main detection module
Uses interaction trace graph analysis for phishing website detection.
"""
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from dataclasses import asdict


class PhishTraceDetector:
    """
    Core phishing detection engine.
    Combines Playwright-based deep crawling with graph-based analysis.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load pre-trained model if available."""
        if self.model_path and Path(self.model_path).exists():
            import joblib
            self.model = joblib.load(self.model_path)
        else:
            # Use heuristic-based detection as fallback
            self.model = None

    def detect_url_sync(self, url: str) -> Dict:
        """Synchronous wrapper for URL detection."""
        try:
            # Use asyncio.run() for clean event loop management
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Already in async context - use thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(asyncio.run, self.detect_url(url)).result(timeout=60)
            else:
                result = asyncio.run(self.detect_url(url))
            return result
        except Exception as e:
            return {
                "url": url,
                "is_phishing": False,
                "confidence": 0.0,
                "method": "error_fallback",
                "error": str(e)
            }

    async def detect_url(self, url: str, headless: bool = True) -> Dict:
        """
        Full detection pipeline for a URL:
        1. Crawl with Playwright
        2. Build interaction graph
        3. Extract features
        4. Classify
        """
        from src.crawler.phishing_crawler import PhishingCrawler
        from src.analyzer.graph_builder import InteractionGraphBuilder

        # Step 1: Crawl
        crawler = PhishingCrawler(headless=headless, timeout=15000)
        try:
            trace = await crawler.crawl(url, max_depth=2)
        except Exception as e:
            return {
                "url": url,
                "is_phishing": False,
                "confidence": 0.0,
                "method": "crawl_failed",
                "error": str(e)
            }

        # Step 2: Build graph from trace
        builder = InteractionGraphBuilder()
        trace_dict = asdict(trace)
        graph = builder.build_graph_from_dict(trace_dict)

        # Step 3: Extract features
        features = builder.extract_features(graph)

        # Step 4: Classify
        if self.model is not None:
            feature_vec = self._features_to_vector(features)
            prediction = self.model.predict([feature_vec])[0]
            proba = self.model.predict_proba([feature_vec])[0]
            is_phishing = bool(prediction == 1)
            confidence = float(proba[1]) if len(proba) > 1 else float(prediction)
            method = "ml_model"
        else:
            # Heuristic fallback
            patterns = builder.detect_phishing_patterns(features)
            is_phishing = patterns['risk_score'] > 0.5
            confidence = patterns['risk_score']
            method = "heuristic"

        return {
            "url": url,
            "is_phishing": is_phishing,
            "confidence": round(confidence, 4),
            "method": method,
            "features": {
                "num_nodes": features.num_nodes,
                "num_edges": features.num_edges,
                "num_forms": features.num_forms,
                "num_redirects": features.num_redirects,
                "has_password_input": features.has_password_input,
                "has_email_input": features.has_email_input,
                "graph_density": round(features.graph_density, 4),
                "clustering_coefficient": round(features.clustering_coefficient, 4),
            },
            "trace_events": len(trace.events),
            "final_url": trace.final_url,
        }

    def _features_to_vector(self, features) -> np.ndarray:
        """Convert GraphFeatures dataclass to numpy vector (all 30 features)."""
        return np.array([
            features.num_nodes,
            features.num_edges,
            features.avg_degree,
            features.max_degree,
            features.graph_density,
            features.clustering_coefficient,
            features.num_connected_components,
            features.avg_path_length,
            features.diameter,
            features.num_forms,
            features.num_redirects,
            1 if features.has_password_input else 0,
            1 if features.has_email_input else 0,
            features.external_redirects,
            features.num_input_fields,
            features.num_buttons,
            features.num_links,
            features.max_redirect_depth,
            features.form_to_node_ratio,
            features.total_interaction_time,
            features.avg_time_between_events,
            features.event_frequency,
            features.betweenness_centrality_max,
            features.closeness_centrality_max,
            features.pagerank_max,
            features.in_degree_max,
            features.out_degree_max,
            1 if features.requests_sensitive_data else 0,
            features.credential_fields_count,
            features.financial_fields_count,
        ])
