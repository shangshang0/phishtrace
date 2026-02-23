"""
PhishTrace - Unit Tests
"""

import sys
import os
import json
import pytest
import numpy as np
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGraphBuilder:
    """Test the interaction graph builder."""

    def test_import(self):
        from src.analyzer.graph_builder import InteractionGraphBuilder
        builder = InteractionGraphBuilder()
        assert builder is not None

    def test_build_empty_graph(self):
        from src.analyzer.graph_builder import InteractionGraphBuilder
        builder = InteractionGraphBuilder()
        trace = {
            "url": "https://example.com",
            "events": [],
        }
        graph = builder.build_graph_from_dict(trace)
        assert graph is not None

    def test_build_graph_with_interactions(self):
        from src.analyzer.graph_builder import InteractionGraphBuilder
        builder = InteractionGraphBuilder()
        trace = {
            "url": "https://example.com",
            "events": [
                {"event_type": "navigation", "element_tag": "a", "element_id": "", "element_class": "",
                 "element_text": "", "element_xpath": "//a", "url_before": "https://example.com",
                 "url_after": "https://example.com", "timestamp": 0},
                {"event_type": "click", "element_tag": "button", "element_id": "login",
                 "element_class": "btn", "element_text": "Login", "element_xpath": "//button",
                 "url_before": "https://example.com", "url_after": "https://example.com",
                 "timestamp": 1.0},
                {"event_type": "input", "element_tag": "input", "element_id": "user",
                 "element_class": "", "element_text": "", "element_xpath": "//input",
                 "url_before": "https://example.com", "url_after": "https://example.com",
                 "timestamp": 2.0},
                {"event_type": "navigation", "element_tag": "", "element_id": "",
                 "element_class": "", "element_text": "", "element_xpath": "",
                 "url_before": "https://example.com", "url_after": "https://example.com/home",
                 "timestamp": 3.0},
            ],
        }
        graph = builder.build_graph_from_dict(trace)
        assert graph is not None
        assert len(graph.nodes) >= 2

    def test_extract_features(self):
        from src.analyzer.graph_builder import InteractionGraphBuilder
        builder = InteractionGraphBuilder()
        trace = {
            "url": "https://example.com",
            "events": [
                {"event_type": "navigation", "element_tag": "", "element_id": "",
                 "element_class": "", "element_text": "", "element_xpath": "",
                 "url_before": "https://example.com", "url_after": "https://example.com",
                 "timestamp": 0},
                {"event_type": "input", "element_tag": "input", "element_id": "password",
                 "element_class": "", "element_text": "", "element_xpath": "//input[@type='password']",
                 "url_before": "https://example.com", "url_after": "https://example.com",
                 "timestamp": 1.0},
            ],
        }
        graph = builder.build_graph_from_dict(trace)
        features = builder.extract_features(graph)
        assert features is not None
        assert hasattr(features, 'num_nodes')
        assert hasattr(features, 'num_edges')
        assert features.num_nodes >= 1


class TestBaselines:
    """Test baseline methods."""

    def test_import(self):
        from experiments.baselines.baseline_methods import run_all_baselines
        assert callable(run_all_baselines)

    def test_run_baselines(self):
        from experiments.baselines.baseline_methods import run_all_baselines
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        results = run_all_baselines(X, y)
        assert len(results) >= 5
        for name, metrics in results.items():
            assert 'accuracy' in metrics
            assert 'f1' in metrics
            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['f1'] <= 1

    def test_individual_baselines(self):
        from experiments.baselines.baseline_methods import (
            PhishpediaBaseline, CrawlPhishBaseline, TesseractBaseline,
            DosAndDontsBaseline, DeltaPhishBaseline
        )
        rng = np.random.RandomState(42)
        X = rng.randn(80, 5)
        y = (X[:, 0] > 0).astype(int)

        for cls in [PhishpediaBaseline, CrawlPhishBaseline, TesseractBaseline,
                    DosAndDontsBaseline, DeltaPhishBaseline]:
            baseline = cls()
            metrics = baseline.evaluate(X, y, n_folds=3)
            assert metrics['accuracy'] > 0


class TestDetector:
    """Test the phishing detector."""

    def test_import(self):
        from src.detector.phish_detector import PhishTraceDetector
        detector = PhishTraceDetector()
        assert detector is not None


class TestCrawlers:
    """Test crawler modules (import only - actual crawling requires network)."""

    def test_phishing_crawler_import(self):
        from src.crawler.phishing_crawler import PhishingCrawler
        assert PhishingCrawler is not None


class TestUtils:
    """Test utility functions."""

    def test_helpers_import(self):
        from src.utils.helpers import load_json, save_json, ensure_dir
        assert callable(load_json)
        assert callable(save_json)
        assert callable(ensure_dir)

    def test_ensure_dir(self, tmp_path):
        from src.utils.helpers import ensure_dir
        test_dir = str(tmp_path / "test_dir" / "nested")
        ensure_dir(test_dir)
        assert os.path.isdir(test_dir)

    def test_save_load_json(self, tmp_path):
        from src.utils.helpers import save_json, load_json
        data = {"key": "value", "num": 42, "nested": {"a": [1, 2, 3]}}
        path = str(tmp_path / "test.json")
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data


class TestTraceValidator:
    """Test the trace quality validator."""

    def test_import(self):
        from src.validator.trace_validator import TraceValidator
        v = TraceValidator()
        assert v is not None

    def test_valid_trace(self):
        from src.validator.trace_validator import TraceValidator
        v = TraceValidator()
        trace = {
            "url": "https://example.com",
            "success": True,
            "trace": {
                "url": "https://example.com",
                "start_time": 0,
                "end_time": 5.0,
                "events": [
                    {"event_type": "click", "element_text": "Login", "element_class": "btn"},
                ],
                "network_requests": [
                    {"url": "https://example.com", "resource_type": "document", "status": 200},
                ],
                "redirects": ["https://example.com"],
                "final_url": "https://example.com",
                "page_title": "Example",
                "console_logs": [],
            },
        }
        result = v.validate_trace(trace)
        assert result.is_valid
        assert result.quality_score > 0.5

    def test_error_page_detection(self):
        from src.validator.trace_validator import TraceValidator
        v = TraceValidator()
        trace = {
            "url": "https://bad.com",
            "success": True,
            "trace": {
                "url": "https://bad.com",
                "start_time": 0,
                "end_time": 2.0,
                "events": [{"event_type": "link", "element_text": "404 - Page Not Found", "element_class": ""}],
                "network_requests": [
                    {"url": "https://bad.com", "resource_type": "document", "status": 404},
                ],
                "redirects": [],
                "final_url": "https://bad.com",
                "page_title": "Not Found",
                "console_logs": [],
            },
        }
        result = v.validate_trace(trace)
        assert not result.is_valid
        assert any("http_error" in i or "error_page_content" in i for i in result.issues)

    def test_blank_page_detection(self):
        from src.validator.trace_validator import TraceValidator
        v = TraceValidator()
        trace = {
            "url": "https://phish.test",
            "success": True,
            "trace": {
                "url": "https://phish.test",
                "start_time": 0,
                "end_time": 1.0,
                "events": [],
                "network_requests": [],
                "redirects": [],
                "final_url": "about:blank",
                "page_title": "",
                "console_logs": [],
            },
        }
        result = v.validate_trace(trace)
        assert not result.is_valid
        assert any("blank" in i for i in result.issues)

    def test_fingerprinting(self):
        from src.validator.trace_validator import TraceValidator
        v = TraceValidator()
        trace1 = {
            "url": "https://a.com",
            "trace": {
                "final_url": "https://a.com",
                "events": [{"event_type": "click"}],
                "network_requests": [{"url": "https://a.com"}],
            },
        }
        trace2 = {
            "url": "https://a.com",
            "trace": {
                "final_url": "https://a.com",
                "events": [{"event_type": "click"}],
                "network_requests": [{"url": "https://a.com"}],
            },
        }
        fp1 = v.trace_fingerprint(trace1)
        fp2 = v.trace_fingerprint(trace2)
        assert fp1 == fp2  # Same content = same fingerprint


class TestWildDomainDiscovery:
    """Test the wild domain discovery scanner."""

    def test_discovery_instantiation(self):
        from src.scanner.wild_scanner import WildDomainDiscovery
        scanner = WildDomainDiscovery(use_proxy=False)
        assert scanner._session is not None
        assert hasattr(scanner, 'keyword_filter')

    def test_discovered_domain_dataclass(self):
        from src.scanner.wild_scanner import DiscoveredDomain
        d = DiscoveredDomain(
            domain="paypal-login-verify.xyz",
            url="https://paypal-login-verify.xyz",
            source="ct_logs",
            discovered_at="2026-01-01T00:00:00",
        )
        assert d.domain == "paypal-login-verify.xyz"
        assert d.source == "ct_logs"
        assert d.as_dict()["domain"] == "paypal-login-verify.xyz"

    def test_suspicious_keyword_filter(self):
        from src.scanner.wild_scanner import _looks_suspicious
        assert _looks_suspicious("paypal-login-verify.xyz")
        assert _looks_suspicious("secure-banking.com")
        assert not _looks_suspicious("example.com")

    def test_discover_all_method_exists(self):
        from src.scanner.wild_scanner import WildDomainDiscovery
        scanner = WildDomainDiscovery(use_proxy=False)
        assert callable(scanner.discover_all)
        assert callable(scanner.discover_ct_logs)
        assert callable(scanner.discover_urlscan)
        assert callable(scanner.discover_nod_whoisds)
        assert callable(scanner.discover_urlhaus)
