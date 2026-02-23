"""
PhishTrace Trace Quality Validator
====================================
Ensures crawled interaction traces are valid, usable data —
not error pages, parking pages, blank pages, or broken crawls.

This module filters out low-quality traces before they enter
the detection pipeline, improving both dataset integrity and
model performance.

Quality checks:
  1. HTTP status validation (reject 4xx/5xx error pages)
  2. Minimum interaction threshold (at least N events)
  3. Error page content detection (404, 500, "Access Denied", etc.)
  4. Parking/placeholder page detection ("buy this domain", etc.)
  5. CAPTCHA/bot-block detection (Cloudflare, hCaptcha, etc.)
  6. Redirect loop detection (infinite redirect chains)
  7. Empty DOM / minimal content detection
  8. Trace structural integrity (required fields present)
  9. Timeout / incomplete crawl detection
  10. Duplicate trace deduplication (same final_url + event fingerprint)
"""

import json
import re
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter

logger = logging.getLogger(__name__)


# ============================================================================
# Validation Result
# ============================================================================

@dataclass
class ValidationResult:
    """Result of trace quality validation."""
    trace_file: str
    is_valid: bool
    quality_score: float  # 0.0 (worst) to 1.0 (best)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def summary(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        parts = [f"[{status}] {self.trace_file} (quality={self.quality_score:.2f})"]
        for issue in self.issues:
            parts.append(f"  ERROR: {issue}")
        for warn in self.warnings:
            parts.append(f"  WARN: {warn}")
        return "\n".join(parts)


# ============================================================================
# Trace Validator
# ============================================================================

class TraceValidator:
    """
    Comprehensive trace quality validator.

    Applies a battery of checks to ensure interaction traces represent
    real, complete, usable website interactions — not error pages,
    parking pages, or broken crawls.
    """

    # --- Error page detection patterns ---
    ERROR_PAGE_PATTERNS = [
        # HTTP errors
        r'404\s*[-–—]\s*(?:page\s*)?not\s*found',
        r'403\s*[-–—]\s*forbidden',
        r'500\s*[-–—]\s*(?:internal\s*)?server\s*error',
        r'502\s*[-–—]\s*bad\s*gateway',
        r'503\s*[-–—]\s*service\s*unavailable',
        r'access\s*denied',
        r'this\s*page\s*(?:is\s*)?(?:not|no\s*longer)\s*(?:available|found)',
        r'page\s*(?:could\s*)?not\s*(?:be\s*)?found',
        r'the\s*(?:requested\s*)?(?:page|resource|url)\s*(?:was\s*)?not\s*found',
        r'(?:^|\b)error\s+(?:code\s+)?\d{3}\b',
        r'server\s*error',
        r'website\s*(?:is\s*)?(?:offline|down|unavailable)',
        r'this\s*site\s*can(?:\'t|\s*not)\s*be\s*reached',
        r'connection\s*(?:refused|timed?\s*out|reset)',
        r'dns\s*(?:resolution|lookup)\s*failed',
    ]

    # --- Parking / placeholder page patterns ---
    PARKING_PAGE_PATTERNS = [
        r'(?:buy|purchase)\s*this\s*domain',
        r'domain\s*(?:is\s*)?for\s*sale',
        r'this\s*domain\s*(?:has\s*)?(?:expired|is\s*available)',
        r'domain\s*parking',
        r'coming\s*soon',
        r'under\s*construction',
        r'parked\s*(?:free|domain)',
        r'(?:dan|sedo|afternic|godaddy)\s*(?:domain\s*)?(?:parking|marketplace)',
        r'this\s*webpage\s*is\s*parked',
        r'domain\s*(?:name\s*)?(?:registration|registrar)',
        r'hugedomains\.com',
        r'namecheap\s*(?:parking|redirect)',
    ]

    # --- CAPTCHA / bot-block detection patterns ---
    CAPTCHA_PATTERNS = [
        r'cloudflare.*(?:checking|verify|challenge)',
        r'attention\s*required.*cloudflare',
        r'please\s*(?:wait|verify|complete)',
        r'(?:h|re)captcha',
        r'bot\s*(?:detection|protection|verification)',
        r'ddos\s*protection',
        r'just\s*a\s*moment',
        r'verifying\s*(?:you\s*are|that\s*you)',
        r'ray\s*id',
        r'challenge-platform',
    ]

    # --- Blank / minimal content patterns ---
    BLANK_INDICATORS = [
        'about:blank',
        'data:text/html',
    ]

    def __init__(
        self,
        min_events: int = 1,
        min_network_requests: int = 1,
        max_redirect_chain: int = 10,
        min_quality_score: float = 0.3,
        error_status_codes: Set[int] = None,
    ):
        self.min_events = min_events
        self.min_network_requests = min_network_requests
        self.max_redirect_chain = max_redirect_chain
        self.min_quality_score = min_quality_score
        self.error_status_codes = error_status_codes or {400, 401, 403, 404, 405,
                                                          408, 410, 429, 500, 502,
                                                          503, 504}

        # Compile regex patterns
        self._error_re = [re.compile(p, re.IGNORECASE) for p in self.ERROR_PAGE_PATTERNS]
        self._parking_re = [re.compile(p, re.IGNORECASE) for p in self.PARKING_PAGE_PATTERNS]
        self._captcha_re = [re.compile(p, re.IGNORECASE) for p in self.CAPTCHA_PATTERNS]

    # -----------------------------------------------------------------------
    # Core validation
    # -----------------------------------------------------------------------

    def validate_trace(self, trace_data: Dict) -> ValidationResult:
        """
        Validate a single trace dict. Returns ValidationResult.

        Args:
            trace_data: Full trace JSON (top-level dict with 'trace', 'url', etc.)
        """
        file_name = trace_data.get('url', 'unknown')
        issues: List[str] = []
        warnings: List[str] = []
        quality = 1.0  # Start perfect, deduct for issues

        trace = trace_data.get('trace', {})
        events = trace.get('events', [])
        network = trace.get('network_requests', [])
        redirects = trace.get('redirects', [])
        page_title = trace.get('page_title', '') or ''
        console_logs = trace.get('console_logs', [])
        final_url = trace.get('final_url', '') or trace_data.get('final_url', '')

        # --- Check 1: Crawl success flag ---
        if not trace_data.get('success', True):
            issues.append("crawl_failed: trace marked as unsuccessful")
            quality -= 0.5

        # --- Check 2: Structural integrity ---
        if 'trace' not in trace_data:
            issues.append("missing_trace: no 'trace' field in JSON")
            return ValidationResult(file_name, False, 0.0, issues, warnings)

        if 'url' not in trace_data and 'url' not in trace:
            issues.append("missing_url: no URL in trace")
            quality -= 0.3

        # --- Check 3: HTTP status from network requests ---
        doc_statuses = []
        for req in network:
            if req.get('resource_type') == 'document':
                status = req.get('status', 200)
                if status and isinstance(status, int):
                    doc_statuses.append(status)

        if doc_statuses:
            primary_status = doc_statuses[0]
            if primary_status in self.error_status_codes:
                issues.append(f"http_error: primary document returned {primary_status}")
                quality -= 0.6
            elif primary_status >= 300 and primary_status < 400:
                # Redirects are fine
                pass
        else:
            warnings.append("no_doc_status: no document request status found")

        # --- Check 4: Minimum events ---
        if len(events) < self.min_events:
            if self.min_events <= 1:
                warnings.append(f"no_events: trace has {len(events)} events")
                quality -= 0.15
            else:
                issues.append(f"too_few_events: {len(events)} < {self.min_events}")
                quality -= 0.3

        # --- Check 5: Minimum network requests ---
        if len(network) < self.min_network_requests:
            warnings.append(f"few_network: only {len(network)} network requests")
            quality -= 0.1

        # --- Check 6: Error page content detection ---
        text_content = self._extract_text_content(trace_data)
        for regex in self._error_re:
            if regex.search(text_content):
                issues.append(f"error_page_content: matched '{regex.pattern}'")
                quality -= 0.4
                break  # One match is enough

        # --- Check 7: Parking page detection ---
        for regex in self._parking_re:
            if regex.search(text_content):
                issues.append(f"parking_page: matched '{regex.pattern}'")
                quality -= 0.5
                break

        # --- Check 8: CAPTCHA / bot-block detection ---
        # Check both page text and network URLs for CAPTCHA indicators
        captcha_text = text_content + ' ' + self._extract_network_urls(trace_data)
        for regex in self._captcha_re:
            if regex.search(captcha_text):
                warnings.append(f"captcha_detected: matched '{regex.pattern}'")
                quality -= 0.2
                break

        # --- Check 9: Blank / about:blank pages ---
        url = trace_data.get('url', '')
        if any(url.startswith(b) for b in self.BLANK_INDICATORS):
            issues.append("blank_page: URL is about:blank or data URI")
            quality -= 0.6

        if final_url and any(final_url.startswith(b) for b in self.BLANK_INDICATORS):
            issues.append("blank_final: final URL is about:blank")
            quality -= 0.5

        # --- Check 10: Redirect loop detection ---
        if len(redirects) > self.max_redirect_chain:
            issues.append(f"redirect_loop: {len(redirects)} redirects (max={self.max_redirect_chain})")
            quality -= 0.4

        # Check for circular redirects
        if len(redirects) > 2:
            redirect_set = set(redirects)
            if len(redirect_set) < len(redirects) * 0.5:
                issues.append("circular_redirects: repeated URLs in redirect chain")
                quality -= 0.4

        # --- Check 11: Crawl duration anomalies ---
        start = trace.get('start_time', 0)
        end = trace.get('end_time', 0)
        if start and end:
            duration = end - start
            if duration < 0.5:
                warnings.append(f"very_fast_crawl: {duration:.1f}s (possible instant error)")
                quality -= 0.1
            elif duration > 300:
                warnings.append(f"very_slow_crawl: {duration:.0f}s (possible timeout)")
                quality -= 0.1

        # --- Check 12: Empty page title for document pages with events ---
        if not page_title and len(events) > 0 and len(network) > 0:
            warnings.append("empty_title: page has no title")
            quality -= 0.05

        # --- Check 13: Console errors indicating broken pages ---
        severe_errors = [
            log for log in console_logs
            if isinstance(log, dict) and log.get('type') == 'error'
        ]
        if len(severe_errors) > 10:
            warnings.append(f"many_console_errors: {len(severe_errors)} errors")
            quality -= 0.1

        # Clamp quality score
        quality = max(0.0, min(1.0, quality))
        is_valid = quality >= self.min_quality_score and len(issues) == 0

        stats = {
            'num_events': len(events),
            'num_network_requests': len(network),
            'num_redirects': len(redirects),
            'doc_statuses': doc_statuses,
            'duration': (end - start) if start and end else None,
            'has_forms': trace.get('forms_submitted', 0) > 0,
            'quality_score': quality,
        }

        return ValidationResult(
            trace_file=file_name,
            is_valid=is_valid,
            quality_score=quality,
            issues=issues,
            warnings=warnings,
            stats=stats,
        )

    def _extract_text_content(self, trace_data: Dict) -> str:
        """
        Extract searchable text content from trace for error/parking detection.
        Only includes user-visible text (title, event text, element classes).
        Excludes network request URLs and console logs as they produce false
        positives (CDN error endpoints, JS debug messages).
        """
        parts = []
        trace = trace_data.get('trace', {})

        # Page title (strongest signal)
        title = trace.get('page_title', '') or ''
        parts.append(title)

        # Event texts (button labels, link text, etc.)
        for event in trace.get('events', []):
            text = event.get('element_text', '') or ''
            parts.append(text)
            cls = event.get('element_class', '') or ''
            parts.append(cls)

        return ' '.join(parts)

    def _extract_network_urls(self, trace_data: Dict) -> str:
        """Extract network request URLs separately for CAPTCHA detection only."""
        trace = trace_data.get('trace', {})
        urls = []
        for req in trace.get('network_requests', []):
            url = req.get('url', '') or ''
            urls.append(url)
        return ' '.join(urls)

    # -----------------------------------------------------------------------
    # Batch validation
    # -----------------------------------------------------------------------

    def validate_directory(
        self, traces_dir: Path, label: str = None
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Validate all trace files in a directory.

        Returns:
            (valid_traces, invalid_traces, summary_stats)
        """
        valid = []
        invalid = []
        all_issues = Counter()
        all_warnings = Counter()

        if label:
            dirs = [traces_dir / label]
        else:
            dirs = [traces_dir / 'phishing', traces_dir / 'benign']

        for d in dirs:
            if not d.exists():
                logger.warning(f"Directory not found: {d}")
                continue

            current_label = d.name
            for trace_file in sorted(d.glob('*.json')):
                try:
                    with open(trace_file, 'r', encoding='utf-8') as f:
                        trace_data = json.load(f)

                    result = self.validate_trace(trace_data)
                    result.trace_file = str(trace_file.name)
                    result.stats['label'] = current_label
                    result.stats['file_path'] = str(trace_file)

                    if result.is_valid:
                        valid.append({
                            'file': str(trace_file),
                            'quality': result.quality_score,
                            'label': current_label,
                            'warnings': result.warnings,
                        })
                    else:
                        invalid.append({
                            'file': str(trace_file),
                            'quality': result.quality_score,
                            'label': current_label,
                            'issues': result.issues,
                            'warnings': result.warnings,
                        })

                    for issue in result.issues:
                        all_issues[issue.split(':')[0]] += 1
                    for warn in result.warnings:
                        all_warnings[warn.split(':')[0]] += 1

                except json.JSONDecodeError as e:
                    invalid.append({
                        'file': str(trace_file),
                        'quality': 0.0,
                        'label': current_label,
                        'issues': [f"json_parse_error: {e}"],
                    })
                except Exception as e:
                    invalid.append({
                        'file': str(trace_file),
                        'quality': 0.0,
                        'label': current_label,
                        'issues': [f"validation_error: {e}"],
                    })

        summary = {
            'total': len(valid) + len(invalid),
            'valid': len(valid),
            'invalid': len(invalid),
            'valid_rate': len(valid) / max(1, len(valid) + len(invalid)),
            'avg_quality_valid': (
                sum(v['quality'] for v in valid) / max(1, len(valid))
            ),
            'issue_distribution': dict(all_issues.most_common()),
            'warning_distribution': dict(all_warnings.most_common()),
        }

        return valid, invalid, summary

    def validate_dataset(self, dataset_dir: Path) -> Dict:
        """
        Validate the entire dataset (both phishing and benign traces).
        Returns comprehensive validation report.
        """
        traces_dir = dataset_dir / 'traces'

        phishing_valid, phishing_invalid, phishing_summary = \
            self.validate_directory(traces_dir, 'phishing')
        benign_valid, benign_invalid, benign_summary = \
            self.validate_directory(traces_dir, 'benign')

        report = {
            'phishing': {
                'summary': phishing_summary,
                'invalid_traces': phishing_invalid[:50],  # Top 50
            },
            'benign': {
                'summary': benign_summary,
                'invalid_traces': benign_invalid[:50],
            },
            'overall': {
                'total': phishing_summary['total'] + benign_summary['total'],
                'valid': phishing_summary['valid'] + benign_summary['valid'],
                'invalid': phishing_summary['invalid'] + benign_summary['invalid'],
                'valid_rate': (
                    (phishing_summary['valid'] + benign_summary['valid']) /
                    max(1, phishing_summary['total'] + benign_summary['total'])
                ),
            }
        }

        return report

    # -----------------------------------------------------------------------
    # Trace fingerprinting for deduplication
    # -----------------------------------------------------------------------

    @staticmethod
    def trace_fingerprint(trace_data: Dict) -> str:
        """
        Generate a fingerprint for a trace to detect duplicates.
        Based on: final_url + sorted event types + network request domains.
        """
        trace = trace_data.get('trace', {})
        parts = []

        # Final URL
        final_url = trace.get('final_url', '') or trace_data.get('final_url', '')
        parts.append(final_url)

        # Event type sequence
        events = trace.get('events', [])
        event_types = sorted(e.get('event_type', '') for e in events)
        parts.append('|'.join(event_types))

        # Network request domains
        from urllib.parse import urlparse
        domains = set()
        for req in trace.get('network_requests', []):
            url = req.get('url', '')
            if url:
                try:
                    domains.add(urlparse(url).netloc)
                except Exception:
                    pass
        parts.append('|'.join(sorted(domains)))

        fingerprint = hashlib.md5('::'.join(parts).encode()).hexdigest()
        return fingerprint

    def deduplicate_traces(
        self, traces_dir: Path, label: str
    ) -> Tuple[List[str], List[str]]:
        """
        Find duplicate traces in a directory.

        Returns:
            (unique_files, duplicate_files)
        """
        seen_fps: Dict[str, str] = {}  # fingerprint -> first file
        unique = []
        duplicates = []

        trace_dir = traces_dir / label
        if not trace_dir.exists():
            return [], []

        for trace_file in sorted(trace_dir.glob('*.json')):
            try:
                with open(trace_file, 'r', encoding='utf-8') as f:
                    trace_data = json.load(f)

                fp = self.trace_fingerprint(trace_data)
                if fp in seen_fps:
                    duplicates.append(str(trace_file))
                    logger.debug(
                        f"Duplicate: {trace_file.name} == {seen_fps[fp]}"
                    )
                else:
                    seen_fps[fp] = trace_file.name
                    unique.append(str(trace_file))
            except Exception as e:
                logger.debug(f"Fingerprint error for {trace_file.name}: {e}")
                unique.append(str(trace_file))  # Keep on error

        return unique, duplicates


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Run trace validation on the dataset."""
    import argparse

    parser = argparse.ArgumentParser(description='PhishTrace Trace Validator')
    parser.add_argument('--dataset-dir', type=str,
                        default=str(Path(__file__).parent.parent.parent / 'dataset'),
                        help='Dataset directory path')
    parser.add_argument('--min-events', type=int, default=1,
                        help='Minimum number of events for a valid trace')
    parser.add_argument('--min-quality', type=float, default=0.3,
                        help='Minimum quality score for valid traces')
    parser.add_argument('--check-duplicates', action='store_true',
                        help='Also check for duplicate traces')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )

    dataset_dir = Path(args.dataset_dir)
    validator = TraceValidator(
        min_events=args.min_events,
        min_quality_score=args.min_quality,
    )

    print("=" * 70)
    print("PhishTrace Trace Quality Validator")
    print("=" * 70)
    print(f"Dataset: {dataset_dir}")
    print()

    report = validator.validate_dataset(dataset_dir)

    for label in ['phishing', 'benign']:
        s = report[label]['summary']
        print(f"\n--- {label.upper()} Traces ---")
        print(f"  Total: {s['total']}")
        print(f"  Valid: {s['valid']} ({s['valid_rate']*100:.1f}%)")
        print(f"  Invalid: {s['invalid']}")
        if s['valid'] > 0:
            print(f"  Avg quality (valid): {s['avg_quality_valid']:.3f}")

        if s['issue_distribution']:
            print(f"  Issue types:")
            for issue, count in s['issue_distribution'].items():
                print(f"    {issue}: {count}")
        if s['warning_distribution']:
            print(f"  Warning types:")
            for warn, count in s['warning_distribution'].items():
                print(f"    {warn}: {count}")

        # Show some invalid samples
        invalids = report[label]['invalid_traces'][:5]
        if invalids:
            print(f"  Sample invalid traces:")
            for inv in invalids:
                print(f"    {Path(inv['file']).name}: {inv['issues']}")

    overall = report['overall']
    print(f"\n--- OVERALL ---")
    print(f"  Total traces: {overall['total']}")
    print(f"  Valid: {overall['valid']} ({overall['valid_rate']*100:.1f}%)")
    print(f"  Invalid: {overall['invalid']}")

    # Deduplication
    if args.check_duplicates:
        traces_dir = dataset_dir / 'traces'
        for label in ['phishing', 'benign']:
            unique, dupes = validator.deduplicate_traces(traces_dir, label)
            print(f"\n  [{label}] Unique={len(unique)}, Duplicates={len(dupes)}")
            if dupes:
                for d in dupes[:5]:
                    print(f"    DUP: {Path(d).name}")

    # Save report
    report_path = dataset_dir / 'reports' / 'trace_validation_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        # Make serializable (limit invalid traces list for report)
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")

    return report


if __name__ == '__main__':
    main()
