"""
PhishTrace Deep Interaction Crawler
Uses Playwright to deeply crawl websites, interact with all elements,
automatically fill and submit forms, and capture comprehensive interaction traces.
"""

import asyncio
import json
import time
import hashlib
import re
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field, asdict
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)

try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    # Define stubs so type hints don't cause NameError at class definition time
    Page = None
    Browser = None
    BrowserContext = None
    logger.warning("Playwright not installed. Crawler will use fallback mode.")


@dataclass
class InteractionEvent:
    """Represents a single interaction event during crawling."""
    timestamp: float
    event_type: str
    element_tag: str
    element_id: str
    element_class: str
    element_text: str
    element_xpath: str
    url_before: str
    url_after: str
    form_data: Optional[Dict] = None
    screenshot_path: Optional[str] = None
    http_status: Optional[int] = None
    response_headers: Optional[Dict] = None
    dom_changes: Optional[List[str]] = None
    js_errors: Optional[List[str]] = None


@dataclass
class NetworkRequest:
    """Captured network request during crawling."""
    url: str
    method: str
    resource_type: str
    status: Optional[int] = None
    content_type: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class InteractionTrace:
    """Complete interaction trace for a website."""
    url: str
    start_time: float
    end_time: float
    events: List[InteractionEvent] = field(default_factory=list)
    final_url: str = ""
    redirects: List[str] = field(default_factory=list)
    forms_submitted: int = 0
    elements_interacted: int = 0
    network_requests: List[NetworkRequest] = field(default_factory=list)
    console_logs: List[str] = field(default_factory=list)
    page_title: str = ""
    dom_snapshot: Optional[str] = None
    cookies: List[Dict] = field(default_factory=list)
    local_storage: Optional[Dict] = None
    dual_submissions_detected: int = 0


class PhishingCrawler:
    """
    Advanced phishing website crawler with deep interaction capabilities.
    
    Features:
    - Automatic discovery of all interactive elements
    - Form filling with realistic test data  
    - Redirect chain tracking
    - Network request capture
    - Console log capture
    - Cookie and local storage extraction
    """

    # Pool of test personas — each crawl randomly picks one.  On
    # dual-submission retries a DIFFERENT persona is used so we can
    # detect "accepts any credentials" (strong phishing signal).
    _PERSONA_POOL = [
        {
            'email': 'john.doe.test2024@gmail.com',
            'password': 'S3cur3P@ssw0rd!2024',
            'text': 'John Doe',   'tel': '+1-555-0142',
            'username': 'johndoe2024',
            'first_name': 'John',  'last_name': 'Doe',
            'address': '123 Main St, Suite 100',
            'city': 'San Francisco', 'state': 'CA', 'zip': '94105',
            'country': 'US',
            'ssn': '123-45-6789', 'cc_number': '4111111111111111',
            'cc_exp': '12/28',    'cc_cvv': '123',
            'dob': '1990-01-15',  'company': 'Acme Corp',
            'pin': '7832',        'otp': '482917',
        },
        {
            'email': 'sarah.miller88@outlook.com',
            'password': 'Tr0ub4dor&3_2025',
            'text': 'Sarah Miller', 'tel': '+1-555-0397',
            'username': 'smiller88',
            'first_name': 'Sarah', 'last_name': 'Miller',
            'address': '4520 Elm Avenue, Apt 7B',
            'city': 'Portland', 'state': 'OR', 'zip': '97201',
            'country': 'US',
            'ssn': '987-65-4321', 'cc_number': '5500000000000004',
            'cc_exp': '03/27',    'cc_cvv': '456',
            'dob': '1988-07-22',  'company': 'TechStart Inc',
            'pin': '2491',        'otp': '713052',
        },
        {
            'email': 'mark.chen.dev@yahoo.com',
            'password': 'C0rr3ct-H0rse!99',
            'text': 'Mark Chen',   'tel': '+1-555-0618',
            'username': 'markc_dev',
            'first_name': 'Mark',  'last_name': 'Chen',
            'address': '891 Oak Boulevard',
            'city': 'Austin', 'state': 'TX', 'zip': '73301',
            'country': 'US',
            'ssn': '456-78-9012', 'cc_number': '378282246310005',
            'cc_exp': '09/26',    'cc_cvv': '7890',
            'dob': '1995-03-10',  'company': 'DataWave LLC',
            'pin': '5063',        'otp': '928374',
        },
        {
            'email': 'lisa.wang.2025@proton.me',
            'password': 'Wh1skey-Tang0-F0x!',
            'text': 'Lisa Wang',   'tel': '+1-555-0854',
            'username': 'lwang2025',
            'first_name': 'Lisa',  'last_name': 'Wang',
            'address': '2200 Pine Street, Unit 14',
            'city': 'Seattle', 'state': 'WA', 'zip': '98101',
            'country': 'US',
            'ssn': '234-56-7890', 'cc_number': '6011111111111117',
            'cc_exp': '06/29',    'cc_cvv': '321',
            'dob': '1992-11-05',  'company': 'CloudNine Software',
            'pin': '8174',        'otp': '504861',
        },
    ]

    # Common defaults shared across all personas
    _COMMON_DATA = {
        'number': '12345',
        'url': 'https://example.com',
        'search': 'test search query',
    }

    def __init__(self, headless: bool = True, timeout: int = 30000,
                 capture_screenshots: bool = False, screenshot_dir: str = "screenshots",
                 proxy: str = None):
        self.headless = headless
        self.timeout = timeout
        self.capture_screenshots = capture_screenshots
        self.screenshot_dir = screenshot_dir
        self.proxy = proxy
        self._visited_urls: Set[str] = set()
        self._screenshot_counter: int = 0
        # Pick a random primary persona for this crawler instance;
        # _alt_persona is used during dual-submission retries to test
        # whether the site accepts ANY credentials.
        self._persona_idx: int = random.randrange(len(self._PERSONA_POOL))
        self._alt_persona_idx: int = (self._persona_idx + 1) % len(self._PERSONA_POOL)

    @property
    def TEST_DATA(self) -> Dict[str, str]:
        """Return the currently active test persona merged with common data."""
        d = dict(self._COMMON_DATA)
        d.update(self._PERSONA_POOL[self._persona_idx])
        return d

    def _rotate_persona(self):
        """Switch to the alternate persona (used during retry probes)."""
        self._persona_idx, self._alt_persona_idx = (
            self._alt_persona_idx, self._persona_idx)

    async def _take_screenshot(self, page: Page, step_label: str,
                               trace_id: str = '') -> Optional[str]:
        """Capture a screenshot at the current interaction step.

        Saves to ``self.screenshot_dir / step_NNN_<label>.png``.
        Returns the file path on success, *None* otherwise.
        """
        if not self.capture_screenshots:
            return None
        try:
            self._screenshot_counter += 1
            out_dir = Path(self.screenshot_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            # Sanitise the label so it is filename-safe
            safe_label = re.sub(r'[^\w\-]', '_', step_label)[:40]
            filename = f"step_{self._screenshot_counter:03d}_{safe_label}.png"
            filepath = out_dir / filename
            await page.screenshot(path=str(filepath), full_page=False,
                                  timeout=8000)
            logger.debug(f"Screenshot saved: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.debug(f"Screenshot failed ({step_label}): {e}")
            return None

    async def crawl(self, url: str, max_depth: int = 3) -> InteractionTrace:
        """Crawl a website with deep interaction."""
        self._visited_urls = set()  # Reset state for each crawl
        self._screenshot_counter = 0  # Reset screenshot counter
        # Re-randomise persona for this crawl session
        self._persona_idx = random.randrange(len(self._PERSONA_POOL))
        self._alt_persona_idx = (self._persona_idx + 1) % len(self._PERSONA_POOL)
        # Compute a trace ID for organizing screenshots
        trace_id = hashlib.md5(url.encode()).hexdigest()[:12]
        if not HAS_PLAYWRIGHT:
            return self._fallback_crawl(url)

        async with async_playwright() as p:
            launch_args = {
                'headless': self.headless,
                'args': ['--no-sandbox', '--disable-setuid-sandbox',
                         '--disable-dev-shm-usage', '--disable-web-security'],
            }
            if self.proxy:
                launch_args['proxy'] = {'server': self.proxy}
            browser = await p.chromium.launch(**launch_args)
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                ignore_https_errors=True,
                java_script_enabled=True,
            )

            page = await context.new_page()
            start_time = time.time()
            events: List[InteractionEvent] = []
            redirects: List[str] = []
            network_reqs: List[NetworkRequest] = []
            console_logs: List[str] = []

            page.on("request", lambda req: network_reqs.append(NetworkRequest(
                url=req.url, method=req.method, resource_type=req.resource_type,
                timestamp=time.time()
            )))

            def _on_response(resp):
                for nr in network_reqs:
                    if nr.url == resp.url and nr.status is None:
                        nr.status = resp.status
                        nr.content_type = resp.headers.get('content-type', '')
                        break
            page.on("response", _on_response)
            page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))

            try:
                # Use domcontentloaded instead of networkidle: large benign
                # sites (ads, analytics, WebSockets) never reach networkidle
                # within 30 s.  DOM-ready is sufficient — we only need forms,
                # buttons and links to be present.  Network requests are still
                # captured by the event listeners above.
                response = await page.goto(url, timeout=self.timeout, wait_until='domcontentloaded')
                # Short stabilisation wait for JS-rendered content
                await page.wait_for_timeout(2000)
                redirects.append(page.url)
                self._visited_urls.add(page.url)
                page_title = await page.title()

                # Capture initial homepage screenshot
                init_ss = await self._take_screenshot(page, 'homepage', trace_id)
                if init_ss:
                    events.append(InteractionEvent(
                        timestamp=time.time(), event_type='screenshot',
                        element_tag='page', element_id='homepage',
                        element_class='', element_text='',
                        element_xpath='//html',
                        url_before=url, url_after=page.url,
                        screenshot_path=init_ss,
                    ))

                events.extend(await self._interact_with_page(page, depth=0, max_depth=max_depth, trace_id=trace_id))
                cookies = await context.cookies()
                try:
                    local_storage = await page.evaluate("""() => {
                        let items = {};
                        for (let i = 0; i < localStorage.length; i++) {
                            let key = localStorage.key(i);
                            items[key] = localStorage.getItem(key);
                        }
                        return items;
                    }""")
                except Exception:
                    local_storage = {}
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
                page_title = ""
                cookies = []
                local_storage = {}

            end_time = time.time()
            trace = InteractionTrace(
                url=url, start_time=start_time, end_time=end_time,
                events=events, final_url=page.url, redirects=redirects,
                forms_submitted=sum(1 for e in events if 'submit' in e.event_type),
                elements_interacted=len(events),
                network_requests=network_reqs, console_logs=console_logs,
                page_title=page_title,
                cookies=[{"name": c["name"], "domain": c["domain"], "path": c["path"]} for c in cookies],
                local_storage=local_storage,
                dual_submissions_detected=sum(1 for e in events if e.event_type == 'dual_submit_resubmit'),
            )
            await browser.close()
            return trace

    # ------------------------------------------------------------------ #
    #  Patterns that indicate login / credential entry (case-insensitive)
    # ------------------------------------------------------------------ #
    _LOGIN_LINK_RE = re.compile(
        r'log\s*in|sign\s*in|sign\s*up|register|my\s*account|connexion|'
        r'anmelden|iniciar\s*ses|entrar|accedi|inloggen|войти|ログイン|登录',
        re.IGNORECASE,
    )

    async def _interact_with_page(self, page: Page, depth: int, max_depth: int,
                                   trace_id: str = '') -> List[InteractionEvent]:
        """
        Deep interaction loop.

        Strategy:
        1. Detect & fill all visible input fields on the current page.
        2. Find and submit every <form>.
        3. Click standalone buttons that are NOT inside a <form>.
        4. Follow "login / sign-in" links to new pages.
        5. Recurse on new pages up to *max_depth*.
        """
        events = []
        if depth >= max_depth:
            return events

        # ── Phase 1: fill visible inputs (outside of forms — forms
        #    have their own fill-and-submit path) ──
        standalone_inputs = await self._get_standalone_inputs(page)
        for inp_info in standalone_inputs:
            try:
                ev = await self._fill_input(page, inp_info['element'],
                                            inp_info['tag'], inp_info['type'],
                                            inp_info['name'], inp_info['id'])
                if ev:
                    events.append(ev)
                    await asyncio.sleep(0.15)
            except Exception:
                pass

        # ── Phase 2: fill & submit every <form> ──
        forms = await page.query_selector_all('form')
        for i, form in enumerate(forms[:5]):
            try:
                form_events = await self._fill_and_submit_form(
                    page, form, i, trace_id=trace_id)
                events.extend(form_events)
            except Exception as e:
                logger.debug(f"Error with form {i}: {e}")

        # ── Phase 3: click standalone (non-form) buttons ──
        url_before_buttons = page.url
        for sel in ['button[type="submit"]', 'button:not([type])',
                     'input[type="submit"]', 'input[type="button"]',
                     '[role="button"]', '.btn-primary', '.login-btn',
                     '.submit-btn', '.btn-login', '.btn-submit']:
            try:
                btns = await page.query_selector_all(sel)
                for btn in btns[:3]:
                    if not await btn.is_visible():
                        continue
                    # Skip if the button is inside a <form> (already handled)
                    inside_form = await btn.evaluate(
                        'el => !!el.closest("form")')
                    if inside_form:
                        continue
                    btn_text = ''
                    try:
                        btn_text = (await btn.text_content() or '')[:100]
                    except Exception:
                        pass
                    try:
                        await btn.click(timeout=3000)
                        await page.wait_for_timeout(2000)
                    except Exception:
                        continue

                    ss_path = await self._take_screenshot(
                        page, f'after_btn_{sel[:12].replace("[","").replace("]","")}',
                        trace_id)
                    events.append(InteractionEvent(
                        timestamp=time.time(), event_type='button_click',
                        element_tag='button', element_id='',
                        element_class=sel, element_text=btn_text,
                        element_xpath=f'//{sel}',
                        url_before=url_before_buttons, url_after=page.url,
                        screenshot_path=ss_path,
                    ))
                    # If the click navigated to a new page, recurse
                    if page.url != url_before_buttons:
                        self._visited_urls.add(page.url)
                        events.extend(await self._interact_with_page(
                            page, depth + 1, max_depth, trace_id))
                        return events  # already recursed
                    break  # one click per selector family is enough
            except Exception:
                pass

        # ── Phase 4: follow "Login / Sign-in" links ──
        try:
            links = await page.query_selector_all('a[href]')
            for link in links[:30]:
                try:
                    if not await link.is_visible():
                        continue
                    href = await link.get_attribute('href') or ''
                    link_text = ''
                    try:
                        link_text = (await link.text_content() or '')[:100]
                    except Exception:
                        pass
                    combined = (link_text + ' ' + href).strip()
                    if not self._LOGIN_LINK_RE.search(combined):
                        continue
                    # Looks like a login link — follow it
                    url_before_link = page.url
                    full_href = urljoin(page.url, href)
                    if full_href in self._visited_urls:
                        continue
                    self._visited_urls.add(full_href)
                    try:
                        await link.click(timeout=5000)
                        await page.wait_for_timeout(2000)
                    except Exception:
                        try:
                            await page.goto(full_href, timeout=self.timeout,
                                            wait_until='domcontentloaded')
                            await page.wait_for_timeout(1000)
                        except Exception:
                            continue

                    ss_path = await self._take_screenshot(
                        page, f'after_login_link', trace_id)
                    events.append(InteractionEvent(
                        timestamp=time.time(), event_type='login_link_follow',
                        element_tag='a', element_id='',
                        element_class='', element_text=link_text[:100],
                        element_xpath='//a',
                        url_before=url_before_link, url_after=page.url,
                        form_data={'href': href},
                        screenshot_path=ss_path,
                    ))
                    # Recurse on the new page
                    events.extend(await self._interact_with_page(
                        page, depth + 1, max_depth, trace_id))
                    return events  # followed a link, done at this depth
                except Exception:
                    pass
        except Exception:
            pass

        # ── Phase 5: JS-based login forms (no <form> tag) ──
        # Look for password fields outside <form> — common in SPAs
        try:
            pwd_fields = await page.query_selector_all(
                'input[type="password"]:not(form input)')
            if pwd_fields:
                for pf in pwd_fields[:2]:
                    if not await pf.is_visible():
                        continue
                    # Fill all visible inputs near the password field
                    # (typically email + password in a container div)
                    container = await pf.evaluate_handle(
                        'el => el.closest("div, section, main, [class*=login], '
                        '[class*=form], [class*=auth]") || el.parentElement')
                    if container:
                        inputs = await container.query_selector_all('input')
                        form_data = {}
                        for inp in inputs:
                            try:
                                itype = (await inp.get_attribute('type') or 'text').lower()
                                iname = await inp.get_attribute('name') or ''
                                iid = await inp.get_attribute('id') or ''
                                if itype in ('text', 'email', 'password', 'tel'):
                                    val = self._get_test_value(itype, iname, iid)
                                    await inp.fill(val)
                                    form_data[iname or iid or itype] = val
                            except Exception:
                                pass
                        # Try to submit by pressing Enter on the password field
                        url_before_js = page.url
                        try:
                            await pf.press('Enter')
                            await page.wait_for_timeout(3000)
                        except Exception:
                            pass
                        ss_path = await self._take_screenshot(
                            page, 'after_js_login_submit', trace_id)
                        events.append(InteractionEvent(
                            timestamp=time.time(), event_type='js_form_submit',
                            element_tag='input', element_id='password',
                            element_class='', element_text='',
                            element_xpath='//input[@type="password"]',
                            url_before=url_before_js, url_after=page.url,
                            form_data=form_data,
                            screenshot_path=ss_path,
                        ))
                        # Check for dual-submission
                        dual_events = await self._check_dual_submission(
                            page, form_data, url_before_js, trace_id=trace_id)
                        events.extend(dual_events)
                        break  # handled JS login
        except Exception as e:
            logger.debug(f"JS login detection error: {e}")

        # ── Depth transition (URL changed during interaction) ──
        current_url = page.url
        if current_url not in self._visited_urls and depth + 1 < max_depth:
            self._visited_urls.add(current_url)
            depth_ss = await self._take_screenshot(
                page, f'depth{depth+1}_page', trace_id)
            if depth_ss:
                events.append(InteractionEvent(
                    timestamp=time.time(), event_type='page_transition',
                    element_tag='page', element_id=f'depth_{depth+1}',
                    element_class='', element_text='',
                    element_xpath='//html',
                    url_before=current_url, url_after=page.url,
                    screenshot_path=depth_ss,
                ))
            events.extend(await self._interact_with_page(
                page, depth + 1, max_depth, trace_id=trace_id))

        return events

    async def _get_standalone_inputs(self, page: Page) -> List[Dict]:
        """Return visible input/textarea/select elements NOT inside a <form>."""
        result = []
        for sel in ['input[type="text"]', 'input[type="email"]',
                     'input[type="password"]', 'input[type="tel"]',
                     'input[type="number"]', 'textarea', 'select',
                     'input[type="checkbox"]', 'input[type="radio"]']:
            try:
                elements = await page.query_selector_all(sel)
                for el in elements[:5]:
                    try:
                        if not await el.is_visible():
                            continue
                        inside_form = await el.evaluate('e => !!e.closest("form")')
                        if inside_form:
                            continue
                        tag = await el.evaluate('e => e.tagName.toLowerCase()')
                        result.append({
                            'element': el,
                            'tag': tag,
                            'type': (await el.get_attribute('type') or 'text').lower(),
                            'name': await el.get_attribute('name') or '',
                            'id': await el.get_attribute('id') or '',
                        })
                    except Exception:
                        pass
            except Exception:
                pass
        return result

    async def _fill_input(self, page: Page, element, tag: str, itype: str,
                          name: str, el_id: str) -> Optional[InteractionEvent]:
        """Fill a single input element and return an event."""
        url_before = page.url
        try:
            if itype in ('text', 'email', 'password', 'tel', 'number', 'url', 'search'):
                val = self._get_test_value(itype, name, el_id)
                await element.fill(val)
                return InteractionEvent(
                    timestamp=time.time(), event_type='input',
                    element_tag=tag, element_id=el_id,
                    element_class=await element.get_attribute('class') or '',
                    element_text='', element_xpath=f'//{tag}',
                    url_before=url_before, url_after=page.url,
                    form_data={name or el_id or 'field': val},
                )
            elif itype in ('checkbox', 'radio'):
                try:
                    await element.check()
                except Exception:
                    await element.click()
                return InteractionEvent(
                    timestamp=time.time(), event_type='check',
                    element_tag=tag, element_id=el_id,
                    element_class='', element_text='',
                    element_xpath=f'//{tag}',
                    url_before=url_before, url_after=page.url,
                )
            elif tag == 'textarea':
                await element.fill('Test message for form evaluation.')
                return InteractionEvent(
                    timestamp=time.time(), event_type='input',
                    element_tag='textarea', element_id=el_id,
                    element_class='', element_text='',
                    element_xpath='//textarea',
                    url_before=url_before, url_after=page.url,
                    form_data={name or 'textarea': 'test'},
                )
            elif tag == 'select':
                opts = await element.query_selector_all('option')
                if len(opts) > 1:
                    await opts[1].click()
                return InteractionEvent(
                    timestamp=time.time(), event_type='select',
                    element_tag='select', element_id=el_id,
                    element_class='', element_text='',
                    element_xpath='//select',
                    url_before=url_before, url_after=page.url,
                )
        except Exception as e:
            logger.debug(f"_fill_input error: {e}")
        return None

    async def _fill_and_submit_form(self, page: Page, form, form_idx: int,
                                     trace_id: str = '') -> List[InteractionEvent]:
        events = []
        url_before = page.url
        form_data = {}

        try:
            inputs = await form.query_selector_all('input, textarea, select')
            for inp in inputs:
                try:
                    tag = await inp.evaluate('el => el.tagName.toLowerCase()')
                    itype = (await inp.get_attribute('type') or 'text').lower()
                    iname = await inp.get_attribute('name') or ''
                    iid = await inp.get_attribute('id') or ''

                    if itype in ('text', 'email', 'password', 'tel', 'number', 'url', 'search'):
                        val = self._get_test_value(itype, iname, iid)
                        await inp.fill(val)
                        form_data[iname or iid or f'field_{form_idx}'] = val
                        events.append(InteractionEvent(
                            timestamp=time.time(), event_type='form_input',
                            element_tag=tag, element_id=iid,
                            element_class=await inp.get_attribute('class') or '',
                            element_text='', element_xpath=f"//{tag}",
                            url_before=url_before, url_after=page.url,
                            form_data={iname: val}
                        ))
                    elif itype in ('checkbox', 'radio'):
                        try:
                            await inp.check()
                        except Exception:
                            pass
                    elif tag == 'textarea':
                        await inp.fill('Test submission content')
                        form_data[iname or 'textarea'] = 'test'
                    elif tag == 'select':
                        opts = await inp.query_selector_all('option')
                        if len(opts) > 1:
                            await opts[1].click()
                except Exception:
                    continue

            submit_btn = await form.query_selector(
                'input[type="submit"], button[type="submit"], button:not([type])'
            )
            if submit_btn:
                # Save form attributes BEFORE clicking (page may navigate)
                try:
                    form_id = await form.get_attribute('id') or ''
                except Exception:
                    form_id = ''
                try:
                    form_class = await form.get_attribute('class') or ''
                except Exception:
                    form_class = ''

                try:
                    await submit_btn.click()
                    await page.wait_for_timeout(3000)
                except Exception:
                    try:
                        await form.evaluate('f => f.submit()')
                        await page.wait_for_timeout(3000)
                    except Exception:
                        pass

                # Screenshot after form submission
                submit_ss = await self._take_screenshot(
                    page, f'after_submit_form{form_idx}', trace_id)

                events.append(InteractionEvent(
                    timestamp=time.time(), event_type='submit',
                    element_tag='form',
                    element_id=form_id,
                    element_class=form_class,
                    element_text='', element_xpath='//form',
                    url_before=url_before, url_after=page.url,
                    form_data=form_data,
                    screenshot_path=submit_ss,
                ))

                # --- Dual-submission detection ---
                # After form submission, check if the page shows an error and
                # re-displays credential fields (indicative of dual-submission attack)
                dual_submit_events = await self._check_dual_submission(
                    page, form_data, url_before, trace_id=trace_id
                )
                events.extend(dual_submit_events)

        except Exception as e:
            logger.debug(f"_fill_and_submit_form error: {e}")

        return events

    async def _check_dual_submission(self, page: Page, original_form_data: Dict,
                                      original_url: str, trace_id: str = '') -> List[InteractionEvent]:
        """
        Advanced post-submission analysis combining three detection strategies:
        
        1. **Dual-submission detection**: After first submit, if the page shows
           an error AND re-displays credential fields, re-fill and re-submit
           (up to MAX_RESUBMITS times) to capture the full attack flow.
           Many phishing kits show a fake "incorrect password" on the first
           attempt, then accept the same credentials on a subsequent attempt.
        
        2. **Unconditional-accept detection**: After submission, if the page
           immediately shows success indicators (welcome, dashboard, logged in)
           for completely random credentials, flag it — no legitimate service
           accepts random passwords.
        
        3. **Progressive retry**: Re-submit with the SAME credentials up to
           MAX_RESUBMITS times. If the page flips from error to success on
           any retry (common phishing pattern), record the full sequence.
        """
        MAX_RESUBMITS = 3
        events = []
        try:
            page_text = await page.evaluate(
                '() => document.body ? document.body.innerText : ""')
            page_text_lower = page_text.lower()

            # ── Success indicators (check FIRST — unconditional accept) ──
            success_patterns = [
                'welcome', 'dashboard', 'logged in', 'sign out',
                'my account', 'your account', 'inbox', 'log out',
                'successfully', 'verification complete', 'thank you',
            ]
            has_success = any(p in page_text_lower for p in success_patterns)

            # Detect unconditional accept: random credentials immediately
            # accepted = strong phishing signal
            if has_success:
                url_changed = page.url != original_url
                events.append(InteractionEvent(
                    timestamp=time.time(),
                    event_type='unconditional_accept',
                    element_tag='page', element_id='success_detection',
                    element_class='', element_text=page_text[:300],
                    element_xpath='//body',
                    url_before=original_url, url_after=page.url,
                    form_data={
                        'pattern': 'unconditional_accept',
                        'success_text_found': True,
                        'url_changed': url_changed,
                    },
                ))
                logger.info("Unconditional-accept detected: random "
                            "credentials accepted immediately")
                # Still continue to check for further forms on the
                # "success" page (multi-stage phishing may ask for more
                # data like SSN, credit card after fake login success)
                return events

            # ── Error indicators ──
            error_patterns = [
                'incorrect password', 'wrong password', 'invalid password',
                'authentication failed', 'login failed', 'try again',
                'invalid credentials', 'account not found', 'error',
                'mot de passe incorrect', 'contraseña incorrecta',
                'password is incorrect', 'please try again',
                'unable to sign in', 'sign-in failed',
                'account does not exist', 'user not found',
            ]
            has_error = any(p in page_text_lower for p in error_patterns)

            if not has_error:
                return events

            # ── Check if credential fields are still present ──
            cred_selectors = [
                'input[type="password"]',
                'input[type="email"]',
                'input[name*="pass"]',
                'input[name*="email"]',
                'input[name*="user"]',
            ]
            has_cred_fields = False
            for sel in cred_selectors:
                try:
                    els = await page.query_selector_all(sel)
                    visible = [e for e in els if await e.is_visible()]
                    if visible:
                        has_cred_fields = True
                        break
                except Exception:
                    continue

            if not has_cred_fields:
                return events

            # ── Dual-submission / progressive-retry loop ──
            logger.info("Dual-submission pattern detected: error + "
                        "credential fields re-displayed")
            events.append(InteractionEvent(
                timestamp=time.time(), event_type='dual_submit_error',
                element_tag='page', element_id='error_detection',
                element_class='', element_text=page_text[:300],
                element_xpath='//body',
                url_before=page.url, url_after=page.url,
                form_data={'pattern': 'dual_submission',
                           'error_detected': True},
            ))

            # Re-submit up to MAX_RESUBMITS times.  On each retry we
            # ROTATE to a different persona so we can distinguish:
            #   - "delayed accept" (same creds eventually pass → phishing)
            #   - "accepts any creds" (different creds also pass → phishing)
            original_persona = self._persona_idx
            for attempt in range(1, MAX_RESUBMITS + 1):
                # Rotate persona so each retry uses different credentials
                self._rotate_persona()
                retry_persona = self._PERSONA_POOL[self._persona_idx]

                url_before_resubmit = page.url
                submitted = False
                forms = await page.query_selector_all('form')
                for form in forms[:3]:
                    try:
                        inputs = await form.query_selector_all(
                            'input, textarea')
                        for inp in inputs:
                            itype = (await inp.get_attribute('type')
                                     or 'text').lower()
                            iname = await inp.get_attribute('name') or ''
                            iid = await inp.get_attribute('id') or ''
                            if itype in ('text', 'email', 'password', 'tel'):
                                val = self._get_test_value(itype, iname, iid)
                                try:
                                    await inp.fill(val)
                                except Exception:
                                    pass

                        submit_btn = await form.query_selector(
                            'input[type="submit"], button[type="submit"], '
                            'button:not([type])')
                        if submit_btn:
                            await submit_btn.click()
                            await page.wait_for_timeout(3000)
                            submitted = True

                            ds_ss = await self._take_screenshot(
                                page, f'after_dual_resubmit_{attempt}',
                                trace_id)

                            events.append(InteractionEvent(
                                timestamp=time.time(),
                                event_type='dual_submit_resubmit',
                                element_tag='form', element_id='resubmit',
                                element_class='',
                                element_text='',
                                element_xpath='//form',
                                url_before=url_before_resubmit,
                                url_after=page.url,
                                form_data={
                                    'pattern': 'dual_submission',
                                    'attempt': attempt,
                                    'resubmit': True,
                                    'persona_email': retry_persona['email'],
                                    'different_creds': True,
                                    'redirect_to': page.url,
                                },
                                screenshot_path=ds_ss,
                            ))
                            break
                    except Exception as e:
                        logger.debug(f"Dual-submit attempt {attempt} "
                                     f"error: {e}")

                if not submitted:
                    break

                # After each retry, check if page flipped to success
                try:
                    retry_text = await page.evaluate(
                        '() => document.body ? document.body.innerText : ""')
                    retry_lower = retry_text.lower()
                    if any(p in retry_lower for p in success_patterns):
                        # Success with DIFFERENT credentials = strong
                        # phishing signal (no legit site does this)
                        events.append(InteractionEvent(
                            timestamp=time.time(),
                            event_type='delayed_accept',
                            element_tag='page',
                            element_id='success_after_retry',
                            element_class='',
                            element_text=retry_text[:300],
                            element_xpath='//body',
                            url_before=url_before_resubmit,
                            url_after=page.url,
                            form_data={
                                'pattern': 'delayed_accept',
                                'attempts_before_success': attempt,
                                'used_different_creds': True,
                                'persona_email': retry_persona['email'],
                            },
                        ))
                        logger.info(f"Delayed-accept detected on "
                                    f"attempt {attempt} (different creds)")
                        break
                    # If page navigated away, also stop
                    if page.url != url_before_resubmit:
                        break
                    # If no more credential fields, stop
                    still_has_fields = False
                    for sel in cred_selectors:
                        try:
                            els = await page.query_selector_all(sel)
                            visible = [e for e in els
                                       if await e.is_visible()]
                            if visible:
                                still_has_fields = True
                                break
                        except Exception:
                            continue
                    if not still_has_fields:
                        break
                except Exception:
                    break

            # Restore original persona for the rest of the crawl
            self._persona_idx = original_persona

        except Exception as e:
            logger.debug(f"_check_dual_submission error: {e}")

        return events

    def _get_test_value(self, input_type: str, name: str = '', el_id: str = '') -> str:
        """Return a contextually appropriate test value for the given field."""
        td = self.TEST_DATA
        combined = (name + ' ' + el_id).lower()

        # ── Email / username / password ──
        if 'email' in combined or 'mail' in combined:
            return td['email']
        if 'pass' in combined or 'pwd' in combined:
            return td['password']
        if 'user' in combined or 'login' in combined or 'uname' in combined:
            return td['username']

        # ── Personal identity ──
        if 'first' in combined and 'name' in combined:
            return td['first_name']
        if 'last' in combined and 'name' in combined:
            return td['last_name']
        if 'full' in combined and 'name' in combined:
            return td['text']
        if 'name' in combined:          # generic "name" field
            return td['text']

        # ── Phone / contact ──
        if 'phone' in combined or 'tel' in combined or 'mobile' in combined:
            return td['tel']

        # ── Address ──
        if 'address' in combined or 'street' in combined:
            return td['address']
        if 'city' in combined or 'town' in combined:
            return td['city']
        if 'state' in combined or 'province' in combined or 'region' in combined:
            return td['state']
        if 'zip' in combined or 'postal' in combined:
            return td['zip']
        if 'country' in combined:
            return td['country']

        # ── Financial ──
        if 'card' in combined or 'cc_num' in combined or 'credit' in combined:
            return td['cc_number']
        if 'cvv' in combined or 'cvc' in combined or 'security_code' in combined:
            return td['cc_cvv']
        if 'expir' in combined or 'exp_date' in combined or 'cc_exp' in combined:
            return td['cc_exp']

        # ── Sensitive identity ──
        if 'ssn' in combined or 'social' in combined or 'national_id' in combined:
            return td['ssn']
        if 'dob' in combined or 'birth' in combined or 'birthday' in combined or 'date_of' in combined:
            return td['dob']

        # ── Verification / OTP / PIN ──
        if 'otp' in combined or 'one_time' in combined or 'verif' in combined or 'code' in combined:
            return td['otp']
        if 'pin' in combined:
            return td['pin']

        # ── Organisation ──
        if 'company' in combined or 'org' in combined or 'employer' in combined:
            return td['company']

        # ── Fallback by input type ──
        if input_type == 'email':
            return td['email']
        if input_type == 'password':
            return td['password']
        if input_type == 'tel':
            return td['tel']

        return td.get(input_type, 'testvalue123')

    def _fallback_crawl(self, url: str) -> InteractionTrace:
        """Fallback when Playwright is unavailable."""
        import requests as req_lib
        start = time.time()
        try:
            resp = req_lib.get(url, timeout=10, verify=False, allow_redirects=True)
            final_url = resp.url
            redirects = [r.url for r in resp.history] + [resp.url]
        except Exception:
            final_url = url
            redirects = [url]
        return InteractionTrace(
            url=url, start_time=start, end_time=time.time(),
            events=[], final_url=final_url, redirects=redirects,
            forms_submitted=0, elements_interacted=0, page_title="(fallback)")

    def save_trace(self, trace: InteractionTrace, output_path: str):
        trace_dict = asdict(trace)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(trace_dict, f, indent=2, ensure_ascii=False, default=str)


async def main():
    logging.basicConfig(level=logging.INFO)
    crawler = PhishingCrawler(headless=True)
    trace = await crawler.crawl("https://example.com", max_depth=2)
    print(f"Events: {len(trace.events)}, Forms: {trace.forms_submitted}")
    crawler.save_trace(trace, 'interaction_trace.json')

if __name__ == "__main__":
    asyncio.run(main())
