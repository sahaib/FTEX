#!/usr/bin/env python3
"""
FTEX Smart Detection Engine v6.0
================================
Self-validating, evidence-based AI analysis engine.

Features:
- 🧠 Pure GenAI analysis with self-validation
- 📊 Evidence-based findings (every insight has ticket IDs)
- 🎯 Confidence scoring (High/Medium/Low)
- 💡 Solution quality analysis
- 🔧 Fully configurable via UserConfig
- 📚 Knowledge base ready (future RAG)
- ⚡ Fallback to statistical when AI unavailable

Uses config.py for SLA/Ollama settings, UserConfig for domain-specific settings.

Author: FTEX Project
License: MIT
"""

import re
import html
import json
import os
import hashlib
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import centralized config
try:
    from config import config as app_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    app_config = None


# =============================================================================
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         USER CONFIGURATION                                 ║
# ║              Edit this section for your product/domain                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# =============================================================================

class UserConfig:
    """
    Configure this for YOUR product/domain.
    All analysis adapts based on these settings.
    
    Note: SLA and Ollama settings come from config.py
    """
    
    # =========================================================================
    # ENTITY CONFIGURATION
    # What primary entity do you track tickets by?
    # =========================================================================
    ENTITY_NAME = "entity"              # e.g., "vessel", "store", "device", "account"
    ENTITY_NAME_PLURAL = "entities"     # Plural form
    
    # Regex patterns to extract entity from ticket text
    ENTITY_PATTERNS = [
        # Examples (uncomment/modify for your domain):
        # r'(?:vessel|ship|mv|m/v)[:\s]+([A-Z][A-Za-z0-9\s\-]{2,25})',  # Maritime
        # r'(?:store|location|branch)[:\s#]+(\w+)',                      # Retail
        # r'(?:device|serial)[:\s#]+([A-Z0-9\-]+)',                      # IoT
        r'(?:id|ref|reference)[:\s#]+([A-Z0-9\-]+)',                    # Generic
    ]
    
    # JSON path to entity in ticket data (optional)
    ENTITY_JSON_PATH = None  # e.g., ['custom_fields', 'cf_vessel_name']
    
    # =========================================================================
    # PRODUCT CONTEXT
    # Help AI understand your product for better analysis
    # =========================================================================
    PRODUCT_NAME = "Your Product"
    PRODUCT_DESCRIPTION = """
    Brief description of what your product does.
    This helps AI understand context when analyzing tickets.
    """
    
    PRODUCT_MODULES = [
        # "Module A",
        # "Module B",
    ]
    
    KNOWN_LIMITATIONS = [
        # "Known limitation 1",
    ]
    
    # =========================================================================
    # KNOWLEDGE BASE (RAG-Ready)
    # =========================================================================
    GLOSSARY = {
        # "term": "definition",
    }
    
    KNOWN_SOLUTIONS = {
        # "issue_pattern": {
        #     "steps": ["step1", "step2"],
        #     "root_cause": "cause",
        #     "prevention": "how to prevent"
        # },
    }
    
    ESCALATION_TRIGGERS = [
        "data loss", "compliance", "audit", "legal", "security breach",
        "all users affected", "production down", "cannot operate"
    ]
    
    # =========================================================================
    # ANOMALY DETECTION THRESHOLDS
    # =========================================================================
    DUPLICATE_REQUEST_DAYS = 365
    DUPLICATE_REQUEST_KEYWORDS = ["activation", "license", "renewal"]
    RECURRING_ISSUE_THRESHOLD = 3
    HIGH_FREQUENCY_MULTIPLIER = 3.0
    SPIKE_MULTIPLIER = 2.0
    
    # Confidence thresholds
    HIGH_CONFIDENCE_MIN_EVIDENCE = 10
    MEDIUM_CONFIDENCE_MIN_EVIDENCE = 3
    
    # =========================================================================
    # AI SETTINGS
    # =========================================================================
    AI_BATCH_SIZE = 30
    AI_VALIDATION_ENABLED = True
    CACHE_CATEGORIES = True
    CACHE_FILE = "analysis_cache.json"


# =============================================================================
# SYSTEM CONFIGURATION (from config.py or fallback)
# =============================================================================

class SystemConfig:
    """System configuration - reads from config.py when available."""
    
    @staticmethod
    def get_ollama_url() -> str:
        if CONFIG_AVAILABLE and app_config:
            return app_config.ollama.base_url
        return os.getenv('OLLAMA_URL', 'http://localhost:11434')
    
    @staticmethod
    def get_ollama_model() -> str:
        if CONFIG_AVAILABLE and app_config:
            return app_config.ollama.model
        return os.getenv('OLLAMA_MODEL', 'qwen3:14b')
    
    @staticmethod
    def get_sla_first_response() -> Dict[str, int]:
        if CONFIG_AVAILABLE and app_config:
            return app_config.sla.first_response
        return {'Urgent': 1, 'High': 4, 'Medium': 8, 'Low': 24}
    
    @staticmethod
    def get_sla_resolution() -> Dict[str, int]:
        if CONFIG_AVAILABLE and app_config:
            return app_config.sla.resolution
        return {'Urgent': 4, 'High': 24, 'Medium': 72, 'Low': 168}
    
    # Constants
    OLLAMA_TIMEOUT = 300
    MAX_ACK_LENGTH = 200
    POSITIVE_WORDS = {
        'thanks', 'thank', 'great', 'perfect', 'works', 'working',
        'resolved', 'fixed', 'ok', 'okay', 'good', 'awesome',
        'excellent', 'cheers', 'noted', 'confirmed', 'done',
        'sorted', 'understood', 'received', 'appreciated',
    }
    ACKNOWLEDGMENT_PATTERNS = [
        r'^thanks?\.?!?$',
        r'^thank\s*you\.?!?$',
        r'^(got\s+it|ok|okay|noted|understood|perfect|great)\.?!?$',
        r'^(works?|working)\s*(now|fine|great)?\.?!?$',
        r'^(issue\s+)?(resolved|fixed|solved|sorted)\.?!?$',
        r'^please\s+close\.?!?$',
        r'^cheers\.?!?$',
        r'^confirmed\.?!?$',
    ]


# Compile patterns
_COMPILED_ACK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SystemConfig.ACKNOWLEDGMENT_PATTERNS]
_COMPILED_ENTITY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in UserConfig.ENTITY_PATTERNS]


# =============================================================================
# UTILITIES
# =============================================================================

def clean_html(text: str) -> str:
    """Remove HTML tags and clean text."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse ISO datetime string."""
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.replace(tzinfo=None)
    except:
        try:
            return datetime.fromisoformat(dt_str[:19])
        except:
            return None


def get_ticket_text(ticket: dict, include_conversations: bool = True) -> str:
    """Extract all text from a ticket."""
    parts = []
    
    if ticket.get('subject'):
        parts.append(f"Subject: {ticket['subject']}")
    
    if ticket.get('description'):
        parts.append(f"Description: {clean_html(ticket['description'])}")
    
    if include_conversations:
        for conv in ticket.get('conversations', []):
            body = clean_html(conv.get('body_text') or conv.get('body', ''))
            if body:
                source = "Customer" if conv.get('incoming', True) else "Agent"
                parts.append(f"{source}: {body}")
    
    return '\n'.join(parts)


def get_solution_from_ticket(ticket: dict) -> Optional[str]:
    """Extract the solution/resolution from a closed ticket."""
    conversations = ticket.get('conversations', [])
    if not conversations:
        return None
    
    sorted_convos = sorted(conversations, key=lambda x: x.get('created_at', ''), reverse=True)
    
    for conv in sorted_convos:
        if not conv.get('incoming', True):
            body = clean_html(conv.get('body_text') or conv.get('body', ''))
            if body and len(body) > 20:
                return body
    
    return None


def get_nested_value(data: dict, path: List[str]) -> Any:
    """Get value from nested dict using path."""
    if not path:
        return None
    current = data
    for key in path:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


# =============================================================================
# ZOMBIE DETECTION
# =============================================================================

def is_acknowledgment_message(text: str) -> bool:
    """Check if a message is just an acknowledgment."""
    if not text:
        return False
    
    cleaned = clean_html(text).strip()
    
    if len(cleaned) > SystemConfig.MAX_ACK_LENGTH:
        return False
    
    for pattern in _COMPILED_ACK_PATTERNS:
        if pattern.search(cleaned):
            return True
    
    words = cleaned.lower().split()
    if len(words) <= 5:
        if any(w.rstrip('.,!?') in SystemConfig.POSITIVE_WORDS for w in words):
            return True
    
    return False


def is_true_zombie_ticket(ticket: dict) -> Tuple[bool, str]:
    """
    Determine if ticket is a TRUE zombie (actually needs response).
    
    Returns: (is_zombie, reason)
    """
    conversations = ticket.get('conversations', [])
    
    if not conversations:
        return True, "No conversations - never responded"
    
    sorted_convos = sorted(conversations, key=lambda x: x.get('created_at', ''))
    
    has_agent_response = any(not c.get('incoming', True) for c in sorted_convos)
    
    if not has_agent_response:
        last = sorted_convos[-1]
        if last.get('incoming', False):
            body = last.get('body_text') or last.get('body', '')
            if is_acknowledgment_message(body):
                return False, "Customer acknowledgment only"
        return True, "No agent response"
    
    last_agent_idx = -1
    for i, c in enumerate(sorted_convos):
        if not c.get('incoming', True):
            last_agent_idx = i
    
    if last_agent_idx < len(sorted_convos) - 1:
        messages_after = sorted_convos[last_agent_idx + 1:]
        customer_messages = [m for m in messages_after if m.get('incoming', False)]
        
        if customer_messages:
            all_acks = all(
                is_acknowledgment_message(m.get('body_text') or m.get('body', ''))
                for m in customer_messages
            )
            if not all_acks:
                return False, "Awaiting follow-up (customer replied)"
    
    return False, "Has agent response"


def get_zombie_stats(tickets: List[dict]) -> Dict[str, Any]:
    """Get zombie ticket statistics."""
    true_zombies = []
    false_positives = []
    
    for ticket in tickets:
        is_zombie, reason = is_true_zombie_ticket(ticket)
        
        entry = {
            'ticket_id': ticket.get('id'),
            'subject': ticket.get('subject', '')[:100],
            'created_at': ticket.get('created_at'),
            'status': ticket.get('status_name'),
            'priority': ticket.get('priority_name'),
            'reason': reason,
        }
        
        if is_zombie:
            true_zombies.append(entry)
        elif 'acknowledgment' in reason.lower():
            false_positives.append(entry)
    
    return {
        'true_zombie_count': len(true_zombies),
        'false_positive_count': len(false_positives),
        'zombie_rate': round(len(true_zombies) / len(tickets) * 100, 2) if tickets else 0,
        'true_zombies': true_zombies,
        'false_positives': false_positives,
    }


# =============================================================================
# OLLAMA CLIENT
# =============================================================================

class OllamaClient:
    """Client for Ollama LLM."""
    
    def __init__(self):
        self.base_url = SystemConfig.get_ollama_url()
        self.model = SystemConfig.get_ollama_model()
        self._available = None
    
    @property
    def available(self) -> bool:
        if self._available is None:
            self.check_connection()
        return self._available
    
    def check_connection(self) -> bool:
        """Check if Ollama is running."""
        try:
            import requests
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            
            if resp.status_code == 200:
                models = resp.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                for m in model_names:
                    if self.model in m or m in self.model:
                        self.model = m
                        self._available = True
                        return True
                
                if model_names:
                    self.model = model_names[0]
                    self._available = True
                    return True
            
            self._available = False
            return False
            
        except Exception:
            self._available = False
            return False
    
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 4000) -> Optional[str]:
        """Generate completion."""
        if not self.available:
            return None
        
        try:
            import requests
            
            if "/no_think" not in prompt:
                prompt = prompt + "\n\n/no_think"
            
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': temperature,
                        'num_predict': max_tokens,
                    }
                },
                timeout=SystemConfig.OLLAMA_TIMEOUT
            )
            
            if resp.status_code == 200:
                response = resp.json().get('response', '').strip()
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                return response
                
        except Exception as e:
            print(f"Ollama error: {e}")
        
        return None
    
    def generate_json(self, prompt: str, temperature: float = 0.2) -> Optional[dict]:
        """Generate and parse JSON."""
        response = self.generate(prompt, temperature)
        
        if response:
            try:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    return json.loads(json_match.group())
                
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    return json.loads(json_match.group())
                    
            except json.JSONDecodeError:
                pass
        
        return None


# =============================================================================
# CATEGORY DISCOVERY ENGINE
# =============================================================================

class CategoryDiscoveryEngine:
    """Discovers issue categories from ticket data."""
    
    def __init__(self, ollama: OllamaClient):
        self.ollama = ollama
        self.categories = {}
        self.category_evidence = defaultdict(list)
    
    def discover_categories(self, tickets: List[dict], progress_callback=None) -> Dict[str, Dict]:
        """Discover categories using AI."""
        if not self.ollama.available:
            return self._fallback_discovery(tickets)
        
        product_context = self._build_product_context()
        sample = self._get_diverse_sample(tickets, min(150, len(tickets)))
        
        if progress_callback:
            progress_callback("Analyzing ticket patterns...")
        
        summaries = self._prepare_ticket_summaries(sample[:50])
        
        prompt = f"""You are analyzing support tickets for: {UserConfig.PRODUCT_NAME}

{product_context}

Here are sample support tickets:
{json.dumps(summaries, indent=2)}

Based on these tickets, identify 10-20 distinct issue CATEGORIES.

For each category provide:
1. A short snake_case name (e.g., "login_authentication")
2. Keywords that identify tickets in this category
3. Brief description
4. Typical root causes

Return ONLY valid JSON:
{{
  "category_name": {{
    "keywords": ["keyword1", "keyword2"],
    "description": "What this category covers",
    "typical_root_causes": ["cause1", "cause2"]
  }}
}}"""

        result = self.ollama.generate_json(prompt)
        
        if result and isinstance(result, dict):
            self.categories = result
        else:
            return self._fallback_discovery(tickets)
        
        if progress_callback:
            progress_callback("Validating categories...")
        
        self._collect_evidence(tickets, progress_callback)
        self._refine_categories()
        
        return self.categories
    
    def _build_product_context(self) -> str:
        """Build product context for AI."""
        parts = [f"Product: {UserConfig.PRODUCT_NAME}"]
        
        if UserConfig.PRODUCT_DESCRIPTION:
            parts.append(f"Description: {UserConfig.PRODUCT_DESCRIPTION.strip()}")
        
        if UserConfig.PRODUCT_MODULES:
            parts.append(f"Key Modules: {', '.join(UserConfig.PRODUCT_MODULES)}")
        
        if UserConfig.GLOSSARY:
            glossary_str = ', '.join(f"{k}={v}" for k, v in list(UserConfig.GLOSSARY.items())[:10])
            parts.append(f"Glossary: {glossary_str}")
        
        return '\n'.join(parts)
    
    def _get_diverse_sample(self, tickets: List[dict], n: int) -> List[dict]:
        """Get diverse sample."""
        if len(tickets) <= n:
            return tickets
        
        sampled = []
        seen_ids = set()
        
        by_status = defaultdict(list)
        for t in tickets:
            by_status[t.get('status_name', 'Unknown')].append(t)
        
        per_status = max(1, n // (len(by_status) + 1))
        for status, tix in by_status.items():
            for t in tix[:per_status]:
                if t.get('id') not in seen_ids:
                    sampled.append(t)
                    seen_ids.add(t.get('id'))
        
        for t in tickets:
            if len(sampled) >= n:
                break
            if t.get('id') in seen_ids:
                continue
            if is_true_zombie_ticket(t)[0] or t.get('priority_name') in ['Urgent', 'High']:
                sampled.append(t)
                seen_ids.add(t.get('id'))
        
        import random
        remaining = [t for t in tickets if t.get('id') not in seen_ids]
        random.shuffle(remaining)
        sampled.extend(remaining[:n - len(sampled)])
        
        return sampled[:n]
    
    def _prepare_ticket_summaries(self, tickets: List[dict]) -> List[Dict]:
        """Prepare ticket summaries for AI."""
        summaries = []
        for t in tickets:
            summaries.append({
                'id': t.get('id'),
                'subject': t.get('subject', '')[:150],
                'description': clean_html(t.get('description', ''))[:300],
                'status': t.get('status_name'),
                'priority': t.get('priority_name'),
            })
        return summaries
    
    def _collect_evidence(self, tickets: List[dict], progress_callback=None):
        """Categorize all tickets."""
        self.category_evidence = defaultdict(list)
        
        for i, ticket in enumerate(tickets):
            categories = self._categorize_ticket(ticket)
            for cat in categories:
                self.category_evidence[cat].append(ticket.get('id'))
            
            if progress_callback and (i + 1) % 500 == 0:
                progress_callback(f"Categorized {i + 1}/{len(tickets)} tickets...")
    
    def _categorize_ticket(self, ticket: dict) -> List[str]:
        """Categorize a ticket using keywords."""
        if not self.categories:
            return ['uncategorized']
        
        text = get_ticket_text(ticket, include_conversations=False).lower()
        matched = []
        
        for category, data in self.categories.items():
            keywords = data.get('keywords', [])
            for kw in keywords:
                if kw.lower() in text:
                    matched.append(category)
                    break
        
        return matched if matched else ['uncategorized']
    
    def _refine_categories(self):
        """Refine categories based on evidence."""
        min_tickets = max(3, len(self.category_evidence) // 20)
        
        to_remove = []
        for cat, ticket_ids in self.category_evidence.items():
            if cat != 'uncategorized' and len(ticket_ids) < min_tickets:
                to_remove.append(cat)
        
        for cat in to_remove:
            if cat in self.categories:
                del self.categories[cat]
            self.category_evidence['uncategorized'].extend(self.category_evidence[cat])
            del self.category_evidence[cat]
        
        for cat in self.categories:
            self.categories[cat]['ticket_count'] = len(self.category_evidence.get(cat, []))
            self.categories[cat]['ticket_ids'] = self.category_evidence.get(cat, [])[:50]
    
    def _fallback_discovery(self, tickets: List[dict]) -> Dict[str, Dict]:
        """Fallback: keyword extraction."""
        word_counts = Counter()
        
        for ticket in tickets:
            text = get_ticket_text(ticket, include_conversations=False).lower()
            words = re.findall(r'\b[a-z]{4,15}\b', text)
            
            stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'would', 'could', 
                         'should', 'there', 'their', 'about', 'which', 'when', 'what', 
                         'please', 'thank', 'thanks', 'hello', 'regards', 'team', 'support'}
            words = [w for w in words if w not in stop_words]
            word_counts.update(words)
        
        categories = {}
        for word, count in word_counts.most_common(15):
            if count >= 5:
                categories[word] = {
                    'keywords': [word],
                    'description': f'Tickets mentioning {word}',
                    'typical_root_causes': ['Statistical detection'],
                    'ticket_count': count,
                }
        
        self.categories = categories
        self._collect_evidence(tickets, None)
        return categories


# =============================================================================
# SOLUTION ANALYZER
# =============================================================================

class SolutionAnalyzer:
    """Analyzes solution quality."""
    
    def __init__(self, ollama: OllamaClient):
        self.ollama = ollama
    
    def analyze_solutions(self, tickets: List[dict], categories: Dict, progress_callback=None) -> Dict:
        """Analyze solution quality."""
        resolved = [t for t in tickets 
                   if t.get('status_name') in ['Resolved', 'Closed'] 
                   and t.get('conversations')]
        
        if not resolved:
            return {'analyzed': 0, 'solutions': []}
        
        solutions = []
        quality_scores = []
        
        for i, ticket in enumerate(resolved):
            solution_text = get_solution_from_ticket(ticket)
            if not solution_text:
                continue
            
            ticket_text = get_ticket_text(ticket, include_conversations=False).lower()
            ticket_category = 'unknown'
            for cat, data in categories.items():
                if any(kw.lower() in ticket_text for kw in data.get('keywords', [])):
                    ticket_category = cat
                    break
            
            quality = self._assess_solution_quality(ticket, solution_text, ticket_category)
            quality['ticket_id'] = ticket.get('id')
            quality['category'] = ticket_category
            solutions.append(quality)
            quality_scores.append(quality['score'])
            
            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(f"Analyzed {i + 1}/{len(resolved)} solutions...")
        
        score_distribution = Counter(s['rating'] for s in solutions)
        
        return {
            'analyzed': len(solutions),
            'average_score': round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else 0,
            'distribution': dict(score_distribution),
            'solutions': solutions,
            'best_solutions': sorted([s for s in solutions if s['score'] >= 8], 
                                    key=lambda x: x['score'], reverse=True)[:20],
            'poor_solutions': sorted([s for s in solutions if s['score'] <= 4],
                                    key=lambda x: x['score'])[:20],
        }
    
    def _assess_solution_quality(self, ticket: dict, solution: str, category: str) -> Dict:
        """Assess solution quality."""
        score = 5
        factors = []
        
        if len(solution) < 50:
            score -= 2
            factors.append("Very brief response")
        elif len(solution) > 200:
            score += 1
            factors.append("Detailed response")
        
        step_indicators = ['step', 'first', 'then', 'next', 'finally', '1.', '2.', '1)', '2)']
        if any(ind in solution.lower() for ind in step_indicators):
            score += 1
            factors.append("Contains actionable steps")
        
        verify_indicators = ['let me know', 'confirm', 'verify', 'check if', 'please try']
        if any(ind in solution.lower() for ind in verify_indicators):
            score += 1
            factors.append("Requests verification")
        
        if category in UserConfig.KNOWN_SOLUTIONS:
            known = UserConfig.KNOWN_SOLUTIONS[category]
            known_steps = known.get('steps', [])
            matches = sum(1 for step in known_steps if step.lower() in solution.lower())
            if matches > 0:
                score += min(2, matches)
                factors.append(f"Matches {matches} known solution steps")
        
        root_cause_indicators = ['cause', 'reason', 'because', 'due to', 'result of']
        if any(ind in solution.lower() for ind in root_cause_indicators):
            score += 1
            factors.append("Addresses root cause")
        
        prevention_indicators = ['prevent', 'avoid', 'future', 'going forward', 'recommend']
        if any(ind in solution.lower() for ind in prevention_indicators):
            score += 1
            factors.append("Includes prevention advice")
        
        score = max(1, min(10, score))
        
        if score >= 8:
            rating = 'excellent'
        elif score >= 6:
            rating = 'good'
        elif score >= 4:
            rating = 'acceptable'
        else:
            rating = 'poor'
        
        return {
            'score': score,
            'rating': rating,
            'factors': factors,
            'solution_preview': solution[:200] + '...' if len(solution) > 200 else solution,
        }


# =============================================================================
# FINDING
# =============================================================================

class Finding:
    """Evidence-based finding."""
    
    def __init__(self, 
                 finding_type: str,
                 title: str,
                 description: str,
                 evidence_ticket_ids: List[int],
                 confidence: str = 'medium',
                 severity: str = 'medium',
                 recommendation: str = None,
                 root_cause: str = None):
        
        self.finding_type = finding_type
        self.title = title
        self.description = description
        self.evidence_ticket_ids = evidence_ticket_ids
        self.evidence_count = len(evidence_ticket_ids)
        self.confidence = confidence
        self.severity = severity
        self.recommendation = recommendation
        self.root_cause = root_cause
        self.validated = False
    
    def calculate_confidence(self):
        """Calculate confidence based on evidence."""
        if self.evidence_count >= UserConfig.HIGH_CONFIDENCE_MIN_EVIDENCE:
            self.confidence = 'high'
        elif self.evidence_count >= UserConfig.MEDIUM_CONFIDENCE_MIN_EVIDENCE:
            self.confidence = 'medium'
        else:
            self.confidence = 'low'
    
    def to_dict(self) -> Dict:
        return {
            'type': self.finding_type,
            'title': self.title,
            'description': self.description,
            'evidence_count': self.evidence_count,
            'ticket_ids': self.evidence_ticket_ids[:20],
            'confidence': self.confidence,
            'severity': self.severity,
            'recommendation': self.recommendation,
            'root_cause': self.root_cause,
            'validated': self.validated,
        }


# =============================================================================
# ANALYSIS ENGINE (Main)
# =============================================================================

class AnalysisEngine:
    """Main analysis engine."""
    
    def __init__(self, tickets: List[dict], use_ai: bool = True):
        self.tickets = tickets
        self.use_ai = use_ai
        self.ollama = OllamaClient() if use_ai else None
        
        self.category_engine = CategoryDiscoveryEngine(self.ollama) if use_ai else None
        self.solution_analyzer = SolutionAnalyzer(self.ollama) if use_ai else None
        
        self.categories = {}
        self._cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cached categories."""
        if UserConfig.CACHE_CATEGORIES:
            cache_path = Path(UserConfig.CACHE_FILE)
            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as f:
                        self._cache = json.load(f)
                except:
                    pass
    
    def _save_cache(self):
        """Save categories to cache."""
        if UserConfig.CACHE_CATEGORIES and self.categories:
            try:
                with open(UserConfig.CACHE_FILE, 'w') as f:
                    json.dump({
                        'categories': self.categories,
                        'timestamp': datetime.now().isoformat(),
                        'ticket_count': len(self.tickets),
                    }, f, indent=2)
            except:
                pass
    
    def run_analysis(self, progress_callback=None) -> Dict[str, Any]:
        """Run complete analysis."""
        
        results = {
            'metadata': {
                'analyzed_at': datetime.now().isoformat(),
                'total_tickets': len(self.tickets),
                'ai_enabled': self.use_ai and self.ollama and self.ollama.available,
                'entity_type': UserConfig.ENTITY_NAME,
                'product': UserConfig.PRODUCT_NAME,
            }
        }
        
        # Stage 1: Foundation
        if progress_callback:
            progress_callback("stage", "Building data foundation...", 0, 6)
        results['foundation'] = self._build_foundation()
        
        # Stage 2: Zombies
        if progress_callback:
            progress_callback("stage", "Analyzing zombie tickets...", 1, 6)
        results['zombies'] = get_zombie_stats(self.tickets)
        
        # Stage 3: Categories
        if progress_callback:
            progress_callback("stage", "Discovering issue categories...", 2, 6)
        
        cache_valid = (
            self._cache.get('categories') and 
            self._cache.get('ticket_count') == len(self.tickets)
        )
        
        if cache_valid:
            self.categories = self._cache['categories']
            if progress_callback:
                progress_callback("info", "Using cached categories")
        elif self.category_engine:
            def cat_progress(msg):
                if progress_callback:
                    progress_callback("info", msg)
            self.categories = self.category_engine.discover_categories(self.tickets, cat_progress)
            self._save_cache()
        else:
            self.categories = self._fallback_categories()
        
        results['categories'] = self._analyze_categories()
        
        # Stage 4: Entities
        if progress_callback:
            progress_callback("stage", f"Analyzing {UserConfig.ENTITY_NAME_PLURAL}...", 3, 6)
        results['entities'] = self._analyze_entities()
        
        # Stage 5: Temporal & Anomalies
        if progress_callback:
            progress_callback("stage", "Detecting patterns and anomalies...", 4, 6)
        results['temporal'] = self._analyze_temporal()
        results['anomalies'] = self._detect_anomalies()
        
        # Stage 6: Solutions
        if progress_callback:
            progress_callback("stage", "Analyzing solution quality...", 5, 6)
        
        if self.solution_analyzer:
            def sol_progress(msg):
                if progress_callback:
                    progress_callback("info", msg)
            results['solutions'] = self.solution_analyzer.analyze_solutions(
                self.tickets, self.categories, sol_progress
            )
        else:
            results['solutions'] = {'analyzed': 0, 'solutions': []}
        
        # SLA
        results['sla'] = self._analyze_sla()
        
        # Findings
        if progress_callback:
            progress_callback("stage", "Generating findings...", 6, 6)
        results['findings'] = self._generate_findings(results)
        
        return results
    
    def _build_foundation(self) -> Dict:
        """Build foundation statistics."""
        statuses = Counter(t.get('status_name', 'Unknown') for t in self.tickets)
        priorities = Counter(t.get('priority_name', 'Unknown') for t in self.tickets)
        sources = Counter(t.get('source_name', 'Unknown') for t in self.tickets)
        
        dates = [parse_datetime(t.get('created_at')) for t in self.tickets]
        dates = [d for d in dates if d]
        
        resolution_times = [t.get('resolution_time_hours') for t in self.tickets if t.get('resolution_time_hours')]
        
        return {
            'by_status': dict(statuses),
            'by_priority': dict(priorities),
            'by_source': dict(sources),
            'date_range': {
                'earliest': min(dates).strftime('%Y-%m-%d') if dates else None,
                'latest': max(dates).strftime('%Y-%m-%d') if dates else None,
                'span_days': (max(dates) - min(dates)).days if len(dates) >= 2 else 0,
            },
            'resolution_stats': {
                'count': len(resolution_times),
                'avg_hours': round(sum(resolution_times) / len(resolution_times), 1) if resolution_times else 0,
                'avg_days': round(sum(resolution_times) / len(resolution_times) / 24, 1) if resolution_times else 0,
            }
        }
    
    def _analyze_categories(self) -> Dict:
        """Analyze tickets by category."""
        category_data = {}
        
        for cat_name, cat_info in self.categories.items():
            ticket_ids = cat_info.get('ticket_ids', [])
            tickets_in_cat = [t for t in self.tickets if t.get('id') in ticket_ids]
            
            if not tickets_in_cat:
                keywords = cat_info.get('keywords', [])
                tickets_in_cat = []
                for t in self.tickets:
                    text = get_ticket_text(t, include_conversations=False).lower()
                    if any(kw.lower() in text for kw in keywords):
                        tickets_in_cat.append(t)
            
            if not tickets_in_cat:
                continue
            
            zombies = sum(1 for t in tickets_in_cat if is_true_zombie_ticket(t)[0])
            res_times = [t.get('resolution_time_hours') for t in tickets_in_cat if t.get('resolution_time_hours')]
            priorities = Counter(t.get('priority_name', 'Unknown') for t in tickets_in_cat)
            
            category_data[cat_name] = {
                'description': cat_info.get('description', ''),
                'keywords': cat_info.get('keywords', []),
                'typical_root_causes': cat_info.get('typical_root_causes', []),
                'total_tickets': len(tickets_in_cat),
                'zombie_count': zombies,
                'zombie_rate': round(zombies / len(tickets_in_cat) * 100, 1) if tickets_in_cat else 0,
                'by_priority': dict(priorities),
                'avg_resolution_hours': round(sum(res_times) / len(res_times), 1) if res_times else None,
                'avg_resolution_days': round(sum(res_times) / len(res_times) / 24, 1) if res_times else None,
                'ticket_ids': [t.get('id') for t in tickets_in_cat[:30]],
            }
        
        return category_data
    
    def _analyze_entities(self) -> Dict:
        """Analyze by entity."""
        entity_tickets = defaultdict(list)
        
        for ticket in self.tickets:
            entity = self._extract_entity(ticket)
            if entity:
                entity_tickets[entity].append(ticket)
        
        entity_data = {}
        
        for entity, tickets in sorted(entity_tickets.items(), key=lambda x: len(x[1]), reverse=True)[:100]:
            zombies = sum(1 for t in tickets if is_true_zombie_ticket(t)[0])
            res_times = [t.get('resolution_time_hours') for t in tickets if t.get('resolution_time_hours')]
            
            issue_types = Counter()
            for t in tickets:
                text = get_ticket_text(t, include_conversations=False).lower()
                for cat_name, cat_info in self.categories.items():
                    if any(kw.lower() in text for kw in cat_info.get('keywords', [])):
                        issue_types[cat_name] += 1
                        break
            
            dates = [parse_datetime(t.get('created_at')) for t in tickets]
            dates = [d for d in dates if d]
            
            entity_data[entity] = {
                'total_tickets': len(tickets),
                'zombie_count': zombies,
                'zombie_rate': round(zombies / len(tickets) * 100, 1),
                'issue_breakdown': dict(issue_types),
                'top_issue': issue_types.most_common(1)[0] if issue_types else ('unknown', 0),
                'avg_resolution_days': round(sum(res_times) / len(res_times) / 24, 1) if res_times else None,
                'date_range': {
                    'first': min(dates).strftime('%Y-%m-%d') if dates else None,
                    'last': max(dates).strftime('%Y-%m-%d') if dates else None,
                },
                'ticket_ids': [t.get('id') for t in tickets[:20]],
            }
        
        return {
            'total_entities': len(entity_tickets),
            'entities': entity_data,
        }
    
    def _extract_entity(self, ticket: dict) -> Optional[str]:
        """Extract entity from ticket."""
        if UserConfig.ENTITY_JSON_PATH:
            entity = get_nested_value(ticket, UserConfig.ENTITY_JSON_PATH)
            if entity:
                return str(entity)
        
        company = ticket.get('company', {})
        if company and company.get('name'):
            return company['name']
        
        text = get_ticket_text(ticket, include_conversations=False)
        for pattern in _COMPILED_ENTITY_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _analyze_temporal(self) -> Dict:
        """Analyze temporal patterns."""
        by_month = defaultdict(list)
        by_day = Counter()
        by_hour = Counter()
        
        for ticket in self.tickets:
            dt = parse_datetime(ticket.get('created_at'))
            if dt:
                by_month[dt.strftime('%Y-%m')].append(ticket)
                by_day[dt.strftime('%A')] += 1
                by_hour[dt.hour] += 1
        
        category_trends = defaultdict(lambda: defaultdict(int))
        for month, tickets in by_month.items():
            for ticket in tickets:
                text = get_ticket_text(ticket, include_conversations=False).lower()
                for cat_name, cat_info in self.categories.items():
                    if any(kw.lower() in text for kw in cat_info.get('keywords', [])):
                        category_trends[cat_name][month] += 1
                        break
        
        emerging = []
        declining = []
        
        for cat_name, monthly in category_trends.items():
            if len(monthly) >= 3:
                sorted_months = sorted(monthly.items())
                recent = sum(v for k, v in sorted_months[-2:])
                earlier = sum(v for k, v in sorted_months[:-2]) / max(1, len(sorted_months) - 2) * 2
                
                if earlier > 0:
                    change = (recent - earlier) / earlier * 100
                    if change > 50:
                        emerging.append({'category': cat_name, 'change_pct': round(change, 1), 'recent_count': recent})
                    elif change < -50:
                        declining.append({'category': cat_name, 'change_pct': round(change, 1), 'recent_count': recent})
        
        return {
            'monthly_volume': {k: len(v) for k, v in sorted(by_month.items())},
            'weekly_distribution': dict(by_day),
            'hourly_distribution': dict(sorted(by_hour.items())),
            'category_trends': {k: dict(sorted(v.items())) for k, v in category_trends.items()},
            'emerging_issues': sorted(emerging, key=lambda x: x['change_pct'], reverse=True),
            'declining_issues': sorted(declining, key=lambda x: x['change_pct']),
        }
    
    def _detect_anomalies(self) -> List[Dict]:
        """Detect anomalies."""
        anomalies = []
        
        # Duplicate requests
        entity_requests = defaultdict(list)
        for ticket in self.tickets:
            entity = self._extract_entity(ticket)
            if not entity:
                continue
            
            text = get_ticket_text(ticket, include_conversations=False).lower()
            for kw in UserConfig.DUPLICATE_REQUEST_KEYWORDS:
                if kw in text:
                    entity_requests[entity].append({
                        'ticket_id': ticket.get('id'),
                        'date': parse_datetime(ticket.get('created_at')),
                        'keyword': kw,
                    })
                    break
        
        for entity, requests in entity_requests.items():
            if len(requests) >= 2:
                requests.sort(key=lambda x: x['date'] or datetime.min)
                for i in range(1, len(requests)):
                    if requests[i]['date'] and requests[i-1]['date']:
                        days = (requests[i]['date'] - requests[i-1]['date']).days
                        if 0 < days <= UserConfig.DUPLICATE_REQUEST_DAYS:
                            anomalies.append({
                                'type': 'duplicate_request',
                                'severity': 'high',
                                'entity': entity,
                                'days_apart': days,
                                'ticket_ids': [requests[i-1]['ticket_id'], requests[i]['ticket_id']],
                                'description': f"{UserConfig.ENTITY_NAME.title()} '{entity}' had duplicate {requests[i]['keyword']} requests {days} days apart",
                            })
        
        # High-frequency entities
        entity_counts = Counter()
        for ticket in self.tickets:
            entity = self._extract_entity(ticket)
            if entity:
                entity_counts[entity] += 1
        
        if entity_counts:
            avg = sum(entity_counts.values()) / len(entity_counts)
            threshold = avg * UserConfig.HIGH_FREQUENCY_MULTIPLIER
            
            for entity, count in entity_counts.most_common(20):
                if count >= threshold and count >= 10:
                    anomalies.append({
                        'type': 'high_frequency_entity',
                        'severity': 'medium',
                        'entity': entity,
                        'count': count,
                        'average': round(avg, 1),
                        'description': f"{UserConfig.ENTITY_NAME.title()} '{entity}' has {count} tickets ({round(count/avg, 1)}x average)",
                    })
        
        return anomalies
    
    def _analyze_sla(self) -> Dict:
        """Analyze SLA performance."""
        sla_frt = SystemConfig.get_sla_first_response()
        sla_res = SystemConfig.get_sla_resolution()
        
        frt_data = {'times': [], 'breaches': 0}
        resolution_data = {'times': [], 'breaches': 0}
        
        by_priority = {p: {'frt_times': [], 'frt_breaches': 0, 'res_times': [], 'res_breaches': 0} 
                      for p in ['Urgent', 'High', 'Medium', 'Low']}
        
        for ticket in self.tickets:
            priority = ticket.get('priority_name', 'Medium')
            created = parse_datetime(ticket.get('created_at'))
            
            if not created:
                continue
            
            # First response time
            for conv in sorted(ticket.get('conversations', []), key=lambda x: x.get('created_at', '')):
                if not conv.get('incoming', True):
                    responded = parse_datetime(conv.get('created_at'))
                    if responded:
                        frt = (responded - created).total_seconds() / 3600
                        frt_data['times'].append(frt)
                        
                        sla_target = sla_frt.get(priority, 24)
                        if frt > sla_target:
                            frt_data['breaches'] += 1
                        
                        if priority in by_priority:
                            by_priority[priority]['frt_times'].append(frt)
                            if frt > sla_target:
                                by_priority[priority]['frt_breaches'] += 1
                    break
            
            # Resolution
            res = ticket.get('resolution_time_hours')
            if res:
                resolution_data['times'].append(res)
                
                sla_target = sla_res.get(priority, 168)
                if res > sla_target:
                    resolution_data['breaches'] += 1
                
                if priority in by_priority:
                    by_priority[priority]['res_times'].append(res)
                    if res > sla_target:
                        by_priority[priority]['res_breaches'] += 1
        
        def calc_stats(times, breaches):
            if not times:
                return {'avg': None, 'breach_count': 0, 'breach_rate': 0, 'compliance_rate': 100}
            return {
                'avg': round(sum(times) / len(times), 2),
                'breach_count': breaches,
                'breach_rate': round(breaches / len(times) * 100, 1),
                'compliance_rate': round((1 - breaches / len(times)) * 100, 1),
            }
        
        return {
            'first_response': calc_stats(frt_data['times'], frt_data['breaches']),
            'resolution': calc_stats(resolution_data['times'], resolution_data['breaches']),
            'by_priority': {
                p: {
                    'first_response': calc_stats(d['frt_times'], d['frt_breaches']),
                    'resolution': calc_stats(d['res_times'], d['res_breaches']),
                }
                for p, d in by_priority.items()
            },
            'targets': {
                'first_response': sla_frt,
                'resolution': sla_res,
            },
        }
    
    def _generate_findings(self, results: Dict) -> List[Dict]:
        """Generate findings."""
        findings = []
        
        # Zombies
        zombie_data = results.get('zombies', {})
        if zombie_data.get('true_zombie_count', 0) > 0:
            zombie_tickets = zombie_data.get('true_zombies', [])
            finding = Finding(
                finding_type='zombie_analysis',
                title=f"{zombie_data['true_zombie_count']} tickets without agent response",
                description=f"{zombie_data['zombie_rate']}% of tickets have no agent response",
                evidence_ticket_ids=[z['ticket_id'] for z in zombie_tickets[:50]],
                severity='high' if zombie_data['zombie_rate'] > 10 else 'medium',
                recommendation="Review and respond to these tickets immediately",
            )
            finding.calculate_confidence()
            findings.append(finding.to_dict())
        
        # Categories
        categories = results.get('categories', {})
        for cat_name, cat_data in sorted(categories.items(), key=lambda x: x[1].get('total_tickets', 0), reverse=True)[:5]:
            if cat_data.get('total_tickets', 0) >= 10:
                finding = Finding(
                    finding_type='category_insight',
                    title=f"'{cat_name}' is a significant issue category",
                    description=f"{cat_data['total_tickets']} tickets ({cat_data.get('zombie_rate', 0)}% zombie rate)",
                    evidence_ticket_ids=cat_data.get('ticket_ids', []),
                    root_cause=', '.join(cat_data.get('typical_root_causes', [])[:3]),
                    recommendation=f"Focus on addressing root causes",
                )
                finding.calculate_confidence()
                findings.append(finding.to_dict())
        
        # Anomalies
        for anomaly in results.get('anomalies', []):
            finding = Finding(
                finding_type=f"anomaly_{anomaly['type']}",
                title=anomaly.get('description', 'Anomaly detected'),
                description=anomaly.get('description', ''),
                evidence_ticket_ids=anomaly.get('ticket_ids', []),
                severity=anomaly.get('severity', 'medium'),
                recommendation=f"Investigate {anomaly['type'].replace('_', ' ')}",
            )
            finding.calculate_confidence()
            findings.append(finding.to_dict())
        
        return findings
    
    def _fallback_categories(self) -> Dict:
        """Fallback categories."""
        return {
            'general': {
                'keywords': ['issue', 'problem', 'help'],
                'description': 'General support requests',
                'typical_root_causes': ['Various'],
                'ticket_ids': [t.get('id') for t in self.tickets[:50]],
            }
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'UserConfig',
    'AnalysisEngine',
    'OllamaClient',
    'Finding',
    'is_true_zombie_ticket',
    'get_zombie_stats',
    'clean_html',
    'get_ticket_text',
    'parse_datetime',
]


if __name__ == '__main__':
    print("="*60)
    print("FTEX Smart Detection Engine v6.0")
    print("="*60)
    print(f"\n📦 Product: {UserConfig.PRODUCT_NAME}")
    print(f"🏷️  Entity: {UserConfig.ENTITY_NAME}")
    print(f"📚 Known Solutions: {len(UserConfig.KNOWN_SOLUTIONS)}")
    
    print(f"\n⚙️  SLA (from config.py):")
    print(f"   First Response: {SystemConfig.get_sla_first_response()}")
    print(f"   Resolution: {SystemConfig.get_sla_resolution()}")
    
    print("\n🤖 Checking AI...")
    ollama = OllamaClient()
    if ollama.available:
        print(f"   ✅ Connected to {ollama.model}")
    else:
        print("   ⚠️ Ollama not available (fallback mode)")
    
    print("\n" + "="*60)