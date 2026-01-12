#!/usr/bin/env python3
"""
FTEX Smart Detection & AI Categorization Module v3.0
=====================================================
Generic, AI-powered ticket analysis that learns from YOUR data.

Features:
- 🧠 GenAI-powered category discovery (uses Ollama)
- 📊 Automatic keyword extraction from ticket data
- 🎯 Smart zombie detection (universal)
- 📈 Dynamic thresholds based on your data
- 🔧 Zero hardcoded categories - learns from tickets

Usage:
    from smart_detection import (
        SmartAnalyzer,
        is_true_zombie_ticket,
        get_zombie_stats,
    )
    
    # Initialize with your tickets
    analyzer = SmartAnalyzer(tickets)
    
    # Discover categories using AI
    analyzer.discover_categories()
    
    # Get full analytics
    analytics = analyzer.get_analytics()

Author: FTEX
License: MIT
"""

import re
import html
import json
import os
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set
from pathlib import Path

# Optional: Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# BASE CONFIGURATION (Tunable, No Hardcoded Categories)
# =============================================================================

class Config:
    """
    Base configuration - generic settings only.
    Categories are discovered from data, not hardcoded.
    """
    
    # -------------------------------------------------------------------------
    # ZOMBIE DETECTION (Universal)
    # -------------------------------------------------------------------------
    MAX_ACK_LENGTH = 200
    ACK_WORD_LIMIT = 5
    
    POSITIVE_WORDS = {
        'thanks', 'thank', 'great', 'perfect', 'works', 'working',
        'resolved', 'fixed', 'ok', 'okay', 'good', 'awesome',
        'excellent', 'cheers', 'noted', 'confirmed', 'done',
        'sorted', 'understood', 'received', 'appreciated', 'cool',
        'nice', 'brilliant', 'fantastic', 'wonderful', 'super',
    }
    
    ACKNOWLEDGMENT_PATTERNS = [
        r'^thanks?\.?!?$',
        r'^thank\s*you\.?!?$',
        r'^thanks?\s+(a\s+lot|very\s+much|so\s+much)\.?!?$',
        r'^thank\s+you\s+(very\s+much|so\s+much|a\s+lot)\.?!?$',
        r'^many\s+thanks\.?!?$',
        r'^thanks?\s+for\s+(your\s+)?(help|support|assistance|response|reply|quick\s+response)\.?!?',
        r'^(got\s+it|ok|okay|noted|understood|perfect|great|awesome|excellent)\.?!?$',
        r'^(works?|working)\s*(now|fine|great|perfectly|well)?\.?!?$',
        r'^(issue\s+)?(resolved|fixed|solved|sorted)\.?!?$',
        r'^all\s+(good|set|done|sorted)\.?!?$',
        r'^(you\s+)?(can|may)\s+close\s+(this|the\s+ticket|it)\.?!?',
        r'^please\s+close\.?!?$',
        r'^much\s+appreciated\.?!?$',
        r'^cheers\.?!?$',
        r'^confirmed\.?!?$',
    ]
    
    # -------------------------------------------------------------------------
    # SLA DEFAULTS (Override with your own)
    # -------------------------------------------------------------------------
    SLA_FIRST_RESPONSE = {
        'Urgent': 1, 'High': 4, 'Medium': 8, 'Low': 24,
    }
    SLA_RESOLUTION = {
        'Urgent': 4, 'High': 24, 'Medium': 72, 'Low': 168,
    }
    
    # -------------------------------------------------------------------------
    # ANALYSIS SETTINGS
    # -------------------------------------------------------------------------
    LONG_RESOLUTION_HOURS = 500
    ZOMBIE_THRESHOLD_DAYS = 180
    MIN_CATEGORY_SIZE = 5          # Minimum tickets to form a category
    TOP_KEYWORDS_PER_CATEGORY = 10
    SAMPLE_SIZE_FOR_AI = 100       # Tickets to sample for AI analysis
    
    # -------------------------------------------------------------------------
    # OLLAMA SETTINGS
    # -------------------------------------------------------------------------
    OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:14b')
    OLLAMA_TIMEOUT = 180
    
    # -------------------------------------------------------------------------
    # STOP WORDS (Generic - for keyword extraction)
    # -------------------------------------------------------------------------
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'what', 'which', 'who', 'whom', 'when', 'where', 'why',
        'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
        'please', 'hello', 'hi', 'dear', 'regards', 'thanks', 'thank',
        're', 'fw', 'fwd', 'subject', 'ticket', 'issue', 'problem', 'help',
        'request', 'query', 'question', 'support', 'team', 'customer',
    }


# Compile patterns
_COMPILED_ACK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in Config.ACKNOWLEDGMENT_PATTERNS]


# =============================================================================
# UTILITY FUNCTIONS
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
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except:
        try:
            return datetime.fromisoformat(dt_str[:19])
        except:
            return None


def extract_keywords(text: str, stop_words: Set[str] = None) -> List[str]:
    """Extract meaningful keywords from text."""
    if not text:
        return []
    
    stop_words = stop_words or Config.STOP_WORDS
    
    # Clean and tokenize
    cleaned = clean_html(text).lower()
    words = re.findall(r'\b[a-z]{3,}\b', cleaned)
    
    # Filter stop words
    keywords = [w for w in words if w not in stop_words]
    
    return keywords


# =============================================================================
# ZOMBIE DETECTION (Universal - Works for Any Helpdesk)
# =============================================================================

def is_acknowledgment_message(text: str) -> bool:
    """Check if a message is a customer acknowledgment/thank you."""
    if not text:
        return False
    
    cleaned = clean_html(text).strip()
    
    if len(cleaned) > Config.MAX_ACK_LENGTH:
        return False
    
    for pattern in _COMPILED_ACK_PATTERNS:
        if pattern.search(cleaned):
            return True
    
    words = cleaned.lower().split()
    if len(words) <= Config.ACK_WORD_LIMIT:
        if any(word.rstrip('.,!?') in Config.POSITIVE_WORDS for word in words):
            return True
    
    return False


def is_true_zombie_ticket(ticket: dict) -> Tuple[bool, str]:
    """
    Determine if a ticket is a TRUE zombie (needs response).
    Universal logic - works for any helpdesk system.
    
    Returns: (is_zombie: bool, reason: str)
    """
    conversations = ticket.get('conversations', [])
    
    if len(conversations) == 0:
        return True, "No conversations"
    
    sorted_convos = sorted(conversations, key=lambda x: x.get('created_at', ''))
    has_agent_response = any(not c.get('incoming', True) for c in sorted_convos)
    
    if not has_agent_response:
        last_convo = sorted_convos[-1] if sorted_convos else None
        if last_convo and last_convo.get('incoming', False):
            body = last_convo.get('body_text') or last_convo.get('body', '')
            if is_acknowledgment_message(body):
                return False, "Customer acknowledgment"
            else:
                return True, "No agent response"
        return True, "No agent response"
    
    last_agent_idx = -1
    for i, c in enumerate(sorted_convos):
        if not c.get('incoming', True):
            last_agent_idx = i
    
    if last_agent_idx < len(sorted_convos) - 1:
        messages_after = sorted_convos[last_agent_idx + 1:]
        all_acks = all(
            is_acknowledgment_message(m.get('body_text') or m.get('body', ''))
            for m in messages_after if m.get('incoming', False)
        )
        if all_acks:
            return False, "Customer acknowledgment after resolution"
        return False, "Has agent response (pending follow-up)"
    
    return False, "Has agent response"


def get_zombie_stats(tickets: List[dict]) -> Dict[str, Any]:
    """Calculate zombie ticket statistics."""
    true_zombies = []
    false_zombies = []
    
    for ticket in tickets:
        is_zombie, reason = is_true_zombie_ticket(ticket)
        entry = {'ticket': ticket, 'reason': reason}
        
        if is_zombie:
            true_zombies.append(entry)
        elif "acknowledgment" in reason.lower():
            false_zombies.append(entry)
    
    return {
        'true_zombies': len(true_zombies),
        'false_zombies': len(false_zombies),
        'zombie_tickets': true_zombies,
        'filtered_tickets': false_zombies,
        'zombie_rate': round(len(true_zombies) / len(tickets) * 100, 2) if tickets else 0,
    }


# =============================================================================
# OLLAMA CLIENT (For GenAI Features)
# =============================================================================

class OllamaClient:
    """Client for local Ollama LLM."""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or Config.OLLAMA_URL
        self.model = model or Config.OLLAMA_MODEL
        self.available = False
        self._checked = False
    
    def check_availability(self) -> bool:
        """Check if Ollama is running."""
        if self._checked:
            return self.available
        
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                # Find matching model
                for m in models:
                    if self.model in m or m in self.model:
                        self.model = m
                        self.available = True
                        break
                # Fallback to any model
                if not self.available and models:
                    self.model = models[0]
                    self.available = True
        except:
            pass
        
        self._checked = True
        return self.available
    
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2000) -> Optional[str]:
        """Generate completion from Ollama."""
        if not self.check_availability():
            return None
        
        try:
            import requests
            response = requests.post(
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
                timeout=Config.OLLAMA_TIMEOUT
            )
            if response.status_code == 200:
                return response.json().get('response', '').strip()
        except:
            pass
        return None


# =============================================================================
# AI-POWERED CATEGORY DISCOVERY
# =============================================================================

class CategoryDiscovery:
    """Discovers categories from ticket data using AI + statistical analysis."""
    
    def __init__(self, tickets: List[dict], ollama: OllamaClient = None):
        self.tickets = tickets
        self.ollama = ollama or OllamaClient()
        self.categories = {}
        self.keyword_index = defaultdict(list)  # keyword -> ticket_ids
    
    def extract_all_keywords(self) -> Counter:
        """Extract and count all keywords from tickets."""
        all_keywords = []
        
        for ticket in self.tickets:
            text = (ticket.get('subject', '') + ' ' + 
                   clean_html(ticket.get('description', '')))
            keywords = extract_keywords(text)
            all_keywords.extend(keywords)
            
            # Build keyword index
            ticket_id = ticket.get('id')
            for kw in set(keywords):
                self.keyword_index[kw].append(ticket_id)
        
        return Counter(all_keywords)
    
    def find_keyword_clusters(self, min_count: int = None) -> Dict[str, List[str]]:
        """Find clusters of related keywords based on co-occurrence."""
        min_count = min_count or Config.MIN_CATEGORY_SIZE
        keyword_counts = self.extract_all_keywords()
        
        # Get significant keywords
        significant = {k: v for k, v in keyword_counts.items() if v >= min_count}
        
        # Find co-occurring keywords
        clusters = {}
        used_keywords = set()
        
        for keyword, count in sorted(significant.items(), key=lambda x: x[1], reverse=True):
            if keyword in used_keywords:
                continue
            
            # Find keywords that often appear with this one
            ticket_ids = set(self.keyword_index[keyword])
            related = []
            
            for other_kw, other_count in significant.items():
                if other_kw == keyword or other_kw in used_keywords:
                    continue
                
                other_ids = set(self.keyword_index[other_kw])
                overlap = len(ticket_ids & other_ids)
                
                # If >30% overlap, they're related
                if overlap >= min(len(ticket_ids), len(other_ids)) * 0.3:
                    related.append(other_kw)
            
            if len(ticket_ids) >= min_count:
                clusters[keyword] = related[:10]
                used_keywords.add(keyword)
                used_keywords.update(related[:5])
        
        return clusters
    
    def discover_with_ai(self, sample_size: int = None) -> Dict[str, Dict]:
        """Use AI to discover and label categories."""
        sample_size = sample_size or Config.SAMPLE_SIZE_FOR_AI
        
        if not self.ollama.check_availability():
            print("⚠️ Ollama not available, using statistical discovery only")
            return self.discover_statistical()
        
        # Sample diverse tickets
        sampled = self._sample_diverse_tickets(sample_size)
        
        # Prepare ticket summaries for AI
        summaries = []
        for t in sampled[:50]:
            subject = t.get('subject', '')[:100]
            desc = clean_html(t.get('description', ''))[:200]
            summaries.append(f"- {subject}: {desc}")
        
        prompt = f"""Analyze these support tickets and identify the main CATEGORIES of issues.

TICKETS:
{chr(10).join(summaries)}

Based on these tickets, identify 8-15 distinct categories. For each category:
1. Give it a short name (2-4 words, lowercase, underscores)
2. List 5-10 keywords that identify tickets in this category
3. Briefly describe what issues fall into this category

Format your response EXACTLY like this JSON:
{{
  "category_name": {{
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "description": "Brief description of this category"
  }},
  "another_category": {{
    "keywords": ["keyword1", "keyword2"],
    "description": "Description"
  }}
}}

Return ONLY valid JSON, no other text."""

        response = self.ollama.generate(prompt, temperature=0.3, max_tokens=3000)
        
        if response:
            try:
                # Extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    categories = json.loads(json_match.group())
                    self.categories = categories
                    return categories
            except json.JSONDecodeError:
                pass
        
        # Fallback to statistical
        return self.discover_statistical()
    
    def discover_statistical(self) -> Dict[str, Dict]:
        """Discover categories using statistical analysis only (no AI)."""
        clusters = self.find_keyword_clusters()
        
        categories = {}
        for main_keyword, related in clusters.items():
            category_name = main_keyword.replace(' ', '_')
            keywords = [main_keyword] + related
            
            categories[category_name] = {
                'keywords': keywords,
                'description': f"Tickets related to {main_keyword}",
                'ticket_count': len(self.keyword_index.get(main_keyword, [])),
            }
        
        self.categories = categories
        return categories
    
    def _sample_diverse_tickets(self, n: int) -> List[dict]:
        """Sample diverse tickets for AI analysis."""
        sampled = []
        seen_ids = set()
        
        # By status
        by_status = defaultdict(list)
        for t in self.tickets:
            by_status[t.get('status_name', 'Unknown')].append(t)
        
        for status, tix in by_status.items():
            for t in tix[:n // 6]:
                if t.get('id') not in seen_ids:
                    seen_ids.add(t.get('id'))
                    sampled.append(t)
        
        # By priority
        by_priority = defaultdict(list)
        for t in self.tickets:
            by_priority[t.get('priority_name', 'Unknown')].append(t)
        
        for priority in ['Urgent', 'High']:
            for t in by_priority.get(priority, [])[:10]:
                if t.get('id') not in seen_ids:
                    seen_ids.add(t.get('id'))
                    sampled.append(t)
        
        # Random fill
        import random
        remaining = [t for t in self.tickets if t.get('id') not in seen_ids]
        random.shuffle(remaining)
        sampled.extend(remaining[:n - len(sampled)])
        
        return sampled[:n]
    
    def label_ticket_with_ai(self, ticket: dict) -> List[str]:
        """Use AI to categorize a single ticket."""
        if not self.ollama.check_availability() or not self.categories:
            return self.label_ticket_keywords(ticket)
        
        subject = ticket.get('subject', '')
        desc = clean_html(ticket.get('description', ''))[:500]
        
        categories_list = list(self.categories.keys())
        
        prompt = f"""Categorize this support ticket into one or more categories.

TICKET:
Subject: {subject}
Description: {desc}

AVAILABLE CATEGORIES:
{', '.join(categories_list)}

Return ONLY the category names that apply, comma-separated. If none fit, return "other".
Example: login_issues, password_reset"""

        response = self.ollama.generate(prompt, temperature=0.1, max_tokens=100)
        
        if response:
            found = []
            for cat in categories_list:
                if cat.lower() in response.lower():
                    found.append(cat)
            if found:
                return found
        
        return self.label_ticket_keywords(ticket)
    
    def label_ticket_keywords(self, ticket: dict) -> List[str]:
        """Categorize ticket using keyword matching."""
        if not self.categories:
            return ['uncategorized']
        
        text = (ticket.get('subject', '') + ' ' + 
               clean_html(ticket.get('description', ''))).lower()
        
        matched = []
        for category, data in self.categories.items():
            keywords = data.get('keywords', [])
            for kw in keywords:
                if kw.lower() in text:
                    matched.append(category)
                    break
        
        return matched if matched else ['uncategorized']


# =============================================================================
# SMART ANALYZER (Main Interface)
# =============================================================================

class SmartAnalyzer:
    """
    Main interface for smart ticket analysis.
    Combines zombie detection, AI categorization, and analytics.
    """
    
    def __init__(self, tickets: List[dict], use_ai: bool = True):
        """
        Initialize analyzer with tickets.
        
        Args:
            tickets: List of ticket dictionaries
            use_ai: Whether to use Ollama for AI-powered features
        """
        self.tickets = tickets
        self.use_ai = use_ai
        self.ollama = OllamaClient() if use_ai else None
        self.discovery = CategoryDiscovery(tickets, self.ollama)
        self.categories = {}
        self._analytics_cache = None
    
    def discover_categories(self, force_statistical: bool = False) -> Dict[str, Dict]:
        """
        Discover categories from ticket data.
        
        Args:
            force_statistical: If True, skip AI and use statistical only
        
        Returns:
            Dictionary of discovered categories
        """
        if force_statistical or not self.use_ai:
            self.categories = self.discovery.discover_statistical()
        else:
            self.categories = self.discovery.discover_with_ai()
        
        self._analytics_cache = None
        return self.categories
    
    def categorize_ticket(self, ticket: dict) -> List[str]:
        """Categorize a single ticket."""
        if not self.categories:
            self.discover_categories()
        
        if self.use_ai and self.ollama and self.ollama.available:
            return self.discovery.label_ticket_with_ai(ticket)
        return self.discovery.label_ticket_keywords(ticket)
    
    def get_analytics(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive analytics for all tickets.
        
        Returns:
            {
                'overview': {...},
                'zombies': {...},
                'categories': {...},
                'customers': {...},
                'sla': {...},
                'priorities': {...},
            }
        """
        if self._analytics_cache and not force_refresh:
            return self._analytics_cache
        
        if not self.categories:
            self.discover_categories()
        
        analytics = {
            'overview': self._get_overview(),
            'zombies': get_zombie_stats(self.tickets),
            'categories': self._get_category_analytics(),
            'customers': self._get_customer_analytics(),
            'sla': self._get_sla_analytics(),
            'priorities': self._get_priority_analytics(),
            'resolution': self._get_resolution_analytics(),
        }
        
        self._analytics_cache = analytics
        return analytics
    
    def _get_overview(self) -> Dict[str, Any]:
        """Get overview statistics."""
        statuses = Counter(t.get('status_name', 'Unknown') for t in self.tickets)
        priorities = Counter(t.get('priority_name', 'Unknown') for t in self.tickets)
        
        return {
            'total_tickets': len(self.tickets),
            'by_status': dict(statuses),
            'by_priority': dict(priorities),
            'unique_companies': len(set(
                t.get('company', {}).get('name', 'Unknown') if t.get('company') else 'Unknown'
                for t in self.tickets
            )),
            'date_range': self._get_date_range(),
        }
    
    def _get_date_range(self) -> Dict[str, str]:
        """Get date range of tickets."""
        dates = []
        for t in self.tickets:
            dt = parse_datetime(t.get('created_at'))
            if dt:
                dates.append(dt)
        
        if dates:
            return {
                'earliest': min(dates).strftime('%Y-%m-%d'),
                'latest': max(dates).strftime('%Y-%m-%d'),
            }
        return {'earliest': 'N/A', 'latest': 'N/A'}
    
    def _get_category_analytics(self) -> Dict[str, Dict]:
        """Get analytics per category."""
        cat_data = defaultdict(lambda: {
            'tickets': [],
            'by_status': Counter(),
            'by_priority': Counter(),
            'resolution_times': [],
            'zombies': 0,
        })
        
        for ticket in self.tickets:
            categories = self.categorize_ticket(ticket)
            is_zombie = is_true_zombie_ticket(ticket)[0]
            
            for cat in categories:
                data = cat_data[cat]
                data['tickets'].append(ticket)
                data['by_status'][ticket.get('status_name', 'Unknown')] += 1
                data['by_priority'][ticket.get('priority_name', 'Unknown')] += 1
                
                if ticket.get('resolution_time_hours'):
                    data['resolution_times'].append(ticket['resolution_time_hours'])
                
                if is_zombie:
                    data['zombies'] += 1
        
        # Calculate summaries
        result = {}
        for cat, data in sorted(cat_data.items(), key=lambda x: len(x[1]['tickets']), reverse=True):
            res_times = data['resolution_times']
            result[cat] = {
                'count': len(data['tickets']),
                'by_status': dict(data['by_status']),
                'by_priority': dict(data['by_priority']),
                'zombies': data['zombies'],
                'open_count': data['by_status'].get('Open', 0) + data['by_status'].get('Pending', 0),
                'avg_resolution_hours': round(sum(res_times) / len(res_times), 1) if res_times else 0,
                'avg_resolution_days': round(sum(res_times) / len(res_times) / 24, 1) if res_times else 0,
                'sample_ids': [t.get('id') for t in data['tickets'][:10]],
                'keywords': self.categories.get(cat, {}).get('keywords', []),
                'description': self.categories.get(cat, {}).get('description', ''),
            }
        
        return result
    
    def _get_customer_analytics(self) -> Dict[str, Dict]:
        """Get analytics per customer."""
        company_data = defaultdict(list)
        
        for ticket in self.tickets:
            company = ticket.get('company', {})
            name = company.get('name', 'Unknown') if company else 'Unknown'
            company_data[name].append(ticket)
        
        result = {}
        for company, tix in sorted(company_data.items(), key=lambda x: len(x[1]), reverse=True)[:50]:
            zombies = sum(1 for t in tix if is_true_zombie_ticket(t)[0])
            open_count = sum(1 for t in tix if t.get('status_name') in ['Open', 'Pending'])
            res_times = [t.get('resolution_time_hours') for t in tix if t.get('resolution_time_hours')]
            
            # Health score
            total = len(tix)
            open_pct = (open_count / total) * 100 if total else 0
            avg_res_days = (sum(res_times) / len(res_times) / 24) if res_times else 0
            
            if zombies >= 5 or open_pct >= 50 or avg_res_days >= 14:
                health = 'critical'
            elif zombies >= 2 or open_pct >= 30 or avg_res_days >= 7:
                health = 'at_risk'
            else:
                health = 'healthy'
            
            result[company] = {
                'total_tickets': total,
                'open_count': open_count,
                'zombies': zombies,
                'health': health,
                'avg_resolution_days': round(avg_res_days, 1),
                'sample_ids': [t.get('id') for t in tix[:5]],
            }
        
        return result
    
    def _get_sla_analytics(self) -> Dict[str, Any]:
        """Get SLA compliance analytics."""
        frt_times = []
        frt_breaches = 0
        res_times = []
        res_breaches = 0
        
        for ticket in self.tickets:
            priority = ticket.get('priority_name', 'Medium')
            
            # FRT
            frt = self._calculate_frt(ticket)
            if frt is not None:
                frt_times.append(frt)
                if frt > Config.SLA_FIRST_RESPONSE.get(priority, 24):
                    frt_breaches += 1
            
            # Resolution
            res = ticket.get('resolution_time_hours')
            if res:
                res_times.append(res)
                if res > Config.SLA_RESOLUTION.get(priority, 168):
                    res_breaches += 1
        
        return {
            'first_response': {
                'measured': len(frt_times),
                'breaches': frt_breaches,
                'compliance_pct': round((1 - frt_breaches / len(frt_times)) * 100, 1) if frt_times else 0,
                'avg_hours': round(sum(frt_times) / len(frt_times), 1) if frt_times else 0,
            },
            'resolution': {
                'measured': len(res_times),
                'breaches': res_breaches,
                'compliance_pct': round((1 - res_breaches / len(res_times)) * 100, 1) if res_times else 0,
                'avg_hours': round(sum(res_times) / len(res_times), 1) if res_times else 0,
                'avg_days': round(sum(res_times) / len(res_times) / 24, 1) if res_times else 0,
            },
        }
    
    def _calculate_frt(self, ticket: dict) -> Optional[float]:
        """Calculate first response time in hours."""
        created = parse_datetime(ticket.get('created_at'))
        if not created:
            return None
        
        for conv in sorted(ticket.get('conversations', []), key=lambda x: x.get('created_at', '')):
            if not conv.get('incoming', True):
                response_time = parse_datetime(conv.get('created_at'))
                if response_time:
                    return (response_time - created).total_seconds() / 3600
        return None
    
    def _get_priority_analytics(self) -> Dict[str, Dict]:
        """Get analytics per priority."""
        result = {}
        
        for priority in ['Urgent', 'High', 'Medium', 'Low']:
            tix = [t for t in self.tickets if t.get('priority_name') == priority]
            if not tix:
                continue
            
            zombies = sum(1 for t in tix if is_true_zombie_ticket(t)[0])
            res_times = [t.get('resolution_time_hours') for t in tix if t.get('resolution_time_hours')]
            
            result[priority] = {
                'count': len(tix),
                'zombies': zombies,
                'avg_resolution_hours': round(sum(res_times) / len(res_times), 1) if res_times else 0,
                'sla_target_frt': Config.SLA_FIRST_RESPONSE.get(priority, 24),
                'sla_target_resolution': Config.SLA_RESOLUTION.get(priority, 168),
            }
        
        return result
    
    def _get_resolution_analytics(self) -> Dict[str, Any]:
        """Get resolution time distribution."""
        buckets = {
            'same_day': 0,
            '1_3_days': 0,
            '3_7_days': 0,
            '1_2_weeks': 0,
            '2_4_weeks': 0,
            'over_month': 0,
        }
        
        for ticket in self.tickets:
            hours = ticket.get('resolution_time_hours')
            if not hours:
                continue
            
            if hours <= 24:
                buckets['same_day'] += 1
            elif hours <= 72:
                buckets['1_3_days'] += 1
            elif hours <= 168:
                buckets['3_7_days'] += 1
            elif hours <= 336:
                buckets['1_2_weeks'] += 1
            elif hours <= 672:
                buckets['2_4_weeks'] += 1
            else:
                buckets['over_month'] += 1
        
        return buckets
    
    def get_category_summary_df(self) -> List[Dict]:
        """Get category summary as list of dicts (for DataFrame/Excel)."""
        analytics = self.get_analytics()
        
        summary = []
        for cat, data in analytics['categories'].items():
            summary.append({
                'Category': cat,
                'Description': data.get('description', ''),
                'Total': data['count'],
                'Open': data['open_count'],
                'Zombies': data['zombies'],
                'Avg Resolution (days)': data['avg_resolution_days'],
                'Urgent': data['by_priority'].get('Urgent', 0),
                'High': data['by_priority'].get('High', 0),
                'Medium': data['by_priority'].get('Medium', 0),
                'Low': data['by_priority'].get('Low', 0),
                'Keywords': ', '.join(data['keywords'][:5]),
                'Sample IDs': ', '.join(map(str, data['sample_ids'][:5])),
            })
        
        return summary
    
    def get_customer_summary_df(self) -> List[Dict]:
        """Get customer summary as list of dicts (for DataFrame/Excel)."""
        analytics = self.get_analytics()
        
        summary = []
        for company, data in analytics['customers'].items():
            summary.append({
                'Company': company,
                'Total Tickets': data['total_tickets'],
                'Open': data['open_count'],
                'Zombies': data['zombies'],
                'Health': data['health'],
                'Avg Resolution (days)': data['avg_resolution_days'],
            })
        
        return summary
    
    def save_config(self, path: str = 'ftex_config.json'):
        """Save discovered categories and config for reuse."""
        config = {
            'discovered_categories': self.categories,
            'sla_thresholds': {
                'first_response': Config.SLA_FIRST_RESPONSE,
                'resolution': Config.SLA_RESOLUTION,
            },
            'generated_at': datetime.now().isoformat(),
            'ticket_count': len(self.tickets),
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return path
    
    def load_config(self, path: str = 'ftex_config.json'):
        """Load previously saved configuration."""
        if Path(path).exists():
            with open(path, 'r') as f:
                config = json.load(f)
            
            self.categories = config.get('discovered_categories', {})
            self._analytics_cache = None
            return True
        return False


# =============================================================================
# CONVENIENCE FUNCTIONS (Backward Compatibility)
# =============================================================================

def categorize_ticket(ticket: dict, categories: Dict[str, Dict] = None) -> List[str]:
    """
    Categorize a ticket using provided or discovered categories.
    For backward compatibility with scripts using old interface.
    """
    if not categories:
        return ['uncategorized']
    
    text = (ticket.get('subject', '') + ' ' + 
           clean_html(ticket.get('description', ''))).lower()
    
    matched = []
    for category, data in categories.items():
        keywords = data.get('keywords', [])
        for kw in keywords:
            if kw.lower() in text:
                matched.append(category)
                break
    
    return matched if matched else ['uncategorized']


def get_category_analytics(tickets: List[dict], categories: Dict[str, Dict] = None) -> Dict[str, Dict]:
    """
    Get category analytics. Creates analyzer if categories not provided.
    For backward compatibility.
    """
    analyzer = SmartAnalyzer(tickets, use_ai=False)
    if categories:
        analyzer.categories = categories
    else:
        analyzer.discover_categories(force_statistical=True)
    
    return analyzer.get_analytics()['categories']


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == '__main__':
    print("FTEX Smart Detection Module v3.0")
    print("=" * 50)
    print("\n🧠 AI-Powered Category Discovery")
    print("📊 Learns from YOUR ticket data")
    print("🔧 Zero hardcoded categories")
    
    # Test zombie detection
    print("\n📝 Zombie Detection Tests:")
    test_messages = [
        "Thanks!",
        "Got it, works now",
        "The issue is still happening",
    ]
    for msg in test_messages:
        result = is_acknowledgment_message(msg)
        status = "✅ ACK" if result else "❌ NOT ACK"
        print(f"  {status}: {msg}")
    
    # Check Ollama
    print("\n🤖 Checking Ollama availability...")
    ollama = OllamaClient()
    if ollama.check_availability():
        print(f"  ✅ Connected to {ollama.model}")
    else:
        print("  ⚠️ Ollama not available (statistical mode only)")
    
    print("\n✅ Module loaded successfully!")
    print("\nUsage:")
    print("  from smart_detection import SmartAnalyzer")
    print("  analyzer = SmartAnalyzer(tickets)")
    print("  analyzer.discover_categories()  # AI-powered!")
    print("  analytics = analyzer.get_analytics()")