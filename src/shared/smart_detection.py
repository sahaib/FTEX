#!/usr/bin/env python3
"""
FTEX Smart Zombie Detection Module
===================================
Shared module for intelligent ticket classification.

Distinguishes TRUE zombies (tickets needing response) from tickets
that appear as "no response" but are actually customer acknowledgments
that inadvertently reopened resolved tickets.

Usage:
    from smart_detection import is_true_zombie_ticket, get_zombie_stats

    # Check single ticket
    is_zombie, reason = is_true_zombie_ticket(ticket)
    
    # Get stats for all tickets
    stats = get_zombie_stats(tickets)

Author: FTEX
License: MIT
"""

import re
import html
from typing import Dict, List, Tuple, Any

# =============================================================================
# ACKNOWLEDGMENT PATTERNS
# =============================================================================

ACKNOWLEDGMENT_PATTERNS = [
    # Direct thanks
    r'^thanks?\.?!?$',
    r'^thank\s*you\.?!?$',
    r'^thanks?\s+(a\s+lot|very\s+much|so\s+much)\.?!?$',
    r'^thank\s+you\s+(very\s+much|so\s+much|a\s+lot)\.?!?$',
    r'^many\s+thanks\.?!?$',
    
    # Thanks with context
    r'^thanks?\s+for\s+(your\s+)?(help|support|assistance|response|reply|quick\s+response)\.?!?',
    r'^thank\s+you\s+for\s+(your\s+)?(help|support|assistance|response|reply|quick\s+response)\.?!?',
    
    # Confirmations
    r'^(got\s+it|ok|okay|noted|understood|perfect|great|awesome|excellent)\.?!?$',
    
    # Status confirmations
    r'^(works?|working)\s*(now|fine|great|perfectly|well)?\.?!?$',
    r'^(issue\s+)?(resolved|fixed|solved|sorted)\.?!?$',
    r'^(this\s+)?(is\s+)?(working|resolved|fixed)\s*(now)?\.?!?$',
    r'^all\s+(good|set|done|sorted)\.?!?$',
    r'^problem\s+solved\.?!?$',
    r'^that\s+(works|worked|did\s+it|fixed\s+it)\.?!?$',
    
    # Closure requests
    r'^(you\s+)?(can|may)\s+close\s+(this|the\s+ticket|it)\.?!?',
    r'^please\s+close\.?!?$',
    r'^(we\s+)?(can|may)\s+close\s+(this|it)\.?!?',
    
    # Appreciation
    r'^much\s+appreciated\.?!?$',
    r'^appreciate\s+(it|your\s+help|the\s+help)\.?!?$',
    r'^cheers\.?!?$',
    r'^regards\.?!?$',
    
    # Simple confirmations
    r'^confirmed\.?!?$',
    r'^received\.?!?$',
]

# Compile patterns for efficiency
COMPILED_ACK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ACKNOWLEDGMENT_PATTERNS]

# Maximum length for acknowledgment messages (longer messages likely contain real content)
MAX_ACK_LENGTH = 200

# Positive words for short message detection
POSITIVE_WORDS = {
    'thanks', 'thank', 'great', 'perfect', 'works', 'working', 
    'resolved', 'fixed', 'ok', 'okay', 'good', 'awesome', 
    'excellent', 'cheers', 'noted', 'confirmed', 'done',
    'sorted', 'understood', 'received', 'appreciated'
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_html(text: str) -> str:
    """
    Remove HTML tags and clean text for analysis.
    
    Args:
        text: Raw text that may contain HTML
        
    Returns:
        Cleaned plain text
    """
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities
    text = html.unescape(text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_acknowledgment_message(text: str) -> bool:
    """
    Check if a message is a customer acknowledgment/thank you.
    
    These are short, positive messages that don't require a response,
    such as "Thanks!", "Got it", "Works now", etc.
    
    Args:
        text: Message text to analyze
        
    Returns:
        True if message is an acknowledgment, False otherwise
    """
    if not text:
        return False
    
    cleaned = clean_html(text).strip()
    
    # Long messages are not simple acknowledgments
    if len(cleaned) > MAX_ACK_LENGTH:
        return False
    
    # Check against compiled patterns
    for pattern in COMPILED_ACK_PATTERNS:
        if pattern.search(cleaned):
            return True
    
    # Check for short positive messages
    words = cleaned.lower().split()
    if len(words) <= 5:
        if any(word.rstrip('.,!?') in POSITIVE_WORDS for word in words):
            return True
    
    return False


# =============================================================================
# MAIN DETECTION FUNCTIONS
# =============================================================================

def is_true_zombie_ticket(ticket: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Determine if a ticket is a TRUE zombie (needs response) vs 
    a ticket reopened by customer acknowledgment.
    
    Logic:
    1. No conversations at all → TRUE zombie
    2. Has agent response → NOT a zombie
    3. No agent response + last message is acknowledgment → NOT a zombie (filtered)
    4. No agent response + real question → TRUE zombie
    
    Args:
        ticket: Ticket dictionary with 'conversations' key
        
    Returns:
        Tuple of (is_zombie: bool, reason: str)
        
    Examples:
        >>> is_true_zombie_ticket({'conversations': []})
        (True, 'No conversations')
        
        >>> ticket_with_thanks = {'conversations': [
        ...     {'incoming': True, 'body_text': 'Thanks!'}
        ... ]}
        >>> is_true_zombie_ticket(ticket_with_thanks)
        (False, 'Customer acknowledgment')
    """
    conversations = ticket.get('conversations', [])
    
    # Case 1: No conversations at all
    if len(conversations) == 0:
        return True, "No conversations"
    
    # Sort conversations by time
    sorted_convos = sorted(conversations, key=lambda x: x.get('created_at', ''))
    
    # Check if there's any agent response
    has_agent_response = any(not c.get('incoming', True) for c in sorted_convos)
    
    if not has_agent_response:
        # No agent has responded - check if last message is just an acknowledgment
        last_convo = sorted_convos[-1] if sorted_convos else None
        
        if last_convo and last_convo.get('incoming', False):
            body = last_convo.get('body_text') or last_convo.get('body', '')
            
            if is_acknowledgment_message(body):
                return False, "Customer acknowledgment"
            else:
                return True, "No agent response"
        else:
            return True, "No agent response"
    
    # Has agent response - find last agent response index
    last_agent_idx = -1
    for i, c in enumerate(sorted_convos):
        if not c.get('incoming', True):
            last_agent_idx = i
    
    # Check messages after last agent response
    if last_agent_idx < len(sorted_convos) - 1:
        messages_after_agent = sorted_convos[last_agent_idx + 1:]
        
        # Check if all messages after agent are acknowledgments
        all_acks = True
        for msg in messages_after_agent:
            if msg.get('incoming', False):
                body = msg.get('body_text') or msg.get('body', '')
                if not is_acknowledgment_message(body):
                    all_acks = False
                    break
        
        if all_acks:
            return False, "Customer acknowledgment after resolution"
        else:
            return False, "Has agent response (pending follow-up)"
    
    return False, "Has agent response"


def get_zombie_stats(tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate zombie ticket statistics with smart detection.
    
    Args:
        tickets: List of ticket dictionaries
        
    Returns:
        Dictionary with:
        - true_zombies: Count of tickets actually needing response
        - false_zombies: Count of filtered acknowledgments
        - zombie_tickets: List of true zombie ticket dicts
        - filtered_tickets: List of filtered acknowledgment ticket dicts
        - total_open: Total open/pending tickets analyzed
    """
    true_zombies = 0
    false_zombies = 0
    zombie_tickets = []
    filtered_tickets = []
    
    for ticket in tickets:
        status = ticket.get('status_name', '')
        
        # Only check open/pending tickets
        if status in ['Resolved', 'Closed']:
            continue
        
        is_zombie, reason = is_true_zombie_ticket(ticket)
        
        if is_zombie:
            true_zombies += 1
            zombie_tickets.append({
                'ticket': ticket,
                'reason': reason
            })
        elif "acknowledgment" in reason.lower():
            false_zombies += 1
            filtered_tickets.append({
                'ticket': ticket,
                'reason': reason
            })
    
    return {
        'true_zombies': true_zombies,
        'false_zombies': false_zombies,
        'zombie_tickets': zombie_tickets,
        'filtered_tickets': filtered_tickets,
        'total_open': true_zombies + false_zombies
    }


def analyze_ticket_response_status(ticket: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get detailed response status analysis for a ticket.
    
    Args:
        ticket: Ticket dictionary
        
    Returns:
        Dictionary with detailed analysis
    """
    conversations = ticket.get('conversations', [])
    sorted_convos = sorted(conversations, key=lambda x: x.get('created_at', ''))
    
    is_zombie, reason = is_true_zombie_ticket(ticket)
    
    agent_responses = [c for c in sorted_convos if not c.get('incoming', True)]
    customer_messages = [c for c in sorted_convos if c.get('incoming', True)]
    
    return {
        'ticket_id': ticket.get('id'),
        'is_true_zombie': is_zombie,
        'classification_reason': reason,
        'needs_response': is_zombie,
        'total_conversations': len(conversations),
        'agent_responses': len(agent_responses),
        'customer_messages': len(customer_messages),
        'status': ticket.get('status_name', 'Unknown'),
        'priority': ticket.get('priority_name', 'Unknown'),
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    # Simple test
    print("Smart Zombie Detection Module")
    print("=" * 40)
    
    # Test cases
    test_messages = [
        "Thanks!",
        "Thank you for your help",
        "Got it",
        "Works now",
        "You can close this",
        "Thanks, but I have another question about the login process",
        "The issue is still happening after I restart",
        "Please help me with vessel configuration",
    ]
    
    print("\nAcknowledgment Detection Tests:")
    for msg in test_messages:
        result = is_acknowledgment_message(msg)
        status = "✅ ACK" if result else "❌ NOT ACK"
        print(f"  {status}: {msg[:50]}")
    
    # Test ticket detection
    print("\nTicket Detection Tests:")
    
    test_tickets = [
        {'conversations': [], 'id': 1},
        {'conversations': [{'incoming': True, 'body_text': 'Thanks!', 'created_at': '2024-01-01'}], 'id': 2},
        {'conversations': [{'incoming': True, 'body_text': 'Help me please', 'created_at': '2024-01-01'}], 'id': 3},
        {'conversations': [
            {'incoming': True, 'body_text': 'Help', 'created_at': '2024-01-01'},
            {'incoming': False, 'body_text': 'Here is the solution', 'created_at': '2024-01-02'},
        ], 'id': 4},
    ]
    
    for t in test_tickets:
        is_zombie, reason = is_true_zombie_ticket(t)
        status = "🧟 ZOMBIE" if is_zombie else "✅ OK"
        print(f"  Ticket #{t['id']}: {status} - {reason}")