#!/usr/bin/env python3
"""
FTEX Configuration Module
=========================
Centralized configuration for all FTEX scripts.
Loads from environment variables and .env file.

Usage:
    from config import config
    
    url = config.freshdesk_url
    api_key = config.api_key
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use env vars only


@dataclass
class FreshdeskConfig:
    """Freshdesk API configuration"""
    domain: str = field(default_factory=lambda: os.getenv('FRESHDESK_DOMAIN', 'your-domain'))
    api_key: str = field(default_factory=lambda: os.getenv('FRESHDESK_API_KEY', ''))
    group_id: Optional[int] = field(default_factory=lambda: int(os.getenv('FRESHDESK_GROUP_ID', '0')) or None)
    
    @property
    def base_url(self) -> str:
        return f"https://{self.domain}.freshdesk.com/api/v2"
    
    @property
    def ticket_url(self) -> str:
        return f"https://{self.domain}.freshdesk.com/a/tickets"
    
    def get_ticket_link(self, ticket_id: int) -> str:
        return f"{self.ticket_url}/{ticket_id}"


@dataclass
class OllamaConfig:
    """Ollama LLM configuration"""
    base_url: str = field(default_factory=lambda: os.getenv('OLLAMA_URL', 'http://localhost:11434'))
    model: str = field(default_factory=lambda: os.getenv('OLLAMA_MODEL', 'qwen3:14b'))
    preferred_models: List[str] = field(default_factory=lambda: [
        'qwen3:14b', 'qwen3:8b', 'qwen2.5:14b', 'llama3.1:8b', 'mistral:7b'
    ])
    temperature_analysis: float = 0.3
    temperature_synthesis: float = 0.4
    temperature_creative: float = 0.5
    max_tokens_standard: int = 4000
    max_tokens_synthesis: int = 6000
    max_tokens_slides: int = 8000


@dataclass  
class SLAConfig:
    """SLA thresholds configuration"""
    first_response: Dict[str, int] = field(default_factory=lambda: {
        'Urgent': 1,   # 1 hour
        'High': 4,     # 4 hours
        'Medium': 8,   # 8 hours
        'Low': 24,     # 24 hours
    })
    resolution: Dict[str, int] = field(default_factory=lambda: {
        'Urgent': 4,    # 4 hours
        'High': 24,     # 1 day
        'Medium': 72,   # 3 days
        'Low': 168,     # 7 days
    })
    
    @classmethod
    def from_file(cls, path: str) -> 'SLAConfig':
        """Load SLA config from JSON file"""
        if Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            return cls(
                first_response=data.get('first_response', cls.first_response),
                resolution=data.get('resolution', cls.resolution)
            )
        return cls()


@dataclass
class AnalysisConfig:
    """Analysis parameters configuration"""
    sample_size: int = 100
    batch_size: int = 20
    top_customers: int = 3
    worst_tickets: int = 15
    no_response_sample: int = 20
    zombie_threshold_days: int = 180
    long_resolution_hours: int = 500
    high_priority_breach_hours: int = 48


@dataclass
class OutputConfig:
    """Output paths configuration"""
    base_dir: str = field(default_factory=lambda: os.getenv('FTEX_OUTPUT_DIR', '.'))
    
    @property
    def output_dir(self) -> Path:
        return Path(self.base_dir) / 'output'
    
    @property
    def reports_dir(self) -> Path:
        return Path(self.base_dir) / 'reports'
    
    @property
    def docs_dir(self) -> Path:
        return Path(self.base_dir) / 'docs'
    
    def ensure_dirs(self):
        """Create output directories if they don't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'tickets').mkdir(exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)


@dataclass
class Config:
    """Main configuration container"""
    freshdesk: FreshdeskConfig = field(default_factory=FreshdeskConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    sla: SLAConfig = field(default_factory=SLAConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv('FTEX_LOG_LEVEL', 'INFO'))
    
    def setup_logging(self):
        """Configure logging for all modules"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    @classmethod
    def load(cls, config_file: Optional[str] = None) -> 'Config':
        """Load configuration from file or environment"""
        config = cls()
        
        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                data = json.load(f)
            # Override with file values
            if 'freshdesk' in data:
                config.freshdesk = FreshdeskConfig(**data['freshdesk'])
            if 'ollama' in data:
                config.ollama = OllamaConfig(**data['ollama'])
            if 'sla' in data:
                config.sla = SLAConfig(**data['sla'])
        
        return config


# Global config instance
config = Config()


# Convenience functions
def get_ticket_url(ticket_id: int) -> str:
    """Get Freshdesk ticket URL"""
    return config.freshdesk.get_ticket_link(ticket_id)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger"""
    config.setup_logging()
    return logging.getLogger(name)


# Example .env file content
ENV_TEMPLATE = """# FTEX Configuration
# Copy this to .env and fill in your values

# Freshdesk (Required)
FRESHDESK_DOMAIN=your-company
FRESHDESK_API_KEY=your-api-key-here
FRESHDESK_GROUP_ID=

# Ollama (Optional)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:14b

# Output (Optional)
FTEX_OUTPUT_DIR=.
FTEX_LOG_LEVEL=INFO

# Custom Stop Words (Optional - comma-separated)
FTEX_STOP_WORDS=
"""


if __name__ == '__main__':
    # Print current config for debugging
    print("Current Configuration:")
    print(f"  Freshdesk Domain: {config.freshdesk.domain}")
    print(f"  Freshdesk URL: {config.freshdesk.base_url}")
    print(f"  Ollama URL: {config.ollama.base_url}")
    print(f"  Ollama Model: {config.ollama.model}")
    print(f"  Output Dir: {config.output.base_dir}")
    print()
    print("Example .env file:")
    print(ENV_TEMPLATE)