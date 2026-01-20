"""
Rate Limiting and Cost Control for MedAI
Prevents API abuse and manages costs
"""

import time
from collections import defaultdict, deque
from typing import Optional
from functools import wraps
import threading

class RateLimiter:
    """
    Token bucket rate limiter with cost tracking
    """
    
    def __init__(self):
        self.requests = defaultdict(deque)
        self.costs = defaultdict(float)
        self.lock = threading.Lock()
        
        # Configuration
        self.limits = {
            "per_minute": 60,
            "per_hour": 500,
            "per_day": 5000
        }
        
        # Cost tracking (in USD)
        self.model_costs = {
            "groq": 0.0001,  # per request (estimate)
            "qdrant": 0.00001,  # per query
            "embedding": 0.00001  # per embedding
        }
    
    def _clean_old_requests(self, session_id: str, window_seconds: int):
        """Remove requests outside time window"""
        now = time.time()
        cutoff = now - window_seconds
        
        while self.requests[session_id] and self.requests[session_id][0] < cutoff:
            self.requests[session_id].popleft()
    
    def check_rate_limit(self, session_id: str, max_requests: int, window_seconds: int) -> bool:
        """Check if request is within rate limit"""
        with self.lock:
            self._clean_old_requests(session_id, window_seconds)
            
            if len(self.requests[session_id]) >= max_requests:
                return False
            
            self.requests[session_id].append(time.time())
            return True
    
    def is_allowed(self, session_id: str) -> tuple[bool, Optional[str]]:
        """
        Check all rate limits
        Returns: (is_allowed, error_message)
        """
        # Check per-minute limit
        if not self.check_rate_limit(session_id, self.limits["per_minute"], 60):
            return False, f"Rate limit exceeded: {self.limits['per_minute']} requests per minute"
        
        # Check per-hour limit
        if not self.check_rate_limit(session_id, self.limits["per_hour"], 3600):
            return False, f"Rate limit exceeded: {self.limits['per_hour']} requests per hour"
        
        # Check per-day limit
        if not self.check_rate_limit(session_id, self.limits["per_day"], 86400):
            return False, f"Rate limit exceeded: {self.limits['per_day']} requests per day"
        
        return True, None
    
    def track_cost(self, session_id: str, service: str, count: int = 1):
        """Track API costs"""
        cost = self.model_costs.get(service, 0) * count
        self.costs[session_id] += cost
    
    def get_session_cost(self, session_id: str) -> float:
        """Get total cost for session"""
        return round(self.costs[session_id], 6)
    
    def get_total_costs(self) -> dict:
        """Get all session costs"""
        return {
            "total_cost": round(sum(self.costs.values()), 6),
            "sessions": {k: round(v, 6) for k, v in self.costs.items()}
        }
    
    def reset_session(self, session_id: str):
        """Reset rate limits for session"""
        with self.lock:
            if session_id in self.requests:
                self.requests[session_id].clear()
            if session_id in self.costs:
                self.costs[session_id] = 0

# Global rate limiter
_rate_limiter = None

def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter

def rate_limit(session_key: str = "session_id"):
    """
    Decorator for rate limiting
    
    Usage:
        @rate_limit()
        def my_endpoint(session_id: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()
            
            # Extract session_id from kwargs
            session_id = kwargs.get(session_key, "default")
            
            # Check rate limit
            allowed, error_msg = limiter.is_allowed(session_id)
            
            if not allowed:
                raise Exception(error_msg)
            
            # Execute function
            result = func(*args, **kwargs)
            
            return result
        
        return wrapper
    return decorator
