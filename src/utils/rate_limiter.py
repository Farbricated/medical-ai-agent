import time
import threading
from collections import defaultdict, deque
from typing import Optional, Tuple
from functools import wraps


class RateLimiter:
    def __init__(self):
        self.requests: Dict = defaultdict(deque)
        self.costs: Dict = defaultdict(float)
        self.lock = threading.Lock()
        self.limits = {"per_minute": 60, "per_hour": 500, "per_day": 5000}
        self.model_costs = {"groq": 0.0001, "qdrant": 0.00001, "embedding": 0.00001}

    def _clean_old_requests(self, session_id: str, window_seconds: int):
        now = time.time()
        cutoff = now - window_seconds
        while self.requests[session_id] and self.requests[session_id][0] < cutoff:
            self.requests[session_id].popleft()

    def check_rate_limit(
        self, session_id: str, max_requests: int, window_seconds: int
    ) -> bool:
        with self.lock:
            self._clean_old_requests(session_id, window_seconds)
            if len(self.requests[session_id]) >= max_requests:
                return False
            self.requests[session_id].append(time.time())
            return True

    def is_allowed(self, session_id: str) -> Tuple[bool, Optional[str]]:
        if not self.check_rate_limit(session_id, self.limits["per_minute"], 60):
            return False, f"Rate limit: {self.limits['per_minute']} requests/minute exceeded"
        if not self.check_rate_limit(session_id, self.limits["per_hour"], 3600):
            return False, f"Rate limit: {self.limits['per_hour']} requests/hour exceeded"
        return True, None

    def track_cost(self, session_id: str, service: str, count: int = 1):
        self.costs[session_id] += self.model_costs.get(service, 0) * count

    def get_session_cost(self, session_id: str) -> float:
        return round(self.costs[session_id], 6)

    def get_total_costs(self) -> dict:
        return {
            "total_cost": round(sum(self.costs.values()), 6),
            "sessions": {k: round(v, 6) for k, v in self.costs.items()},
        }

    def reset_session(self, session_id: str):
        with self.lock:
            self.requests.pop(session_id, None)
            self.costs.pop(session_id, None)


_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter