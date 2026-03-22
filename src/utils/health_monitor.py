import os
import time
import psutil
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class HealthStatus:
    component: str
    status: str  # healthy | degraded | down
    message: str
    timestamp: str
    response_time: Optional[float] = None


class HealthMonitor:
    def __init__(self):
        self.start_time = time.time()

    def get_uptime(self) -> float:
        return time.time() - self.start_time

    def get_system_metrics(self) -> Dict:
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            return {
                "cpu_usage_percent": round(cpu, 2),
                "memory_usage_percent": round(mem.percent, 2),
                "memory_available_mb": round(mem.available / 1024 / 1024, 2),
                "disk_usage_percent": round(disk.percent, 2),
                "disk_free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
            }
        except Exception as e:
            return {"error": str(e)}

    def check_groq(self) -> HealthStatus:
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            return HealthStatus("groq", "down", "GROQ_API_KEY not set", datetime.now().isoformat())
        return HealthStatus("groq", "healthy", "Groq API key configured", datetime.now().isoformat())

    def check_qdrant(self) -> HealthStatus:
        url = os.getenv("QDRANT_URL", "")
        key = os.getenv("QDRANT_API_KEY", "")
        if not url or not key:
            return HealthStatus("qdrant", "down", "Qdrant credentials missing", datetime.now().isoformat())
        try:
            from qdrant_client import QdrantClient
            t0 = time.time()
            client = QdrantClient(url=url, api_key=key)
            client.get_collections()
            rt = round(time.time() - t0, 3)
            return HealthStatus("qdrant", "healthy", "Qdrant reachable", datetime.now().isoformat(), rt)
        except Exception as e:
            return HealthStatus("qdrant", "down", f"Qdrant error: {str(e)[:60]}", datetime.now().isoformat())

    def check_system_resources(self) -> HealthStatus:
        m = self.get_system_metrics()
        if m.get("cpu_usage_percent", 0) > 90 or m.get("memory_usage_percent", 0) > 90:
            return HealthStatus("system", "degraded", "High resource usage", datetime.now().isoformat())
        return HealthStatus("system", "healthy", "Resources normal", datetime.now().isoformat())

    def get_health_report(self) -> Dict:
        components = {
            "groq": self.check_groq(),
            "qdrant": self.check_qdrant(),
            "system": self.check_system_resources(),
        }
        statuses = [c.status for c in components.values()]
        overall = "unhealthy" if "down" in statuses else ("degraded" if "degraded" in statuses else "healthy")
        return {
            "overall_status": overall,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": round(self.get_uptime(), 2),
            "components": {k: asdict(v) for k, v in components.items()},
            "system_metrics": self.get_system_metrics(),
        }


_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    global _monitor
    if _monitor is None:
        _monitor = HealthMonitor()
    return _monitor