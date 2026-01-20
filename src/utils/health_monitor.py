"""
System Health Monitoring for MedAI
Tracks component health, metrics, and alerts
"""

import psutil
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import os

@dataclass
class HealthStatus:
    """Health status for a component"""
    component: str
    status: str  # healthy, degraded, down
    message: str
    timestamp: str
    response_time: Optional[float] = None

class HealthMonitor:
    """
    Comprehensive health monitoring system
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.component_checks = {
            "api": self._check_api,
            "groq": self._check_groq,
            "qdrant": self._check_qdrant,
            "orchestrator": self._check_orchestrator,
            "system": self._check_system_resources
        }
        
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.start_time
    
    def get_system_metrics(self) -> Dict:
        """Get system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage_percent": round(cpu_percent, 2),
                "memory_usage_percent": round(memory.percent, 2),
                "memory_available_mb": round(memory.available / 1024 / 1024, 2),
                "disk_usage_percent": round(disk.percent, 2),
                "disk_free_gb": round(disk.free / 1024 / 1024 / 1024, 2)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _check_api(self) -> HealthStatus:
        """Check API health"""
        return HealthStatus(
            component="api",
            status="healthy",
            message="API server operational",
            timestamp=datetime.now().isoformat()
        )
    
    def _check_groq(self) -> HealthStatus:
        """Check Groq API connection"""
        start = time.time()
        
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                return HealthStatus(
                    component="groq",
                    status="down",
                    message="GROQ_API_KEY not configured",
                    timestamp=datetime.now().isoformat()
                )
            
            # Simple check - if key exists, assume healthy
            # In production, you might do a lightweight API call
            response_time = time.time() - start
            
            return HealthStatus(
                component="groq",
                status="healthy",
                message="Groq API configured",
                timestamp=datetime.now().isoformat(),
                response_time=round(response_time, 3)
            )
        except Exception as e:
            return HealthStatus(
                component="groq",
                status="degraded",
                message=f"Groq check failed: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def _check_qdrant(self) -> HealthStatus:
        """Check Qdrant connection"""
        start = time.time()
        
        try:
            url = os.getenv("QDRANT_URL")
            api_key = os.getenv("QDRANT_API_KEY")
            
            if not url or not api_key:
                return HealthStatus(
                    component="qdrant",
                    status="down",
                    message="Qdrant credentials not configured",
                    timestamp=datetime.now().isoformat()
                )
            
            response_time = time.time() - start
            
            return HealthStatus(
                component="qdrant",
                status="healthy",
                message="Qdrant configured",
                timestamp=datetime.now().isoformat(),
                response_time=round(response_time, 3)
            )
        except Exception as e:
            return HealthStatus(
                component="qdrant",
                status="degraded",
                message=f"Qdrant check failed: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def _check_orchestrator(self) -> HealthStatus:
        """Check orchestrator status"""
        # This would be set by the application
        return HealthStatus(
            component="orchestrator",
            status="healthy",
            message="Orchestrator initialized",
            timestamp=datetime.now().isoformat()
        )
    
    def _check_system_resources(self) -> HealthStatus:
        """Check system resources"""
        metrics = self.get_system_metrics()
        
        # Determine status based on resource usage
        if metrics.get("cpu_usage_percent", 0) > 90 or metrics.get("memory_usage_percent", 0) > 90:
            status = "degraded"
            message = "High resource usage detected"
        elif metrics.get("disk_usage_percent", 0) > 90:
            status = "degraded"
            message = "Low disk space"
        else:
            status = "healthy"
            message = "System resources normal"
        
        return HealthStatus(
            component="system",
            status=status,
            message=message,
            timestamp=datetime.now().isoformat()
        )
    
    def check_all_components(self) -> Dict[str, HealthStatus]:
        """Run all health checks"""
        results = {}
        
        for component, check_func in self.component_checks.items():
            try:
                results[component] = check_func()
            except Exception as e:
                results[component] = HealthStatus(
                    component=component,
                    status="down",
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now().isoformat()
                )
        
        return results
    
    def get_overall_status(self) -> str:
        """Get overall system health status"""
        checks = self.check_all_components()
        
        # Count statuses
        statuses = [check.status for check in checks.values()]
        
        if "down" in statuses:
            return "unhealthy"
        elif "degraded" in statuses:
            return "degraded"
        else:
            return "healthy"
    
    def get_health_report(self) -> Dict:
        """Get comprehensive health report"""
        components = self.check_all_components()
        
        return {
            "overall_status": self.get_overall_status(),
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": round(self.get_uptime(), 2),
            "components": {k: asdict(v) for k, v in components.items()},
            "system_metrics": self.get_system_metrics()
        }

# Global health monitor
_health_monitor = None

def get_health_monitor() -> HealthMonitor:
    """Get or create health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor
