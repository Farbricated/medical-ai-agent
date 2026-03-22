import time
import json
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict


@dataclass
class EvaluationResult:
    query_type: str
    accuracy_score: float
    response_time: float
    retrieval_precision: float
    confidence_score: float
    source_quality: float
    timestamp: str


class MedicalAgentEvaluator:
    def __init__(self):
        self.evaluation_history: List[EvaluationResult] = []
        self.test_cases = self._load_test_cases()

    def _load_test_cases(self) -> List[Dict]:
        return [
            {
                "type": "diagnosis",
                "input": "Patient: 55M, chest pain radiating to left arm, sweating, elevated troponin",
                "expected_condition": "Acute Myocardial Infarction",
                "expected_keywords": ["myocardial", "infarction", "troponin", "chest pain", "urgent"],
            },
            {
                "type": "diagnosis",
                "input": "Patient: 45F, polyuria, polydipsia, HbA1c 8.5%, fasting glucose 180",
                "expected_condition": "Type 2 Diabetes Mellitus",
                "expected_keywords": ["diabetes", "glucose", "HbA1c", "metformin"],
            },
            {
                "type": "qa",
                "input": "What are the first-line treatments for hypertension?",
                "expected_keywords": ["ACE inhibitor", "ARB", "calcium channel", "diuretic"],
            },
            {
                "type": "research",
                "input": "Latest studies on statins for cardiovascular disease",
                "expected_keywords": ["LDL", "cholesterol", "cardiovascular", "mortality"],
            },
        ]

    def evaluate_response(
        self,
        query: str,
        response: str,
        query_type: str,
        response_time: float,
        retrieved_docs: List[str] = None,
    ) -> EvaluationResult:
        test_case = next(
            (tc for tc in self.test_cases if query.lower()[:30] in tc["input"].lower()), None
        )

        if test_case:
            keywords = test_case.get("expected_keywords", [])
            response_lower = response.lower()
            hits = sum(1 for kw in keywords if kw.lower() in response_lower)
            accuracy_score = (hits / len(keywords)) * 100 if keywords else 85.0
            if "expected_condition" in test_case:
                if test_case["expected_condition"].lower() in response_lower:
                    accuracy_score = min(accuracy_score + 10, 100)
        else:
            accuracy_score = self._heuristic_accuracy(response, query_type)

        retrieval_precision = 90.0
        if retrieved_docs and test_case:
            docs_text = " ".join(retrieved_docs).lower()
            keywords = test_case.get("expected_keywords", [])
            if keywords:
                hits = sum(1 for kw in keywords if kw.lower() in docs_text)
                retrieval_precision = (hits / len(keywords)) * 100

        result = EvaluationResult(
            query_type=query_type,
            accuracy_score=round(accuracy_score, 1),
            response_time=round(response_time, 2),
            retrieval_precision=round(retrieval_precision, 1),
            confidence_score=round(self._calculate_confidence(response, query_type), 1),
            source_quality=round(self._assess_source_quality(response), 1),
            timestamp=datetime.now().isoformat(),
        )
        self.evaluation_history.append(result)
        return result

    def _heuristic_accuracy(self, response: str, query_type: str) -> float:
        score = 70.0
        if len(response) > 100:
            score += 10
        medical_terms = ["patient", "diagnosis", "treatment", "symptoms", "condition",
                         "evidence", "clinical", "research", "study"]
        score += min(sum(1 for t in medical_terms if t in response.lower()) * 2, 20)
        return min(score, 95.0)

    def _calculate_confidence(self, response: str, query_type: str) -> float:
        score = 70.0
        if any(w in response.lower() for w in ["recommend", "suggest", "indicate"]):
            score += 10
        if any(w in response.lower() for w in ["evidence", "study", "research"]):
            score += 10
        if 200 <= len(response) <= 1500:
            score += 10
        return min(score, 98.0)

    def _assess_source_quality(self, response: str) -> float:
        score = 75.0
        if "source:" in response.lower() or "according to" in response.lower():
            score += 15
        if any(w in response.lower() for w in ["study", "trial", "research", "meta-analysis"]):
            score += 10
        return min(score, 100.0)

    def get_aggregate_metrics(self) -> Dict[str, float]:
        if not self.evaluation_history:
            return {"avg_accuracy": 0.0, "avg_response_time": 0.0,
                    "avg_retrieval_precision": 0.0, "total_queries": 0, "success_rate": 0.0}
        total = len(self.evaluation_history)
        return {
            "avg_accuracy": round(sum(r.accuracy_score for r in self.evaluation_history) / total, 1),
            "avg_response_time": round(sum(r.response_time for r in self.evaluation_history) / total, 2),
            "avg_retrieval_precision": round(sum(r.retrieval_precision for r in self.evaluation_history) / total, 1),
            "total_queries": total,
            "success_rate": round(sum(1 for r in self.evaluation_history if r.accuracy_score >= 80) / total * 100, 1),
        }

    def get_agent_comparison(self) -> Dict[str, Dict]:
        agents = ["diagnosis", "qa", "research"]
        comparison = {}
        for agent in agents:
            results = [r for r in self.evaluation_history if r.query_type == agent]
            if results:
                comparison[agent] = {
                    "accuracy": round(sum(r.accuracy_score for r in results) / len(results), 1),
                    "speed": round(sum(r.response_time for r in results) / len(results), 2),
                    "count": len(results),
                }
            else:
                comparison[agent] = {"accuracy": 0.0, "speed": 0.0, "count": 0}
        return comparison

    def run_automated_tests(self, orchestrator) -> Dict[str, Any]:
        """Run predefined test cases. FIX: uses orchestrator.process() not route_query()."""
        print("🧪 Running Automated Evaluation Tests...")
        results = []

        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[Test {i}/{len(self.test_cases)}] {test_case['type'].upper()}")
            print(f"Query: {test_case['input'][:60]}...")
            start_time = time.time()
            try:
                # FIXED: was orchestrator.route_query() — wrong method
                result = orchestrator.process(user_query=test_case["input"])
                response = result["response"]
                response_time = time.time() - start_time

                eval_result = self.evaluate_response(
                    query=test_case["input"],
                    response=response,
                    query_type=test_case["type"],
                    response_time=response_time,
                )
                results.append(eval_result)
                print(f"✅ Accuracy: {eval_result.accuracy_score}% | Time: {eval_result.response_time}s")
            except Exception as e:
                print(f"❌ Test failed: {str(e)}")

        metrics = self.get_aggregate_metrics()
        print(f"\n📊 SUMMARY — Accuracy: {metrics['avg_accuracy']}% | Success rate: {metrics['success_rate']}%")
        return {
            "individual_results": [asdict(r) for r in results],
            "aggregate_metrics": metrics,
            "agent_comparison": self.get_agent_comparison(),
        }

    def save_evaluation_report(self, filename: str = "evaluation_report.json"):
        report = {
            "timestamp": datetime.now().isoformat(),
            "aggregate_metrics": self.get_aggregate_metrics(),
            "agent_comparison": self.get_agent_comparison(),
            "evaluation_history": [asdict(r) for r in self.evaluation_history],
        }
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✅ Report saved to {filename}")