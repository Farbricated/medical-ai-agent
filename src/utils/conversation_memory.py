from typing import List, Dict, Optional
from datetime import datetime


class ConversationMemory:
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
        self.patient_contexts: Dict[str, Dict] = {}

    def create_session(self, session_id: str) -> None:
        if session_id not in self.conversations:
            self.conversations[session_id] = []
            self.patient_contexts[session_id] = {
                "patient_info": {},
                "symptoms_mentioned": [],
                "diagnoses_received": [],
                "medications_mentioned": [],
                "created_at": datetime.now().isoformat(),
            }

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        self.create_session(session_id)
        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "metadata": metadata or {},
        }
        self.conversations[session_id].append(message)
        if role == "assistant" and metadata:
            self._update_patient_context(session_id, metadata)

    def _update_patient_context(self, session_id: str, metadata: Dict) -> None:
        context = self.patient_contexts[session_id]
        if "patient_info" in metadata:
            context["patient_info"].update(metadata["patient_info"])
        if "symptoms" in metadata:
            for symptom in metadata["symptoms"]:
                if symptom not in context["symptoms_mentioned"]:
                    context["symptoms_mentioned"].append(symptom)
        if "diagnosis" in metadata:
            context["diagnoses_received"].append(
                {
                    "diagnosis": metadata["diagnosis"],
                    "confidence": metadata.get("confidence", 0),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        if "medications" in metadata:
            for med in metadata["medications"]:
                if med not in context["medications_mentioned"]:
                    context["medications_mentioned"].append(med)

    def get_conversation(
        self, session_id: str, last_n: Optional[int] = None
    ) -> List[Dict]:
        if session_id not in self.conversations:
            return []
        messages = self.conversations[session_id]
        return messages[-last_n:] if last_n else messages

    def get_patient_context(self, session_id: str) -> Dict:
        return self.patient_contexts.get(session_id, {})

    def get_context_summary(self, session_id: str) -> str:
        context = self.get_patient_context(session_id)
        if not context:
            return "No patient context available."
        parts = []
        if context.get("patient_info"):
            info = context["patient_info"]
            parts.append(
                f"Patient Information: {', '.join(f'{k}: {v}' for k, v in info.items())}"
            )
        if context.get("symptoms_mentioned"):
            parts.append(f"Symptoms mentioned: {', '.join(context['symptoms_mentioned'])}")
        if context.get("diagnoses_received"):
            diagnoses = [d["diagnosis"] for d in context["diagnoses_received"]]
            parts.append(f"Previous diagnoses: {', '.join(diagnoses)}")
        if context.get("medications_mentioned"):
            parts.append(f"Medications discussed: {', '.join(context['medications_mentioned'])}")
        return "\n".join(parts) if parts else "No context available."

    def clear_session(self, session_id: str) -> None:
        self.conversations.pop(session_id, None)
        self.patient_contexts.pop(session_id, None)

    def export_session(self, session_id: str) -> Dict:
        return {
            "session_id": session_id,
            "conversation": self.conversations.get(session_id, []),
            "patient_context": self.patient_contexts.get(session_id, {}),
        }