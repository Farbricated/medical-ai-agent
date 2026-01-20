from typing import List, Dict, Optional
from datetime import datetime
import json

class ConversationMemory:
    """Manages conversation history and patient context"""
    
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
        self.patient_contexts: Dict[str, Dict] = {}
    
    def create_session(self, session_id: str) -> None:
        """Create a new conversation session"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
            self.patient_contexts[session_id] = {
                'patient_info': {},
                'symptoms_mentioned': [],
                'diagnoses_received': [],
                'medications_mentioned': [],
                'created_at': datetime.now().isoformat()
            }
    
    def add_message(self, session_id: str, role: str, content: str, 
                   metadata: Optional[Dict] = None) -> None:
        """Add a message to conversation history"""
        self.create_session(session_id)
        
        message = {
            'timestamp': datetime.now().isoformat(),
            'role': role,  # 'user' or 'assistant'
            'content': content,
            'metadata': metadata or {}
        }
        
        self.conversations[session_id].append(message)
        
        # Extract and update patient context
        if role == 'assistant' and metadata:
            self._update_patient_context(session_id, metadata)
    
    def _update_patient_context(self, session_id: str, metadata: Dict) -> None:
        """Extract patient information from metadata"""
        context = self.patient_contexts[session_id]
        
        # Extract patient info (age, gender, etc.)
        if 'patient_info' in metadata:
            context['patient_info'].update(metadata['patient_info'])
        
        # Track symptoms
        if 'symptoms' in metadata:
            for symptom in metadata['symptoms']:
                if symptom not in context['symptoms_mentioned']:
                    context['symptoms_mentioned'].append(symptom)
        
        # Track diagnoses
        if 'diagnosis' in metadata:
            diagnosis = metadata['diagnosis']
            context['diagnoses_received'].append({
                'diagnosis': diagnosis,
                'confidence': metadata.get('confidence', 0),
                'timestamp': datetime.now().isoformat()
            })
        
        # Track medications
        if 'medications' in metadata:
            for med in metadata['medications']:
                if med not in context['medications_mentioned']:
                    context['medications_mentioned'].append(med)
    
    def get_conversation(self, session_id: str, 
                        last_n: Optional[int] = None) -> List[Dict]:
        """Get conversation history"""
        if session_id not in self.conversations:
            return []
        
        messages = self.conversations[session_id]
        if last_n:
            return messages[-last_n:]
        return messages
    
    def get_patient_context(self, session_id: str) -> Dict:
        """Get accumulated patient context"""
        return self.patient_contexts.get(session_id, {})
    
    def get_context_summary(self, session_id: str) -> str:
        """Get a formatted summary of patient context"""
        context = self.get_patient_context(session_id)
        
        if not context:
            return "No patient context available."
        
        summary_parts = []
        
        # Patient info
        if context.get('patient_info'):
            info = context['patient_info']
            summary_parts.append(f"Patient Information: {', '.join([f'{k}: {v}' for k, v in info.items()])}")
        
        # Symptoms
        if context.get('symptoms_mentioned'):
            symptoms = ', '.join(context['symptoms_mentioned'])
            summary_parts.append(f"Symptoms mentioned: {symptoms}")
        
        # Diagnoses
        if context.get('diagnoses_received'):
            diagnoses = [d['diagnosis'] for d in context['diagnoses_received']]
            summary_parts.append(f"Previous diagnoses: {', '.join(diagnoses)}")
        
        # Medications
        if context.get('medications_mentioned'):
            meds = ', '.join(context['medications_mentioned'])
            summary_parts.append(f"Medications discussed: {meds}")
        
        return "\n".join(summary_parts) if summary_parts else "No context available."
    
    def clear_session(self, session_id: str) -> None:
        """Clear a conversation session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
        if session_id in self.patient_contexts:
            del self.patient_contexts[session_id]
    
    def export_session(self, session_id: str) -> Dict:
        """Export session data as JSON"""
        return {
            'session_id': session_id,
            'conversation': self.conversations.get(session_id, []),
            'patient_context': self.patient_contexts.get(session_id, {})
        }
    
    def import_session(self, data: Dict) -> None:
        """Import session data from JSON"""
        session_id = data['session_id']
        self.conversations[session_id] = data['conversation']
        self.patient_contexts[session_id] = data['patient_context']