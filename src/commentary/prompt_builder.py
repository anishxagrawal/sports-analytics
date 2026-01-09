# src/commentary/prompt_builder.py

"""
Prompt builder module for sports analytics system.

Converts narrative intents into constrained natural-language prompts
for LLM-based commentary generation.
"""

from typing import List, Dict, Optional


# Intent to sentence template mapping
INTENT_TEMPLATES = {
    "start_run": "Player {entity_id} starts a fast run.",
    "slow_down": "Player {entity_id} slows down."
}


class PromptBuilder:
    """
    Builds LLM prompts from narrative intents.
    
    Converts structured intent dictionaries into factual sentence
    descriptions and assembles them into a commentary prompt.
    
    Usage:
        builder = PromptBuilder()
        prompt = builder.build(intents)
        if prompt:
            # Send prompt to LLM
            pass
    """
    
    def build(self, intents: List[Dict[str, str]]) -> Optional[str]:
        """
        Build commentary prompt from narrative intents.
        
        Args:
            intents: List of intent dictionaries with:
                - entity_id: str
                - intent: str
        
        Returns:
            Formatted prompt string for LLM, or None if no valid intents
        
        Notes:
            - Only processes intents with known templates
            - Returns None if no sentences can be generated
            - Output is deterministic and factual
        """
        if len(intents) == 0:
            return None
        
        sentences = []
        
        for intent_dict in intents:
            entity_id = intent_dict.get("entity_id")
            intent = intent_dict.get("intent")
            
            if entity_id is None or intent is None:
                continue
            
            if intent not in INTENT_TEMPLATES:
                continue
            
            template = INTENT_TEMPLATES[intent]
            sentence = template.format(entity_id=entity_id)
            sentences.append(sentence)
        
        if len(sentences) == 0:
            return None
        
        prompt_lines = [
            "You are a football commentator.",
            "Describe the following events concisely and factually:",
            ""
        ]
        
        for sentence in sentences:
            prompt_lines.append(f"- {sentence}")
        
        prompt = "\n".join(prompt_lines)
        
        return prompt