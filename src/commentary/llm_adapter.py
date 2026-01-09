# src/commentary/llm_adapter.py

"""
LLM adapter module for sports analytics system.

Provides interface for generating commentary text from prompts.
Current implementation is a mock/stub for testing without real LLM calls.
"""

from typing import Optional


class LLMAdapter:
    """
    Adapter for LLM-based commentary generation.
    
    Mock implementation that returns placeholder text instead of
    calling real LLM APIs. Can be replaced with actual LLM integration
    in production.
    
    Usage:
        adapter = LLMAdapter()
        commentary = adapter.generate(prompt)
        if commentary:
            # Output or display commentary
            pass
    """
    
    def generate(self, prompt: Optional[str]) -> Optional[str]:
        """
        Generate commentary text from prompt.
        
        Args:
            prompt: Natural language prompt string
        
        Returns:
            Generated commentary text, or None if prompt is empty
        
        Notes:
            - Current implementation returns placeholder text
            - Replace with actual LLM call for production use
        """
        if prompt is None or len(prompt.strip()) == 0:
            return None
        
        return "[COMMENTARY GENERATED]"