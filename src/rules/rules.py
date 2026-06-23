"""
Rule Engine

Implements domain constraints and validation rules
to enforce consistency and prevent hallucinations.
"""

from typing import List, Dict, Callable
import re


class Rule:
    """Single validation rule."""
    
    def __init__(
        self,
        name: str,
        condition: Callable[[str], bool],
        action: Callable[[str], str] = None,
    ):
        """
        Initialize rule.
        
        Args:
            name: Rule name.
            condition: Function that returns True if rule is violated.
            action: Function to fix violation (optional).
        """
        self.name = name
        self.condition = condition
        self.action = action or (lambda x: x)
    
    def check(self, text: str) -> bool:
        """Check if rule is violated."""
        return self.condition(text)
    
    def fix(self, text: str) -> str:
        """Apply fix action."""
        return self.action(text)


class RuleEngine:
    """Rule-based validation and correction engine."""
    
    def __init__(self):
        """Initialize rule engine."""
        self.rules: List[Rule] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Set up default agricultural domain rules."""
        
        # Rule 1: Prevent completely empty responses
        def empty_check(text: str) -> bool:
            return len(text.strip()) == 0
        
        def empty_fix(text: str) -> str:
            if len(text.strip()) == 0:
                return "I need more information to answer this question."
            return text
        
        self.add_rule(Rule(
            "non_empty_response",
            empty_check,
            empty_fix
        ))
        
        # Rule 2: Ensure answers are relevant to agriculture
        def agriculture_check(text: str) -> bool:
            # Check if response contains agricultural keywords
            agri_keywords = [
                'crop', 'soil', 'farm', 'irrigation', 'harvest',
                'yield', 'fertilizer', 'plant', 'agriculture'
            ]
            text_lower = text.lower()
            return not any(keyword in text_lower for keyword in agri_keywords)
        
        self.add_rule(Rule(
            "agricultural_relevance",
            agriculture_check,
        ))
        
        # Rule 3: Limit response length
        def length_check(text: str) -> bool:
            return len(text) > 1000
        
        def length_fix(text: str) -> str:
            if len(text) > 1000:
                return text[:997] + "..."
            return text
        
        self.add_rule(Rule(
            "reasonable_length",
            length_check,
            length_fix
        ))
        
        # Rule 4: Ensure proper sentence structure
        def sentence_check(text: str) -> bool:
            # Check if response starts with lowercase
            return len(text) > 0 and text[0].islower()
        
        def sentence_fix(text: str) -> str:
            if len(text) > 0 and text[0].islower():
                return text[0].upper() + text[1:]
            return text
        
        self.add_rule(Rule(
            "proper_capitalization",
            sentence_check,
            sentence_fix
        ))
    
    def add_rule(self, rule: Rule):
        """Add a rule to the engine."""
        self.rules.append(rule)
    
    def validate(self, text: str) -> str:
        """
        Validate and fix text according to all rules.
        
        Args:
            text: Text to validate.
        
        Returns:
            Corrected text.
        """
        for rule in self.rules:
            if rule.check(text):
                text = rule.fix(text)
        
        return text
    
    def check_all(self, text: str) -> Dict[str, bool]:
        """
        Check all rules on text.
        
        Args:
            text: Text to check.
        
        Returns:
            Dictionary of rule names -> violation status.
        """
        results = {}
        for rule in self.rules:
            results[rule.name] = rule.check(text)
        
        return results
    
    def get_violations(self, text: str) -> List[str]:
        """
        Get list of violated rules.
        
        Args:
            text: Text to check.
        
        Returns:
            List of violated rule names.
        """
        violations = []
        for rule in self.rules:
            if rule.check(text):
                violations.append(rule.name)
        
        return violations
