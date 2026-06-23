"""
Inference Pipeline

Orchestrates tokenization, model inference, memory retrieval,
and rule-based output generation.
"""

import torch
import json
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class InferenceOutput:
    """Structured output format."""
    answer: str
    confidence: float
    sources: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": self.sources,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class InferencePipeline:
    """End-to-end inference pipeline."""
    
    def __init__(
        self,
        model,
        tokenizer,
        memory_store,
        rule_engine=None,
        device: str = "cpu",
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model: Transformer model.
            tokenizer: Text tokenizer.
            memory_store: Memory/knowledge store.
            rule_engine: Optional rule engine for output validation.
            device: Device to run inference on (cpu/cuda).
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.memory_store = memory_store
        self.rule_engine = rule_engine
        self.device = device
        self.model.eval()
    
    def __call__(
        self,
        question: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 40,
        retrieve_context: bool = True,
        num_retrieved_facts: int = 3,
    ) -> InferenceOutput:
        """
        Run inference on a question.
        
        Args:
            question: Agricultural question.
            max_length: Maximum output length.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            retrieve_context: Whether to retrieve context from memory.
            num_retrieved_facts: Number of facts to retrieve.
        
        Returns:
            Structured inference output.
        """
        with torch.no_grad():
            # Step 1: Retrieve relevant facts
            retrieved_facts = []
            retrieved_sources = []
            
            if retrieve_context and self.memory_store is not None:
                results = self.memory_store.retrieve(
                    question,
                    top_k=num_retrieved_facts
                )
                
                for fact, confidence, source in results:
                    retrieved_facts.append(fact)
                    retrieved_sources.append(source)
            
            # Step 2: Construct augmented input
            augmented_input = question
            if retrieved_facts:
                context = "\n".join(f"- {fact}" for fact in retrieved_facts)
                augmented_input = f"{question}\n\nContext:\n{context}\n\nAnswer:"
            
            # Step 3: Tokenize
            input_ids = self.tokenizer.encode(
                augmented_input,
                add_special_tokens=True
            )
            input_tensor = torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=self.device
            )
            
            # Step 4: Generate with model
            output_ids = self.model.generate(
                input_tensor,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
            )
            
            # Step 5: Decode output
            generated_text = self.tokenizer.decode(
                output_ids[0].tolist(),
                skip_special_tokens=True
            )
            
            # Step 6: Parse answer
            answer = self._parse_answer(generated_text)
            
            # Step 7: Compute confidence
            confidence = self._compute_confidence(
                generated_text,
                len(retrieved_facts)
            )
            
            # Step 8: Apply rule engine
            if self.rule_engine is not None:
                answer = self.rule_engine.validate(answer)
        
        return InferenceOutput(
            answer=answer,
            confidence=confidence,
            sources=retrieved_sources,
        )
    
    def _parse_answer(self, text: str) -> str:
        """Extract answer from generated text."""
        # Simple heuristic: take first sentence
        sentences = text.split('.')
        if sentences:
            return sentences[0].strip() + '.'
        return text.strip()
    
    def _compute_confidence(self, text: str, num_sources: int) -> float:
        """
        Compute confidence score based on text length and sources.
        
        Args:
            text: Generated text.
            num_sources: Number of retrieved sources.
        
        Returns:
            Confidence score between 0 and 1.
        """
        # Longer, sourced answers have higher confidence
        base_confidence = min(len(text) / 200, 1.0)  # Max confidence at 200 chars
        source_boost = min(num_sources * 0.1, 0.3)  # Up to 0.3 boost
        
        confidence = min(base_confidence + source_boost, 1.0)
        return round(confidence, 2)
