"""
Inference Script

Entry point for running inference with AgriNexus AI.
Usage: python infer.py --question "How do I improve soil fertility?"
"""

import argparse
import torch
from pathlib import Path

from src.model import TransformerModel
from src.tokenizer import CharTokenizer
from src.memory import MemoryStore
from src.inference import InferencePipeline
from src.rules import RuleEngine
from src.utils import setup_device


def load_or_create_components(args):
    """Load or create model and supporting components."""
    
    device = setup_device(use_cuda=not args.cpu)
    
    # Load or create tokenizer
    if args.tokenizer_path and Path(args.tokenizer_path).exists():
        tokenizer = CharTokenizer()
        tokenizer.load(args.tokenizer_path)
    else:
        tokenizer = CharTokenizer()
        # Default ASCII vocabulary
        tokenizer.vocab = {
            f"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
            **{chr(i): i + 4 for i in range(32, 127)}
        }
        tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
    
    # Load or create model
    model = TransformerModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    )
    
    if args.model_path and Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    model.to(device)
    
    # Load or create memory store
    memory_store = MemoryStore(embedding_dim=args.d_model)
    
    if args.knowledge_base_path and Path(args.knowledge_base_path).exists():
        memory_store.load(args.knowledge_base_path)
    else:
        # Add some example facts
        example_facts = [
            "Crop rotation improves soil fertility by alternating nutrient demands.",
            "Irrigation systems require proper maintenance to prevent clogging.",
            "Nitrogen, phosphorus, and potassium are essential soil nutrients.",
            "Composting organic waste enriches soil with natural fertilizer.",
            "pH balance is crucial for optimal plant growth.",
        ]
        memory_store.add_facts(example_facts, source="example_knowledge")
        memory_store.build_embeddings()
    
    # Create rule engine
    rule_engine = RuleEngine()
    
    # Create inference pipeline
    pipeline = InferencePipeline(
        model=model,
        tokenizer=tokenizer,
        memory_store=memory_store,
        rule_engine=rule_engine,
        device=device,
    )
    
    return pipeline, device


def main(args):
    """Main inference function."""
    
    print("\n🌾 AgriNexus AI - Inference Script")
    print("=" * 60)
    
    # Load components
    pipeline, device = load_or_create_components(args)
    
    print(f"✅ Model loaded on {device}")
    print("=" * 60)
    
    if args.interactive:
        # Interactive mode
        print("\n💬 Interactive Mode (type 'exit' to quit)\n")
        
        while True:
            question = input("\n❓ Question: ").strip()
            
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Run inference
            output = pipeline(
                question=question,
                max_length=args.max_length,
                temperature=args.temperature,
                retrieve_context=args.retrieve,
                num_retrieved_facts=args.num_facts,
            )
            
            # Print result
            print(f"\n📝 Answer: {output.answer}")
            print(f"📊 Confidence: {output.confidence:.2%}")
            if output.sources:
                print(f"📚 Sources: {', '.join(output.sources)}")
    
    else:
        # Single question mode
        question = args.question
        
        print(f"\n❓ Question: {question}\n")
        
        # Run inference
        output = pipeline(
            question=question,
            max_length=args.max_length,
            temperature=args.temperature,
            retrieve_context=args.retrieve,
            num_retrieved_facts=args.num_facts,
        )
        
        # Print result
        print(f"📝 Answer:")
        print(f"  {output.answer}\n")
        print(f"📊 Confidence: {output.confidence:.2%}")
        
        if output.sources:
            print(f"📚 Sources:")
            for source in output.sources:
                print(f"  - {source}")
        
        # Print as JSON
        if args.json:
            import json
            print(f"\n📄 JSON Output:")
            print(json.dumps(output.to_dict(), indent=2))
    
    print("\n" + "=" * 60)
    print("✅ Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with AgriNexus AI"
    )
    
    # Query parameters
    parser.add_argument("--question", type=str, default=None,
                        help="Question to ask (if not provided, interactive mode)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode (ask multiple questions)")
    
    # Inference parameters
    parser.add_argument("--max-length", type=int, default=100,
                        help="Maximum length of generated answer")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--retrieve", action="store_true", default=True,
                        help="Use memory retrieval for context")
    parser.add_argument("--num-facts", type=int, default=3,
                        help="Number of facts to retrieve")
    
    # Model parameters
    parser.add_argument("--d-model", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=1024,
                        help="Feed-forward dimension")
    
    # Paths
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to pretrained model")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="Path to saved tokenizer")
    parser.add_argument("--knowledge-base-path", type=str, default=None,
                        help="Path to knowledge base file")
    
    # Output
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")
    
    args = parser.parse_args()
    
    # If no question provided and not interactive, use interactive mode
    if args.question is None and not args.interactive:
        args.interactive = True
    
    main(args)
