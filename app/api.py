"""
FastAPI Server

REST API for AgriNexus AI inference.
Provides endpoints for question answering and model information.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
import os
import json
from datetime import datetime

from src.model import TransformerModel
from src.tokenizer import CharTokenizer
from src.memory import MemoryStore
from src.inference import InferencePipeline, InferenceOutput
from src.rules import RuleEngine
from src.utils import count_parameters


# Pydantic models for API
class QuestionRequest(BaseModel):
    """Request model for inference."""
    question: str = Field(..., description="Agricultural question to answer")
    max_length: int = Field(100, description="Maximum length of generated answer", ge=10, le=500)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.1, le=2.0)
    retrieve_context: bool = Field(True, description="Whether to retrieve context from knowledge base")
    num_retrieved_facts: int = Field(3, description="Number of facts to retrieve", ge=1, le=10)


class Answer(BaseModel):
    """Response model for answers."""
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., description="Confidence score (0-1)")
    sources: List[str] = Field(default_factory=list, description="Source documents")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ModelInfo(BaseModel):
    """Response model for model information."""
    name: str
    vocab_size: int
    model_size: int
    total_parameters: int
    device: str
    version: str


class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    device: str
    model_loaded: bool
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class KnowledgeFact(BaseModel):
    """Model for knowledge base facts."""
    fact: str = Field(..., description="Agricultural fact")
    source: str = Field(default="user_added", description="Source of the fact")


class KnowledgeBase(BaseModel):
    """Knowledge base response."""
    total_facts: int
    facts: List[dict] = Field(default_factory=list, description="List of facts")


class RetrievalResult(BaseModel):
    """Result of knowledge retrieval."""
    query: str
    results: List[dict] = Field(default_factory=list, description="Retrieved facts with similarity scores")


# Initialize FastAPI app with CORS
app = FastAPI(
    title="AgriNexus AI API",
    description="Agricultural language model inference server - From-scratch transformer for farming domain",
    version="0.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables for model and components
model = None
tokenizer = None
memory_store = None
pipeline = None
device = None
rule_engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize model and components on startup."""
    global model, tokenizer, memory_store, pipeline, device, rule_engine
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer = CharTokenizer()
        tokenizer.vocab = {
            "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
            **{chr(i): i + 4 for i in range(32, 127)}
        }
        tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
        print(f"✅ Tokenizer initialized (vocab size: {tokenizer.get_vocab_size()})")
        
        # Initialize memory store
        print("Initializing memory store...")
        memory_store = MemoryStore(embedding_dim=256)
        
        # Load knowledge base if it exists
        kb_path = "data/knowledge_base.json"
        if os.path.exists(kb_path):
            try:
                memory_store.load(kb_path)
                print(f"✅ Loaded {len(memory_store.facts)} facts from knowledge base")
            except Exception as e:
                print(f"⚠️  Could not load knowledge base: {e}")
        else:
            # Add default facts
            example_facts = [
                "Crop rotation improves soil fertility by alternating nutrient demands.",
                "Irrigation systems require proper maintenance to prevent clogging.",
                "Nitrogen, phosphorus, and potassium are essential soil nutrients.",
                "Composting organic waste enriches soil with natural fertilizer.",
                "pH balance is crucial for optimal plant growth.",
                "Soil moisture affects crop yield significantly.",
                "Pest management requires integrated strategies.",
                "Crop spacing influences air circulation and disease prevention.",
            ]
            memory_store.add_facts(example_facts, source="agrinexus_knowledge_base")
        
        memory_store.build_embeddings()
        print(f"✅ Memory store initialized with {len(memory_store.facts)} facts")
        
        # Initialize model
        print("Initializing transformer model...")
        model = TransformerModel(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=256,
            num_layers=4,
            num_heads=8,
            d_ff=1024,
        )
        model.to(device)
        
        # Try to load pretrained weights
        model_path = "checkpoints/best_model.pt"
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"✅ Loaded pretrained model from {model_path}")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
        
        total_params = count_parameters(model)
        print(f"✅ Model initialized with {total_params:,} trainable parameters")
        
        # Initialize rule engine
        print("Initializing rule engine...")
        rule_engine = RuleEngine()
        print(f"✅ Rule engine initialized with {len(rule_engine.rules)} rules")
        
        # Initialize inference pipeline
        print("Initializing inference pipeline...")
        pipeline = InferencePipeline(
            model=model,
            tokenizer=tokenizer,
            memory_store=memory_store,
            rule_engine=rule_engine,
            device=device,
        )
        
        print(f"\n✅ All components initialized successfully on {device}")
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        raise


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    return HealthStatus(
        status="healthy" if pipeline is not None else "initializing",
        device=device or "cpu",
        model_loaded=model is not None,
    )


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AgriNexus AI API",
        "version": "0.1.0",
        "description": "Agricultural language model inference server",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "knowledge_base": "/knowledge-base",
            "retrieve": "/knowledge-base/retrieve",
            "add_fact": "/knowledge-base/add",
        }
    }


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get detailed model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        name="AgriNexus AI Transformer",
        vocab_size=tokenizer.get_vocab_size(),
        model_size=model.d_model,
        total_parameters=count_parameters(model),
        device=device,
        version="0.1.0",
    )


@app.post("/predict", response_model=Answer, tags=["Inference"])
async def predict(request: QuestionRequest):
    """
    Predict answer to agricultural question.
    
    **Example:**
    ```json
    {
        "question": "How do I improve soil fertility?",
        "max_length": 100,
        "temperature": 0.7,
        "retrieve_context": true,
        "num_retrieved_facts": 3
    }
    ```
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Run inference
        output = pipeline(
            question=request.question,
            max_length=request.max_length,
            temperature=request.temperature,
            retrieve_context=request.retrieve_context,
            num_retrieved_facts=request.num_retrieved_facts,
        )
        
        return Answer(
            answer=output.answer,
            confidence=output.confidence,
            sources=output.sources,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/batch-predict", tags=["Inference"])
async def batch_predict(requests: List[QuestionRequest]):
    """
    Predict answers for multiple questions at once.
    
    **Example:**
    ```json
    [
        {"question": "How do I improve soil fertility?"},
        {"question": "What is crop rotation?"}
    ]
    ```
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        answers = []
        for request in requests:
            output = pipeline(
                question=request.question,
                max_length=request.max_length,
                temperature=request.temperature,
                retrieve_context=request.retrieve_context,
                num_retrieved_facts=request.num_retrieved_facts,
            )
            
            answers.append(Answer(
                answer=output.answer,
                confidence=output.confidence,
                sources=output.sources,
            ))
        
        return {"count": len(answers), "answers": answers}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")


@app.get("/knowledge-base", response_model=KnowledgeBase, tags=["Knowledge Base"])
async def get_knowledge_base(limit: int = 10):
    """Get knowledge base facts."""
    if memory_store is None:
        raise HTTPException(status_code=503, detail="Memory store not initialized")
    
    facts_data = []
    for i, (fact, meta) in enumerate(zip(memory_store.facts, memory_store.metadata)):
        if i >= limit:
            break
        facts_data.append({
            "id": i,
            "fact": fact,
            "source": meta.get("source", "unknown"),
        })
    
    return KnowledgeBase(
        total_facts=len(memory_store.facts),
        facts=facts_data,
    )


@app.post("/knowledge-base/add", tags=["Knowledge Base"])
async def add_knowledge_fact(fact_request: KnowledgeFact):
    """Add a new fact to the knowledge base."""
    if memory_store is None:
        raise HTTPException(status_code=503, detail="Memory store not initialized")
    
    try:
        memory_store.add_fact(fact_request.fact, source=fact_request.source)
        memory_store.build_embeddings()
        
        # Save to file
        memory_store.save("data/knowledge_base.json")
        
        return {
            "status": "success",
            "message": f"Added fact from source '{fact_request.source}'",
            "total_facts": len(memory_store.facts),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding fact: {str(e)}")


@app.post("/knowledge-base/retrieve", response_model=RetrievalResult, tags=["Knowledge Base"])
async def retrieve_knowledge(query: str, top_k: int = 5):
    """Retrieve relevant facts from knowledge base."""
    if memory_store is None:
        raise HTTPException(status_code=503, detail="Memory store not initialized")
    
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
    
    try:
        results = memory_store.retrieve(query, top_k=top_k)
        
        results_data = []
        for fact, similarity, source in results:
            results_data.append({
                "fact": fact,
                "similarity": round(float(similarity), 3),
                "source": source,
            })
        
        return RetrievalResult(
            query=query,
            results=results_data,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")


@app.get("/model/stats", tags=["Model"])
async def get_model_stats():
    """Get model statistics."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "name": "AgriNexus AI",
        "device": device,
        "embedding_dim": model.d_model,
        "num_layers": len(model.layers),
        "total_parameters": count_parameters(model),
        "model_dtype": str(list(model.parameters())[0].dtype),
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
