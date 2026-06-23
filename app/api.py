"""
FastAPI Server

REST API for AgriNexus AI inference.
Provides endpoints for question answering and model information.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import os

from src.model import TransformerModel
from src.tokenizer import CharTokenizer
from src.memory import MemoryStore
from src.inference import InferencePipeline, InferenceOutput
from src.rules import RuleEngine


# Pydantic models for API
class QuestionRequest(BaseModel):
    """Request model for inference."""
    question: str
    max_length: int = 100
    temperature: float = 0.7
    retrieve_context: bool = True
    num_retrieved_facts: int = 3


class Answer(BaseModel):
    """Response model for answers."""
    answer: str
    confidence: float
    sources: List[str]


class ModelInfo(BaseModel):
    """Response model for model information."""
    name: str
    vocab_size: int
    model_size: int
    device: str


# Initialize FastAPI app
app = FastAPI(
    title="AgriNexus AI API",
    description="Agricultural language model inference server",
    version="0.1.0",
)


# Global variables for model and components
model = None
tokenizer = None
memory_store = None
pipeline = None
device = None


@app.on_event("startup")
async def startup_event():
    """Initialize model and components on startup."""
    global model, tokenizer, memory_store, pipeline, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Initialize tokenizer
        tokenizer = CharTokenizer()
        # TODO: Load tokenizer from file if it exists
        
        # Initialize memory store
        memory_store = MemoryStore(embedding_dim=256)
        # TODO: Load facts from knowledge base
        memory_store.build_embeddings()
        
        # Initialize model
        model = TransformerModel(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=256,
            num_layers=4,
            num_heads=8,
            d_ff=1024,
        )
        model.to(device)
        
        # TODO: Load pretrained weights if they exist
        
        # Initialize rule engine
        rule_engine = RuleEngine()
        
        # Initialize inference pipeline
        pipeline = InferencePipeline(
            model=model,
            tokenizer=tokenizer,
            memory_store=memory_store,
            rule_engine=rule_engine,
            device=device,
        )
        
        print(f"✅ Model initialized on {device}")
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None,
    }


@app.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        name="AgriNexus AI",
        vocab_size=tokenizer.get_vocab_size(),
        model_size=sum(p.numel() for p in model.parameters()),
        device=device,
    )


@app.post("/predict", response_model=Answer)
async def predict(request: QuestionRequest):
    """
    Predict answer to agricultural question.
    
    Args:
        request: Question request with inference parameters.
    
    Returns:
        Structured answer with confidence and sources.
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


@app.post("/batch-predict")
async def batch_predict(requests: List[QuestionRequest]):
    """
    Predict answers for multiple questions.
    
    Args:
        requests: List of question requests.
    
    Returns:
        List of answers.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
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
    
    return answers


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to AgriNexus AI API",
        "docs": "/docs",
        "version": "0.1.0",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
