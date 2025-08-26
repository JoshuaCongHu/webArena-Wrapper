# api/server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import torch
import asyncio
import uvicorn
import logging
import time
import os
import sys
from pathlib import Path

# Add workspace to path
sys.path.append('/workspace')
sys.path.append('/workspace/mas_webarena')

from mas.enhanced_webarena_mas import EnhancedWebArenaMAS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="WebArena MAS API",
    description="API for WebArena Multi-Agent System with LLM Orchestrator",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
mas_model = None
model_lock = asyncio.Lock()

# Request/Response models
class TaskRequest(BaseModel):
    intent: str = Field(..., description="Task intent/goal")
    sites: List[str] = Field(default=["example.com"], description="Target websites")
    budget: float = Field(default=1.0, gt=0, description="Budget constraint")
    method: str = Field(default="p3o", description="RL method to use")
    use_replanning: bool = Field(default=True, description="Enable dynamic replanning")
    expected_steps: int = Field(default=5, description="Expected number of steps")

class TaskResponse(BaseModel):
    success: bool
    cost: float
    reward: float
    dag: Dict[str, Any]
    trajectory: List[Dict[str, Any]]
    replanning_count: int
    execution_time: float
    method_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    available_methods: List[str]
    gpu_available: bool
    gpu_count: int

# Dependency for getting model
async def get_model():
    if mas_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return mas_model

@app.on_event("startup")
async def startup():
    """Load model on startup"""
    global mas_model
    
    try:
        checkpoint_path = os.environ.get(
            'MODEL_PATH',
            '/workspace/models/best_checkpoint.pt'
        )
        
        if Path(checkpoint_path).exists():
            logger.info(f"Loading model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
            
            # Create MAS instance
            mas_model = EnhancedWebArenaMAS(
                method=checkpoint.get('method', 'p3o'),
                budget=1.0,
                use_llm_orchestrator=True,
                device='cuda:0' if torch.cuda.is_available() else 'cpu'
            )
            
            # Load model weights if available
            if 'algorithm_state' in checkpoint:
                mas_model.algorithm.load_state_dict(checkpoint['algorithm_state'])
            
            logger.info("Model loaded successfully")
        else:
            logger.warning("No checkpoint found, creating new model")
            mas_model = EnhancedWebArenaMAS(
                method='p3o',
                budget=1.0,
                use_llm_orchestrator=True,
                device='cuda:0' if torch.cuda.is_available() else 'cpu'
            )
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        mas_model = None

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global mas_model
    mas_model = None

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "WebArena MAS API", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if mas_model is not None else "unhealthy",
        model_loaded=mas_model is not None,
        available_methods=["p3o", "ppo_lagrangian", "macpo"],
        gpu_available=torch.cuda.is_available(),
        gpu_count=torch.cuda.device_count()
    )

@app.post("/solve_task", response_model=TaskResponse)
async def solve_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    model=Depends(get_model)
):
    """Solve a WebArena task"""
    
    start_time = time.time()
    
    # Prepare task
    task = {
        'intent': request.intent,
        'sites': request.sites,
        'expected_steps': request.expected_steps,
        'difficulty': 'medium'  # Could be inferred from intent
    }
    
    # Use model lock to prevent concurrent modifications
    async with model_lock:
        # Update model configuration if needed
        if request.method != model.method:
            logger.warning(f"Method mismatch: requested {request.method}, model uses {model.method}")
        
        # Update budget
        model.budget_tracker.reset(request.budget)
        
        # Execute task (run in executor to avoid blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, model.solve_task, task)
    
    execution_time = time.time() - start_time
    
    # Log execution in background
    background_tasks.add_task(
        log_task_execution,
        request,
        result,
        execution_time
    )
    
    return TaskResponse(
        success=result['success'],
        cost=result['cost'],
        reward=result['reward'],
        dag=result.get('dag', {}),
        trajectory=result.get('trajectory', []),
        replanning_count=result.get('replanning_count', 0),
        execution_time=execution_time,
        method_info=result.get('method_info', {})
    )

@app.post("/batch_solve", response_model=List[TaskResponse])
async def batch_solve(
    requests: List[TaskRequest],
    model=Depends(get_model)
):
    """Solve multiple tasks in batch"""
    
    responses = []
    
    for request in requests:
        response = await solve_task(request, BackgroundTasks(), model)
        responses.append(response)
    
    return responses

def log_task_execution(
    request: TaskRequest,
    result: Dict[str, Any],
    execution_time: float
):
    """Log task execution for monitoring"""
    log_entry = {
        'timestamp': time.time(),
        'request': request.dict(),
        'result': {
            'success': result['success'],
            'cost': result['cost'],
            'reward': result['reward'],
            'replanning_count': result.get('replanning_count', 0)
        },
        'execution_time': execution_time
    }
    
    # Log to file or database
    logger.info(f"Task execution: {log_entry}")

def main():
    """Run the API server"""
    port = int(os.environ.get('API_PORT', 8000))
    workers = int(os.environ.get('API_WORKERS', 4))
    
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()