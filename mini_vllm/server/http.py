"""HTTP server using FastAPI."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from typing import Optional

from mini_vllm.server.schemas import GenerateRequest, GenerateResponse
from mini_vllm.engine.runtime import InferenceRuntime


def create_app(runtime: Optional[InferenceRuntime] = None) -> FastAPI:
    """Create FastAPI app.
    
    Args:
        runtime: Optional InferenceRuntime instance. If None, endpoints will return errors.
        
    Returns:
        FastAPI app instance
    """
    app = FastAPI(title="mini-vLLM", version="0.1.0")
    
    @app.get("/healthz")
    def healthz():
        """Health check endpoint."""
        return {"status": "ok"}
    
    @app.post("/v1/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest):
        """Generate text from prompt."""
        if runtime is None:
            raise HTTPException(
                status_code=503,
                detail="Runtime not initialized"
            )
        
        try:
            logger.info(f"Generate request: prompt_len={len(request.prompt)}, max_tokens={request.max_tokens}")
            
            # Call runtime
            generated_text, generated_tokens = runtime.generate(
                prompt=request.prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                stop_sequences=request.stop,
            )
            
            return GenerateResponse(
                text=generated_text,
                tokens=generated_tokens,
                num_tokens=len(generated_tokens),
            )
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    return app
