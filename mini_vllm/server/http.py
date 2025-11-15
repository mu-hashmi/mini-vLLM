"""HTTP server using FastAPI."""

import asyncio
import json
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from loguru import logger

from mini_vllm.engine.runtime import InferenceRuntime
from mini_vllm.metrics.stats import get_stats
from mini_vllm.server.schemas import GenerateRequest, GenerateResponse, StreamChunk


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
            raise HTTPException(status_code=503, detail="Runtime not initialized")

        try:
            logger.info(
                f"Generate request: prompt_len={len(request.prompt)}, max_tokens={request.max_tokens}"
            )

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

    @app.post("/v1/generate_stream")
    async def generate_stream(request: GenerateRequest):
        """Generate text with Server-Sent Events (SSE) streaming."""
        if runtime is None:
            raise HTTPException(status_code=503, detail="Runtime not initialized")

        async def event_generator():
            try:
                logger.debug(
                    f"[HTTP] Stream request received: prompt_len={len(request.prompt)}, max_tokens={request.max_tokens}"
                )

                stats = get_stats()
                start_time = time.time()
                first_token_time = None
                token_count = 0
                accumulated_text = ""
                ttft = None
                tokens_per_sec = None
                last_yield_time = start_time

                logger.debug(
                    "[HTTP] Starting to iterate over runtime.generate_stream()..."
                )

                # Stream tokens - iterate over sync generator, yielding control to event loop after each token
                generator = runtime.generate_stream(
                    prompt=request.prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    stop_sequences=request.stop,
                )

                # Iterate over sync generator, yielding to event loop after each token
                for token, accumulated_text, timestamp in generator:
                    loop_start = time.time()
                    logger.debug(
                        f"[HTTP] Token received from generator: token={repr(token[:20])}, "
                        f"count={token_count + 1}, timestamp={timestamp:.3f}, "
                        f"time_since_last_yield={loop_start - last_yield_time:.3f}s"
                    )

                    # Record first token time
                    if first_token_time is None:
                        first_token_time = timestamp
                        ttft = first_token_time - start_time
                        stats.record_ttft(ttft)
                        logger.debug(f"[HTTP] First token received! TTFT={ttft:.3f}s")
                    else:
                        ttft = None

                    token_count += 1

                    # Calculate tokens/sec
                    elapsed = timestamp - start_time
                    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0.0

                    # Create chunk
                    chunk = StreamChunk(
                        token=token,
                        text=accumulated_text,
                        finished=False,
                        ttft=ttft,
                        tokens_per_sec=tokens_per_sec if token_count > 1 else None,
                    )

                    # Emit metrics
                    if tokens_per_sec > 0:
                        stats.record_tokens_per_sec(tokens_per_sec)

                    # Send SSE event
                    data = chunk.model_dump_json()
                    before_yield = time.time()
                    logger.debug(
                        f"[HTTP] About to yield SSE data for token {token_count}, data_len={len(data)}"
                    )
                    yield f"data: {data}\n\n"
                    # Yield control to event loop to ensure data is flushed
                    await asyncio.sleep(0)
                    after_yield = time.time()
                    last_yield_time = after_yield
                    logger.debug(
                        f"[HTTP] Yielded token {token_count}, yield_took={(after_yield - before_yield) * 1000:.2f}ms"
                    )

                logger.debug(
                    f"[HTTP] Generator loop finished. Total tokens: {token_count}"
                )

                # Send final chunk
                final_chunk = StreamChunk(
                    token="",
                    text=accumulated_text,
                    finished=True,
                    ttft=ttft,
                    tokens_per_sec=tokens_per_sec,
                )
                data = final_chunk.model_dump_json()
                logger.debug("[HTTP] Yielding final chunk")
                yield f"data: {data}\n\n"
                logger.debug("[HTTP] Final chunk yielded, stream complete")

            except Exception as e:
                logger.error(f"[HTTP] Streaming error: {e}", exc_info=True)
                error_data = json.dumps({"error": str(e)})
                yield f"data: {error_data}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.websocket("/v1/generate_stream_ws")
    async def generate_stream_ws(websocket: WebSocket):
        """Generate text with WebSocket streaming."""
        await websocket.accept()

        if runtime is None:
            await websocket.send_json({"error": "Runtime not initialized"})
            await websocket.close()
            return

        try:
            # Receive request
            data = await websocket.receive_json()
            request = GenerateRequest(**data)

            logger.info(
                f"WebSocket stream request: prompt_len={len(request.prompt)}, max_tokens={request.max_tokens}"
            )

            stats = get_stats()
            start_time = time.time()
            first_token_time = None
            token_count = 0
            accumulated_text = ""
            ttft = None
            tokens_per_sec = None

            # Stream tokens
            for token, accumulated_text, timestamp in runtime.generate_stream(
                prompt=request.prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                stop_sequences=request.stop,
            ):
                # Record first token time
                if first_token_time is None:
                    first_token_time = timestamp
                    ttft = first_token_time - start_time
                    stats.record_ttft(ttft)
                else:
                    ttft = None

                token_count += 1

                # Calculate tokens/sec
                elapsed = timestamp - start_time
                tokens_per_sec = token_count / elapsed if elapsed > 0 else 0.0

                # Create chunk
                chunk = StreamChunk(
                    token=token,
                    text=accumulated_text,
                    finished=False,
                    ttft=ttft,
                    tokens_per_sec=tokens_per_sec if token_count > 1 else None,
                )

                # Emit metrics
                if tokens_per_sec > 0:
                    stats.record_tokens_per_sec(tokens_per_sec)

                # Send WebSocket message
                await websocket.send_json(chunk.model_dump())

            # Send final chunk
            final_chunk = StreamChunk(
                token="",
                text=accumulated_text,
                finished=True,
                ttft=ttft,
                tokens_per_sec=tokens_per_sec,
            )
            await websocket.send_json(final_chunk.model_dump())

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket streaming error: {e}", exc_info=True)
            try:
                await websocket.send_json({"error": str(e)})
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    @app.get("/v1/metrics")
    def get_metrics():
        """Get current metrics statistics."""
        stats = get_stats()
        return {
            "ttft": stats.get_ttft_stats(),
            "tokens_per_sec": stats.get_tokens_per_sec_stats(),
        }

    return app
