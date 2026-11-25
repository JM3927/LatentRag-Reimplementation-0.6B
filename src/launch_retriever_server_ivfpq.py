"""REST retriever server on the IVF-PQ index (Qwen3-0.6B port; not in the original repo).

Same endpoints and response format as src/launch_retriever_server.py, so
retriever.rest_retriever.RESTRetriever (used by evaluate*.py when retriever.base_url is
set) works unchanged. Config: config/retriever_server_ivfpq.yaml.
"""
import logging
import os
from typing import List, Optional, Union

import hydra
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from omegaconf import DictConfig
from pydantic import BaseModel

from retriever.encoder import Encoder
from retriever.ivfpq_retriever import IVFPQRetriever

logger = logging.getLogger(__name__)

app = FastAPI()
retriever = None


class QueryRequest(BaseModel):
    queries: Union[List[str], List[List[float]]]
    topk: Optional[int] = None
    mode: Optional[str] = 'text'


@app.get("/retriever_info")
def get_retriever_info():
    return {"retriever_tag": retriever.get_retriever_tag()}


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    try:
        if request.mode == 'text':
            out = retriever.retrieve_with_text(request.queries, top_k=request.topk)
        elif request.mode == 'embedding':
            out = retriever.retrieve_with_embeddings(np.array(request.queries, dtype=np.float32),
                                                     top_k=request.topk)
        else:
            return {'results': None, 'success': False,
                    'error': f"Unsupported retrieval mode: {request.mode}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {'results': out, 'success': True}


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config: DictConfig):
    global retriever
    retriever = IVFPQRetriever(config)
    if config.ivfpq.load_encoder:
        retriever.set_encoder(Encoder.load(config, trainable=False, use_lora=False))
        logger.info("Query encoder loaded (text mode enabled)")
    else:
        logger.info("No query encoder (ivfpq.load_encoder=false): embedding mode only")

    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        from fastapi.testclient import TestClient
        client = TestClient(app)
        print(client.get("/retriever_info").json())
        if retriever.get_encoder() is not None:
            print(client.post("/retrieve", json={"queries": ["What is Latent Search?"],
                                                 "topk": 3, "mode": "text"}).json())

    uvicorn.run(app, host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
