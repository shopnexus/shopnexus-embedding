import os
import logging
import torch
torch.set_num_threads(os.cpu_count())
from flask import Flask, request, jsonify
# from pymilvus.model.hybrid import BGEM3EmbeddingFunction as EmbeddingFunction
from pymilvus.model.hybrid import MGTEEmbeddingFunction as EmbeddingFunction

import transformers.utils.import_utils as _import_utils
if not hasattr(_import_utils, "is_torch_fx_available"):
    _import_utils.is_torch_fx_available = lambda: False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
ef = EmbeddingFunction(use_fp16=False, device="cpu")


def sparse_to_dict(sparse_matrix):
    coo = sparse_matrix.tocoo()
    return {str(int(col)): float(val) for col, val in zip(coo.col, coo.data)}


@app.route("/embed", methods=["POST"])
def embed():
    data = request.json
    texts = data.get("texts")
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Missing or invalid 'texts' field. Expected a list of strings."}), 400


    try:
        result = ef.encode_documents(texts)
        embeddings = []
        for i in range(len(texts)):
            embeddings.append({
                "dense": result["dense"][i].tolist(),
                "sparse": sparse_to_dict(result["sparse"][[i]]),
            })
        return jsonify({"embeddings": embeddings})
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
