import logging
from flask import Flask, request, jsonify
from embeddings import EmbeddingService

# Patch for transformers to avoid torch.fx import error in environments without PyTorch FX support
import transformers.utils.import_utils as _import_utils
if not hasattr(_import_utils, "is_torch_fx_available"):
    _import_utils.is_torch_fx_available = lambda: False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize embedding service
embedding_service = EmbeddingService()


def sparse_to_dict(sparse_matrix):
    """Convert a scipy sparse matrix (single row) to a dict of {index: value}.

    The BGEM3 model returns sparse vectors as scipy CSR matrices.
    We convert to COO format for easy index/value extraction.
    """
    coo = sparse_matrix.tocoo()
    return {str(int(col)): float(val) for col, val in zip(coo.col, coo.data)}


@app.route("/embed", methods=["POST"])
def embed():
    data = request.json
    texts = data.get("texts")
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Missing or invalid 'texts' field. Expected a list of strings."}), 400

    try:
        results = embedding_service.embed_texts(texts)
        embeddings = []
        for result in results:
            embeddings.append({
                "dense": result["dense"].tolist(),
                "sparse": sparse_to_dict(result["sparse"]),
            })
        return jsonify({"embeddings": embeddings})
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


def main():
    """Main function - run as Flask API server"""
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
