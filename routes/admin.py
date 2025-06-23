"""
routes/admin.py - Admin routes for JazzBot

This module handles administrative endpoints like file uploads and system management.
"""

import os
from datetime import datetime
from flask import Blueprint, request, Response

from config import Config
from utils.logger import logger
from routes.chat import get_retriever_service

# Initialize config
config = Config()

# Create blueprint
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


@admin_bp.route("/upload", methods=["POST"])
def admin_upload():
    """Enhanced admin upload endpoint with Roman Urdu support"""
    try:
        if "file" not in request.files:
            return Response("No file provided", status=400)

        file = request.files["file"]
        category = request.form.get("category", "unknown")
        file_type = request.form.get("file_type", "unknown")

        if file.filename == "":
            return Response("No file selected", status=400)

        # Save file
        category_dir = os.path.join(config.DATA_DIR, category)
        os.makedirs(category_dir, exist_ok=True)

        filename = (
            f"{category}_{file_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        file_path = os.path.join(category_dir, filename)
        file.save(file_path)

        logger.info(f"File uploaded: {file_path}")

        # Special handling for Roman Urdu category
        if category.lower() == "roman urdu":
            logger.info("Processing Roman Urdu content")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Add preprocessing for Roman Urdu if required (e.g., normalization)
            content = _preprocess_roman_urdu(content)

            # Save the preprocessed file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("Roman Urdu preprocessing complete")

        # Rebuild index
        retriever_service = get_retriever_service()
        success = retriever_service.rebuild_index()

        success_msg = (
            "File uploaded and index rebuilt successfully"
            if success
            else "File uploaded but index rebuild failed"
        )
        logger.info(success_msg)

        return Response(success_msg, status=200)

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return Response(f"Upload failed: {str(e)}", status=500)


@admin_bp.route("/rebuild_index", methods=["POST"])
def rebuild_index():
    """Force rebuild the FAISS index"""
    try:
        retriever_service = get_retriever_service()
        success = retriever_service.rebuild_index()
        
        if success:
            return Response("Index rebuilt successfully", status=200)
        else:
            return Response("Index rebuild failed", status=500)
    except Exception as e:
        logger.error(f"Index rebuild error: {e}")
        return Response(f"Index rebuild failed: {str(e)}", status=500)


@admin_bp.route("/status", methods=["GET"])
def admin_status():
    """Get detailed system status"""
    try:
        retriever_service = get_retriever_service()
        retriever_status = retriever_service.get_status()
        
        # Check data directory
        data_dir_exists = os.path.exists(config.DATA_DIR)
        data_files_count = 0
        
        if data_dir_exists:
            for root, dirs, files in os.walk(config.DATA_DIR):
                data_files_count += len([f for f in files if f.endswith('.txt')])
        
        status = {
            "system": "online",
            "data_directory": {
                "exists": data_dir_exists,
                "path": config.DATA_DIR,
                "file_count": data_files_count
            },
            "retriever": retriever_status,
            "config": {
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "search_k": config.SEARCH_K,
                "memory_size": config.MEMORY_SIZE,
                "embedding_model": config.EMBEDDING_MODEL,
                "llm_model": config.LLM_MODEL
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return status, 200
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return {"error": str(e)}, 500


@admin_bp.route("/list_files", methods=["GET"])
def list_files():
    """List all uploaded files by category"""
    try:
        if not os.path.exists(config.DATA_DIR):
            return {"categories": {}, "total_files": 0}, 200
        
        files_by_category = {}
        total_files = 0
        
        for category in os.listdir(config.DATA_DIR):
            category_path = os.path.join(config.DATA_DIR, category)
            if not os.path.isdir(category_path):
                continue
            
            files = []
            for file in os.listdir(category_path):
                if file.endswith('.txt'):
                    file_path = os.path.join(category_path, file)
                    file_stats = os.stat(file_path)
                    files.append({
                        "name": file,
                        "path": file_path,
                        "size": file_stats.st_size,
                        "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    })
                    total_files += 1
            
            if files:
                files_by_category[category] = files
        
        return {
            "categories": files_by_category,
            "total_files": total_files
        }, 200
    
    except Exception as e:
        logger.error(f"List files error: {e}")
        return {"error": str(e)}, 500


@admin_bp.route("/delete_file", methods=["DELETE"])
def delete_file():
    """Delete a specific file"""
    try:
        data = request.get_json()
        file_path = data.get("file_path")
        
        if not file_path:
            return Response("File path not provided", status=400)
        
        # Security check - ensure file is in data directory
        if not file_path.startswith(config.DATA_DIR):
            return Response("Invalid file path", status=400)
        
        if not os.path.exists(file_path):
            return Response("File not found", status=404)
        
        os.remove(file_path)
        logger.info(f"File deleted: {file_path}")
        
        # Rebuild index after deletion
        retriever_service = get_retriever_service()
        success = retriever_service.rebuild_index()
        
        message = (
            "File deleted and index rebuilt successfully"
            if success
            else "File deleted but index rebuild failed"
        )
        
        return Response(message, status=200)
    
    except Exception as e:
        logger.error(f"Delete file error: {e}")
        return Response(f"Delete failed: {str(e)}", status=500)


def _preprocess_roman_urdu(content: str) -> str:
    """
    Normalize Roman Urdu text by handling common inconsistencies or typos.
    Placeholder for further text cleaning or processing logic.
    """
    # Example normalization steps
    replacements = {
        "krdo": "kar do",
        "acha": "acha",
        "theek": "thik",
        "han": "haan",
        "nai": "nahi",
        # Add more mappings as needed
    }

    for key, value in replacements.items():
        content = content.replace(key, value)
    return content