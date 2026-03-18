# VisionEndee: Multimodal Visual Search Engine

VisionEndee is a high-performance AI-powered search engine that demonstrates the capabilities of the **Endee** vector database. It enables users to search through an image collection using either natural language descriptions ("Text-to-Image") or by uploading similar images ("Image-to-Image").

## 🚀 Overview

The project leverages **Endee** as its core vector retrieval engine and uses **CLIP (Contrastive Language-Image Pre-training)** embeddings to bridge the gap between text and visuals. This allows for semantic retrieval that goes beyond simple keyword matching.

### Key Features
- **Semantic Image Search**: Find images by describing their content.
- **Visual Similarity Search**: Upload an image to find visually similar items in the database.
- **High Performance**: Powered by Endee's HNSW-based vector indexing for sub-millisecond retrieval.
- **Fully Containerized**: Easy deployment using Docker Compose.

## 🏗️ System Design

The system is composed of two main services:

1.  **Endee Engine (C++ Vector DB)**: 
    - Handles high-speed vector storage and similarity search.
    - Uses HNSW (Hierarchical Navigable Small World) for dense vector indexing.
    - Provides a REST API for indexing and querying.

2.  **Vision AI App (Python/Streamlit)**:
    - **Frontend**: A clean, interactive dashboard built with Streamlit.
    - **Embedding Module**: Uses `sentence-transformers` with the `clip-ViT-B-32` model to generate 512-dimensional multimodal vectors.
    - **Integration Logic**: Acts as a client to the Endee REST API, managing the indexing pipeline and search queries.

## 💾 Use of Endee

Endee is the backbone of this project, providing:
- **Index Management**: Creating and configuring high-dimensional vector indices.
- **Vector Storage**: Efficiently storing 512-dimension CLIP embeddings.
- **Metadata Payloads**: Storing file paths and labels alongside vectors for rich search results.
- **Similarity Search**: Performing ultra-fast KNN searches using cosine similarity.

## 🛠️ Setup Instructions

### Prerequisites
- Docker & Docker Compose
- (Optional) Python 3.9+ (if running the app locally without Docker)

### Running with Docker (Recommended)

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/maazahmad7739/endee-Maaz.git
    cd endee-Maaz
    ```

2.  **Start the Services**:
    ```bash
    docker-compose up --build
    ```

3.  **Access the Dashboard**:
    - Open your browser and navigate to `http://localhost:8502`.

### Initial Data Indexing
- The project comes with a few sample images.
3.  Open `http://localhost:8502` to use the search engine.
4.  Use the "Index Images" button in the sidebar to load the samples into Endee.
 Endee database.

## 📊 Performance & Optimization
Endee is optimized for SIMD instructions (AVX2/AVX512), ensuring that even as your image collection grows to millions of items, search results remain near-instantaneous.

---
Built with ❤️ using [Endee](https://github.com/endee-io/endee)
