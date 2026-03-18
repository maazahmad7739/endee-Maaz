import streamlit as st
import os
from engine import initialize_system, ENDEE_INDEX
from PIL import Image
import numpy as np

st.set_page_config(page_title="VisionEndee - Multimodal Search", layout="wide")

st.title("🖼️ VisionEndee")
st.markdown("### High-Performance Multimodal Visual Search powered by **Endee**")

client, engine = initialize_system()

# Sidebar for indexing
with st.sidebar:
    st.header("Settings")
    image_dir = st.text_input("Local Image Directory", value="./data/sample_images")
    if st.button("Index Images"):
        if not os.path.exists(image_dir):
            st.error(f"Directory {image_dir} not found!")
        else:
            files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            progress = st.progress(0)
            status = st.empty()
            
            for i, f in enumerate(files):
                img_path = os.path.join(image_dir, f)
                status.text(f"Processing {f}...")
                vec = engine.get_image_embedding(img_path)
                client.insert_vector(ENDEE_INDEX, i, vec, {"path": img_path, "filename": f})
                progress.progress((i + 1) / len(files))
            
            st.success(f"Indexed {len(files)} images!")

# Main search interface
tab1, tab2 = st.tabs(["🔍 Text-to-Image", "📸 Image-to-Image"])

with tab1:
    query = st.text_input("Enter a description to search (e.g., 'a peaceful lake at sunset')")
    if query:
        with st.spinner("Searching..."):
            query_vec = engine.get_text_embedding(query)
            results = client.search(ENDEE_INDEX, query_vec)
            
            if "results" in results:
                cols = st.columns(3)
                for idx, res in enumerate(results["results"]):
                    with cols[idx % 3]:
                        path = res["payload"].get("path")
                        if os.path.exists(path):
                            st.image(path, caption=f"Score: {res['score']:.4f}")
                        else:
                            st.warning(f"Image not found at {path}")
            else:
                st.info("No results found or index is empty.")

with tab2:
    uploaded_file = st.file_uploader("Upload an image to find similar ones", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Query Image", width=300)
        
        if st.button("Find Similar"):
            with st.spinner("Analyzing image..."):
                query_vec = engine.get_image_embedding(uploaded_file)
                results = client.search(ENDEE_INDEX, query_vec)
                
                if "results" in results:
                    cols = st.columns(3)
                    for idx, res in enumerate(results["results"]):
                        with cols[idx % 3]:
                            path = res["payload"].get("path")
                            if os.path.exists(path):
                                st.image(path, caption=f"Score: {res['score']:.4f}")
                else:
                    st.info("No results found.")
