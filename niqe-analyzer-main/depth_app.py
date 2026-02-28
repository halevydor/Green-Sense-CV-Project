import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from depth_consistency import DepthConsistencyEvaluator

st.set_page_config(page_title="Depth Consistency Analysis", layout="wide")

@st.cache_resource
def get_evaluator(version=1):
    # Version argument added to force cache reload if needed
    return DepthConsistencyEvaluator()

st.title("Depth Consistency Analysis (MiDaS)")
st.markdown("""
This tool evaluates the **Depth Consistency** of images using:
1.  **Smoothness (30%)**: Stability of depth gradients.
2.  **Edge Alignment (40%)**: How well depth boundaries match object boundaries.
3.  **Distribution (30%)**: Richness of depth information.
""")

evaluator = get_evaluator(version=2) # Increment version to bust cache

uploaded_files = st.file_uploader(
    "Upload Images", 
    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'], 
    accept_multiple_files=True
)

if uploaded_files:
    results = []
    
    # Progress bar
    progress_bar = st.progress(0, text="Starting analysis...")
    
    tabs = st.tabs(["Detailed View", "Batch Summary"])
    
    with tabs[0]:
        st.subheader("Individual Image Analysis")
        for i, file in enumerate(uploaded_files):
            progress_bar.progress((i + 1) / len(uploaded_files), text=f"Processing {file.name}...")
            
            try:
                # Load
                img_pil, img_np = evaluator.load_image(file)
                
                # Estimate
                depth_map = evaluator.estimate_depth(img_pil)
                if depth_map is None:
                    st.error(f"Failed to estimate depth for {file.name}")
                    continue
                    
                # Calculate Scores
                final_score, metrics = evaluator.calculate_consistency_score(img_np, depth_map)
                
                results.append({
                    "Filename": file.name,
                    **metrics
                })
                
                with st.expander(f"{file.name} - Score: {final_score:.1f}/100"):
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        st.image(img_pil, caption="Original Image", use_column_width=True)
                    
                    with col2:
                        # Colorize depth for display
                        depth_colored = plt.get_cmap('inferno')(depth_map)[:, :, :3]
                        st.image(depth_colored, caption="Depth Map (MiDaS)", use_column_width=True)
                        
                    with col3:
                        st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "Score"}), use_container_width=True)
            
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
                
    progress_bar.empty()
    
    with tabs[1]:
        if results:
            df = pd.DataFrame(results)
            
            # Summary Metrics
            st.subheader("Batch Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Total Score", f"{df['Total Score'].mean():.1f}")
            col2.metric("Mean Smoothness", f"{df['Smoothness Score'].mean():.1f}")
            col3.metric("Mean Edge Align", f"{df['Edge Alignment Score'].mean():.1f}")
            col4.metric("Mean Distribution", f"{df['Distribution Score'].mean():.1f}")
            
            st.markdown("---")
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "Total Score": st.column_config.ProgressColumn(
                        "Total Score", format="%.1f", min_value=0, max_value=100
                    ),
                    "Smoothness Score": st.column_config.NumberColumn(format="%.1f"),
                    "Edge Alignment Score": st.column_config.NumberColumn(format="%.1f"),
                    "Distribution Score": st.column_config.NumberColumn(format="%.1f"),
                }
            )
            
            # CSV Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Report (CSV)",
                csv,
                "depth_consistency_report.csv",
                "text/csv",
                key='download-csv'
            )
