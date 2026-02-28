import streamlit as st
import numpy as np
from PIL import Image
from niqe import niqe
import pandas as pd
import io

# Set page configuration with a cleaner, professional title
st.set_page_config(
    page_title="NIQE Analysis Dashboard",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional "SaaS Dashboard" look
st.markdown("""
    <style>
        /* Import clean font features */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Main app background is handled by config.toml, but we refine spacing here */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* CARD STYLE: White background, subtle border, shadow */
        .css-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
        }

        /* Metric styling enhancement */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            text-align: center;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 0.875rem;
            color: #64748b;
            font-weight: 500;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            color: #0f172a;
            font-weight: 700;
        }

        /* Typography overrides */
        h1 {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: #0f172a;
        }
        
        h2, h3 {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: #334155;
        }
        
        p, li, .stMarkdown {
            font-family: 'Inter', sans-serif;
            color: #475569;
            line-height: 1.6;
        }

        /* Expander and other widget styling */
        .streamlit-expanderHeader {
            font-weight: 500;
            color: #334155;
        }
        
        /* Custom Helper Classes */
        .sub-header {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)


# Sidebar for controls
with st.sidebar:
    st.header("Configuration")
    st.markdown("**1. Upload Images**")
    st.write("Select the images you want to analyze. You can select multiple files at once.")
    
    uploaded_files = st.file_uploader(
        "Upload Images", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'], 
        accept_multiple_files=True,
        help="Drag and drop images here or click 'Browse files' to select. Supported formats: PNG, JPG, BMP, TIFF."
    )
    
    st.markdown("---")
    
    with st.expander("ℹ️ About the App"):
        st.markdown("""
        **What is this?**
        This dashboard calculates the **NIQE** score for your images.
        
        **What is NIQE?**
        *Naturalness Image Quality Evaluator.*  
        It is a "no-reference" metric, meaning it judges quality without needing a perfect original image to compare against.
        
        **How to read the score?**
        - **Lower Score** = Better, more natural quality.
        - **Higher Score** = Worse, more distorted.
        
        **Buttons & Features:**
        - **Upload**: Select your image files.
        - **Data Table**: View exact numbers and sort them.
        - **Visual Grid**: See images with their scores and a color-coded quality bar.
        - **Download CSV**: Save the analysis results to a file for Excel/Sheets.
        """)

    st.markdown("""
        <div style='font-size: 12px; color: #64748b; margin-top: 20px;'>
            v1.1.0 | NIQE Analysis Tool
        </div>
    """, unsafe_allow_html=True)

# Main Dashboard Area
st.title("Image Quality Analysis Dashboard")
st.markdown("""
    Welcome! This tool analyzes the **perceptual quality** of your images using the NIQE algorithm.
    
    **Instructions:**
    1.  Open the sidebar on the left (if closed).
    2.  Upload one or more images.
    3.  Wait for the processing to finish.
    4.  Review the statistics and individual scores below.
""")

if uploaded_files:
    # --- Processing ---
    results_data = []
    
    # Progress indication
    progress_text = "Processing image batch..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Load and process
            img = Image.open(uploaded_file).convert('RGB')
            gray_img = np.array(img.convert('LA'))[:, :, 0]
            score = niqe(gray_img)
            
            # Store data
            results_data.append({
                "Filename": uploaded_file.name,
                "NIQE Score": float(score),
                "Resolution": f"{img.size[0]}x{img.size[1]}",
                "object": img 
            })
            
            # Update progress
            my_bar.progress((i + 1) / len(uploaded_files), text=f"Processing {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    my_bar.empty()

    if results_data:
        df = pd.DataFrame(results_data)
        
        # --- Summary Statistics Section ---
        st.subheader("Batch Summary")
        
        mean_score = df["NIQE Score"].mean()
        median_score = df["NIQE Score"].median()
        std_dev = df["NIQE Score"].std()
        best_score = df["NIQE Score"].min()
        worst_score = df["NIQE Score"].max()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Images Processed", len(df))
        col2.metric("Mean Score", f"{mean_score:.3f}")
        col3.metric("Median Score", f"{median_score:.3f}")
        col4.metric("Best (Min)", f"{best_score:.3f}")
        col5.metric("Std Dev", f"{std_dev:.3f}")
        
        st.markdown("---")

        # --- Detailed Analysis Tabs ---
        tab1, tab2 = st.tabs(["📄 Data Table", "🖼️ Visual Grid"])
        
        with tab1:
            st.markdown("### Detailed Results")
            
            # Use Streamlit's native ProgressColumn to visualize the score
            # Lower NIQE is better, but ProgressColumn fills from 0 to Max.
            # We will just show the value as a number formatted nicely, 
            # and maybe a 'Quality Bar' if we normalize it, but simple is better for stability.
            
            st.dataframe(
                df[["Filename", "NIQE Score", "Resolution"]],
                use_container_width=True,
                height=400,
                column_config={
                    "NIQE Score": st.column_config.ProgressColumn(
                        "NIQE Score (Lower is Better)",
                        help="The NIQE score for the image. Lower values indicate better naturalness.",
                        format="%.4f",
                        min_value=0,
                        max_value=max(df["NIQE Score"].max() + 5, 20), # Dynamic max
                    ),
                    "Resolution": st.column_config.TextColumn("Resolution")
                }
            ) 
            
            # Download CSV button
            csv = df[["Filename", "NIQE Score", "Resolution"]].to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Report (CSV)",
                csv,
                "niqe_report.csv",
                "text/csv",
                key='download-csv',
                help="Click to download the table above as a CSV file compatible with Excel."
            )

        with tab2:
            st.markdown("### Visual Inspection")
            # Grid layout for images
            cols = st.columns(3)
            for idx, row in df.iterrows():
                with cols[idx % 3]:
                    with st.container():
                        st.image(row["object"], use_column_width=True)
                        st.markdown(f"**{row['Filename']}**")
                        st.markdown(f"<span style='color: #64748b; font-size: 0.9em;'>Score:</span> **{row['NIQE Score']:.4f}**", unsafe_allow_html=True)
                        
                        # Add a visual bar relative to the batch range
                        normalized_score = (row['NIQE Score'] - best_score) / (worst_score - best_score + 1e-6)
                        # Color from Green (best/low) to Red (worst/high)
                        color_hex = "#10b981" if normalized_score < 0.5 else "#ef4444"
                        st.markdown(f"""
                            <div style="width:100%; height:4px; background-color: #e2e8f0; border-radius:2px; margin-top:4px;">
                                <div style="width:{normalized_score*100}%; height:100%; background-color: {color_hex}; border-radius:2px;"></div>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)

else:
    # Empty state
    st.info("Upload images in the sidebar to begin analysis.")


