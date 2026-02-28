"""
Greenspace Quality Classifier - Interactive App
Simple Streamlit dashboard for single-image classification.
"""
import streamlit as st
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import torch
import pickle
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from scene_features import SceneFeatureExtractor
from vegetation_detector import VegetationDetector, mask_road_signs
from vegetation_features import VegetationFeatureExtractor, ColorTextureAnalyzer
from dataset import get_clip_preprocess

# Page Config
st.set_page_config(
    page_title="Greenspace Quality Classifier",
    page_icon="🌳",
    layout="centered"
)

# Title and Style
st.title("🌳 Greenspace Quality Analysis")
st.markdown("""
<style>
    /* --- MODERN ORGANIC ZEN THEME --- */
    :root {
        /* Palette: Earth, Stone & Sage */
        --bg-app: #111311;         /* Deep Organic Charcoal */
        --bg-panel: #1A1D1A;       /* Dark Moss/Stone */
        --bg-panel-hover: #222622;
        
        --text-heading: #D1FAE5;   /* Pastel Sage Green (Requested) */
        --text-body: #E2E8F0;      /* Soft White/Gray */
        --text-muted: #94A3B8;     /* Stone Gray */
        
        --accent-sage: #6EE7B7;    /* Muted Seafoam */
        --accent-leaf: #10B981;    /* Natural Emerald */
        
        --border-subtle: #2C332C;  /* Dark organic border */
    }
    
    /* Global RESET & Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
        color: var(--text-body);
        -webkit-font-smoothing: antialiased;
    }
    
    /* 1. APP BACKGROUND - Matte & Grounded */
    .stApp {
        background-color: var(--bg-app);
        /* Very subtle grain texture for organic feel (optional, simulated here with flat color for calm) */
    }
    
    /* 2. CHASSIS / MAIN CONTAINER - "The Tablet" Look */
    .main .block-container {
        background-color: var(--bg-panel);
        padding: 4rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24); /* Subtle material shadow */
        max-width: 1000px;
        margin-top: 2rem;
        border: 1px solid var(--border-subtle);
    }
    
    /* 3. HEADINGS - Pastel Green Priority */
    h1 {
        color: var(--text-heading) !important;
        font-weight: 500 !important; /* Lighter weight for elegance */
        font-size: 2.8rem !important;
        letter-spacing: -0.03em !important;
        background: none !important;
        -webkit-text-fill-color: initial !important;
    }
    
    h2, h3 {
        color: var(--text-heading) !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em;
    }
    
    /* 4. METRICS - Clean Data */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #F0FDF4 !important; /* Almost white mint */
        font-weight: 400; /* Sophisticated thin look */
        background: none;
        -webkit-text-fill-color: initial;
    }
    
    div[data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    div[data-testid="metric-container"] {
        background-color: #222622; /* Slightly elevated surface */
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: none;
    }
    
    /* 5. BUTTONS - Matter & Functional */
    .stButton > button {
        background-color: #2C332C !important;
        color: var(--text-heading) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 6px !important; /* Professional squared-off corners */
        padding: 0.6rem 1.5rem !important;
        font-weight: 500 !important;
        transition: background-color 0.2s ease !important;
        box-shadow: none !important;
    }
    
    .stButton > button:hover {
        background-color: #384038 !important;
        border-color: var(--accent-sage) !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* Primary Action - The "Go" Button */
    .stButton > button[kind="primary"] {
        background-color: var(--accent-leaf) !important;
        color: #064E3B !important; /* Dark green text on bright button for contrast */
        border: none !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: var(--accent-sage) !important;
    }
    
    /* 6. UPLOADER - Architectural Dotted Line */
    div[data-testid="stFileUploader"] {
        background-color: transparent;
        border: 1px dashed var(--border-subtle) !important;
        border-radius: 8px;
        padding: 2rem;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: var(--text-heading) !important;
        background-color: rgba(209, 250, 229, 0.02);
    }
    
    /* 7. TABS - Understated */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--border-subtle);
        gap: 2.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: var(--text-muted);
        font-weight: 500;
        padding: 1rem 0;
        font-family: inherit;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--text-heading) !important;
        border-bottom: 2px solid var(--text-heading);
    }
    
    /* 8. VISUALS */
    img {
        border-radius: 8px;
        border: none;
        opacity: 0.95; /* Slight blend with dark theme */
    }
    
    .stPlotlyChart, .stVegaLiteChart, .stDataFrame {
        background-color: #222622 !important;
        border-radius: 8px;
        border: 1px solid var(--border-subtle);
    }
    
    /* Sidebar - Seamless */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-app);
        border-right: 1px solid var(--border-subtle);
    }
    
    /* Divider */
    hr {
        border-color: var(--border-subtle);
        margin: 3rem 0;
        opacity: 0.5;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--text-heading) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_feature_extractors_v5():
    """Load heavy feature extraction models (CLIP + DINOv2)."""
    device = "cpu"
    config = Config()
    
    with st.spinner("Loading AI models (CLIP + DINOv2)..."):
        # Load with DINO enabled
        scene_extractor = SceneFeatureExtractor(
            device=device,
            scene_prompts=config.scene_prompts,
            use_dino=config.use_dino,
            dino_model_name=config.dino_model_name,
            dino_image_size=config.dino_image_size
        )
        veg_detector = VegetationDetector(device=device)
        veg_extractor = VegetationFeatureExtractor(scene_extractor, config.clip_image_size)
        color_analyzer = ColorTextureAnalyzer()
        preprocess = get_clip_preprocess(config.clip_image_size)
        
        # Get DINO preprocess if enabled
        dino_preprocess = None
        if config.use_dino and scene_extractor.dino_extractor:
            dino_preprocess = scene_extractor.dino_extractor.preprocess_image
            
    return scene_extractor, veg_detector, veg_extractor, color_analyzer, preprocess, dino_preprocess

def load_classifier():
    """Load the trained classifier (tries DINO model first, falls back to CLIP-only)."""
    # Try DINO+CLIP model first
    classifier_path = os.path.join(os.path.dirname(__file__), 'models', 'best_classifier_dino.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler_dino.pkl')
    
    if os.path.exists(classifier_path):
        with open(classifier_path, 'rb') as f:
            classifier = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return classifier, scaler
    
    # Fallback to CLIP-only model
    classifier_path = os.path.join(os.path.dirname(__file__), 'models', 'best_classifier.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
    
    if os.path.exists(classifier_path):
        with open(classifier_path, 'rb') as f:
            classifier = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        st.warning("⚠️ Using CLIP-only model (DINO model not found)")
        return classifier, scaler
        
    return None, None

# Load models
feature_models = load_feature_extractors_v5()
classifier_models = load_classifier()

if feature_models and classifier_models[0]:
    scene_extractor, veg_detector, veg_extractor, color_analyzer, preprocess, dino_preprocess = feature_models
    classifier, scaler = classifier_models
    
    # Tabs
    tab_single, tab_batch = st.tabs(["Single Image Analysis", "Batch Evaluation"])
    
    # === TAB 1: SINGLE IMAGE ===
    with tab_single:
        st.write("Upload an image of a park or greenspace to analyze its quality (Healthy / Dried / Contaminated).")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"], key="single")
        
        if uploaded_file is not None:
            # Display Image
# Image optimization
            image = Image.open(uploaded_file).convert('RGB')
            
            # Resize if too large to prevent OOM
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                
            st.image(image, caption='Uploaded Image') 
            
            if st.button("Analyze Quality", type="primary", key="btn_single"):
                with st.spinner("Analyzing vegetation and scene features..."):
                    # 1. Feature Extraction
                    # Clean image: remove road signs before scene analysis
                    clean_image = mask_road_signs(image)
                    
                    image_tensor = preprocess(clean_image).unsqueeze(0)
                    
                    # CLIP Scene (uses cleaned image)
                    scene_emb, prompt_scores = scene_extractor.extract_scene_features(image_tensor)
                    
                    # DINO Scene (uses cleaned image)
                    if dino_preprocess and scene_extractor.use_dino:
                        dino_tensor = dino_preprocess(clean_image).unsqueeze(0)
                        dino_scene_emb = scene_extractor.extract_dino_scene_features(dino_tensor)
                    else:
                        dino_scene_emb = None
                    
                    # Vegetation Detection
                    detections = veg_detector.detect_vegetation(image)
                    boxes = [d['box'] for d in detections]
                    
                    # Draw boxes for visualization
                    if boxes:
                        img_draw = image.copy()
                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(img_draw)
                        for box in boxes:
                            draw.rectangle(box, outline='lime', width=5)
                        st.image(img_draw, caption=f"Detected Vegetation ({len(boxes)} regions)") 
                    
                    # CLIP Veg Features
                    _, veg_emb, _ = veg_extractor.extract_crop_embeddings(image, boxes)
                    if veg_emb is None:
                        veg_emb = torch.zeros_like(scene_emb)
                    
                    # DINO Veg Features (if enabled)
                    if scene_extractor.use_dino:
                        _, dino_veg_emb = veg_extractor.extract_dino_crop_embeddings(image, boxes)
                        if dino_veg_emb is None:
                            dino_veg_emb = torch.zeros(scene_extractor.dino_extractor.embedding_dim)
                    else:
                        dino_veg_emb = None
                    
                    # Color stats
                    color_stats = color_analyzer.aggregate_stats(image, boxes, None)
                    
                    # Prepare feature vector
                    scene_vec = scene_emb.numpy().flatten()
                    veg_vec = veg_emb.numpy().flatten()
                    color_vec = np.array([
                        color_stats.get('mean_green_ratio', 0),
                        color_stats.get('mean_edge_density', 0),
                        color_stats.get('vegetation_coverage', 0)
                    ])
                    
                    # Build feature vector (matches training format)
                    feature_parts = [scene_vec, veg_vec]
                    
                    # DINO features - always add if use_dino=True to match scaler dimensions (1795)
                    if scene_extractor.use_dino:
                        # DINO Scene features
                        if dino_scene_emb is not None:
                            dino_scene_vec = dino_scene_emb.cpu().numpy().flatten()
                        else:
                            dino_scene_vec = np.zeros(scene_extractor.dino_extractor.embedding_dim, dtype=np.float32)
                        feature_parts.append(dino_scene_vec)
                        
                        # DINO Vegetation features
                        if dino_veg_emb is not None:
                            dino_veg_vec = dino_veg_emb.cpu().numpy().flatten() if torch.is_tensor(dino_veg_emb) else dino_veg_emb.flatten()
                        else:
                            dino_veg_vec = np.zeros(scene_extractor.dino_extractor.embedding_dim, dtype=np.float32)
                        feature_parts.append(dino_veg_vec)
                    
                    feature_parts.append(color_vec)
                    features = np.concatenate(feature_parts).reshape(1, -1)
                    
                    # 2. Classification with Ensemble Voting
                    features_scaled = scaler.transform(features)
                    rf_prediction = classifier.predict(features_scaled)[0]
                    rf_probs = classifier.predict_proba(features_scaled)[0]
                    
                    # Handle case where classifier wasn't trained on all 3 classes
                    if len(rf_probs) < 3:
                        full_probs = np.zeros(3)
                        known_classes = classifier.classes_
                        for i, cls in enumerate(known_classes):
                            full_probs[cls] = rf_probs[i]
                        rf_probs = full_probs
                    
                    # CLIP Prompt-based voting
                    all_prompts = Config().scene_prompts
                    prompt_class_map = []
                    
                    # Robust index-based mapping (matches Config ranges)
                    # 0-9 = Healthy (10 prompts)
                    # 10-19 = Dried (10 prompts)
                    # 20+ = Contaminated
                    for i, _ in enumerate(all_prompts):
                        if i <= 9:
                            prompt_class_map.append(0) # Healthy
                        elif i <= 19:
                            prompt_class_map.append(1) # Dried
                        else:
                            prompt_class_map.append(2) # Contaminated
                    
                    scores = prompt_scores[0].cpu().numpy()
                    clip_class_scores = np.zeros(3)
                    for i, score in enumerate(scores):
                        clip_class_scores[prompt_class_map[i]] += score
                    clip_class_scores /= clip_class_scores.sum() if clip_class_scores.sum() > 0 else 1
                    
                    # Check for consensus in top 3 matches
                    top_3_indices = np.argsort(scores)[::-1][:3]
                    top_3_classes = [prompt_class_map[i] for i in top_3_indices]
                    
                    # 3/3 = full consensus, 2/3 = majority agreement
                    full_consensus = len(set(top_3_classes)) == 1
                    consensus_class = top_3_classes[0] if full_consensus else -1
                    
                    # Check for 2/3 majority
                    from collections import Counter
                    class_counts = Counter(top_3_classes)
                    majority_class, majority_count = class_counts.most_common(1)[0]
                    has_majority = (majority_count >= 2) and not full_consensus
                    
                    # Ensemble: 60% RF + 40% CLIP
                    ensemble_probs = 0.6 * rf_probs + 0.4 * clip_class_scores
                    ensemble_prediction = np.argmax(ensemble_probs)
                    
                    # Boost for #1 top prompt class (+5%)
                    top_1_class = prompt_class_map[top_3_indices[0]]
                    ensemble_probs[top_1_class] += 0.05
                    ensemble_probs = ensemble_probs / ensemble_probs.sum()
                    ensemble_prediction = np.argmax(ensemble_probs)
                    
                    # Decision Hierarchy:
                    # 1. CLIP Full Consensus — 3/3 top prompts agree (strongest signal)
                    # 2. CLIP Majority — 2/3 top prompts agree (+7% boost)
                    # 3. Ensemble (if RF uncertain)
                    # 4. Random Forest (default)
                    
                    if full_consensus:
                        prediction = consensus_class
                        probs = np.zeros(3)
                        probs[prediction] = 0.98
                        others = [c for c in [0,1,2] if c != prediction]
                        for c in others: probs[c] = 0.01
                        prediction_source = "CLIP Consensus (3/3 Prompts)"
                    elif has_majority:
                        # Boost the majority class by 7% in ensemble
                        boosted_probs = ensemble_probs.copy()
                        boosted_probs[majority_class] += 0.07
                        boosted_probs = boosted_probs / boosted_probs.sum()
                        prediction = np.argmax(boosted_probs)
                        probs = boosted_probs
                        prediction_source = "CLIP Majority (2/3 Prompts)"
                    elif rf_probs[rf_prediction] < 0.65:
                        prediction = ensemble_prediction
                        probs = ensemble_probs
                        prediction_source = "Ensemble (RF+CLIP)"
                    else:
                        prediction = rf_prediction
                        probs = rf_probs
                        prediction_source = "Random Forest"
                    
                    confidence = probs[prediction]
                    class_names = ['Healthy', 'Dried', 'Contaminated']
                    class_name = class_names[prediction]
                    
                    # 3. Results UI
                    st.markdown("---")
                    st.subheader("Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Modern Organic Styles - Matte & Deep
                    color_styles = {
                        'Healthy': {
                            'bg': '#14532D',      # Deep Forest Green
                            'text': '#D1FAE5',
                            'emoji': '🌿',
                            'border': '#166534'
                        },
                        'Dried': {
                            'bg': '#713F12',      # Deep Earth/Ochre
                            'text': '#FEF3C7',
                            'emoji': '🍂',
                            'border': '#854D0E'
                        },
                        'Contaminated': {
                            'bg': '#7F1D1D',      # Deep Red Clay
                            'text': '#FEE2E2',
                            'emoji': '⚠️',
                            'border': '#991B1B'
                        }
                    }
                    style = color_styles[class_name]
                    
                    with col1:
                        st.metric("Prediction", class_name, delta_color="off")
                    with col2:
                        st.metric("Confidence", f"{confidence:.1%}")
                    with col3:
                        coverage_pct = color_stats.get('vegetation_coverage', 0)
                        st.metric("Vegetation Coverage", f"{coverage_pct:.1%}")
                    
                    # Colored Banner with organic matte finish
                    st.markdown(f"""
                    <div style="background-color: {style['bg']}; padding: 24px; border-radius: 8px; text-align: center; 
                                border: 1px solid {style['border']}; margin-top: 20px;">
                        <h2 style="margin:0; color: {style['text']} !important; font-weight: 500; font-size: 1.8rem; letter-spacing: 0.02em;">
                            {style['emoji']} {class_name.upper()} {style['emoji']}
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed Stats Expander
                    st.markdown("### 📊 Analysis Details")
                    
                    qa_tab1, qa_tab2 = st.tabs(["Confidence Scores", "Scene Descriptions"])
                    
                    with qa_tab1:
                        st.write("**Prediction Confidence:**")
                        
                        # Show final probabilities as bar chart
                        prob_dict = {name: float(p) for name, p in zip(class_names, probs)}
                        st.bar_chart(prob_dict)

                    with qa_tab2:
                        st.write("**🎯 Best Matching Descriptions**")
                        st.caption("Top 3 scene descriptions that match your image")
                        
                        all_prompts = Config().scene_prompts
                        scores = prompt_scores[0].numpy()
                        
                        # Get top 3 prompts
                        top_indices = np.argsort(scores)[::-1][:3]
                        top_scores = scores[top_indices]
                        
                        # Normalize scores to 0-100 range for better visual representation
                        if len(top_scores) > 0 and top_scores.max() > top_scores.min():
                            score_range = top_scores.max() - top_scores.min()
                            normalized_widths = ((top_scores - top_scores.min()) / score_range * 70) + 30  # 30-100% range
                        else:
                            normalized_widths = [100, 80, 60]
                        
                        # Display top 3 with relative color coding
                        for i, idx in enumerate(top_indices):
                            score = scores[idx]
                            prompt = all_prompts[idx]
                            width = normalized_widths[i]
                            
                            # Relative color coding by rank (1st=best, 3rd=weakest of top 3)
                            if i == 0:
                                color = "#28a745"  # Green for #1
                                strength = "🥇 Best Match"
                                medal = "🥇"
                            elif i == 1:
                                color = "#17a2b8"  # Blue for #2
                                strength = "🥈 2nd Best"
                                medal = "🥈"
                            else:
                                color = "#ffc107"  # Yellow for #3
                                strength = "🥉 3rd Best"
                                medal = "🥉"
                            
                            # Display with visual bar
                            st.markdown(f"**{medal} {prompt}**")
                            
                            # Progress bar with custom color
                            st.markdown(f"""
                            <div style="background-color: #f0f0f0; border-radius: 10px; padding: 5px; margin-bottom: 15px;">
                                <div style="background-color: {color}; width: {width}%; height: 20px; border-radius: 8px; 
                                            display: flex; align-items: center; justify-content: flex-end; padding-right: 10px;">
                                </div>
                                <div style="text-align: right; margin-top: 3px; font-size: 11px; color: #666;">
                                    {strength}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)




    # === TAB 2: BATCH EVALUATION ===
    with tab_batch:
        st.header("📊 Batch Evaluation Statistics")
        st.write("Upload images for each class to calculate detailed performance metrics.")
        
        col_up1, col_up2, col_up3 = st.columns(3)
        
        with col_up1:
            st.markdown("<h3 style='text-align: center;'>🌿<br>Healthy</h3>", unsafe_allow_html=True)
            files_healthy = st.file_uploader("Upload Healthy", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="up_h", label_visibility="collapsed")
            
        with col_up2:
            st.markdown("<h3 style='text-align: center;'>🍂<br>Dried</h3>", unsafe_allow_html=True)
            files_dried = st.file_uploader("Upload Dried", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="up_d", label_visibility="collapsed")
            
        with col_up3:
            st.markdown("<h3 style='text-align: center;'>🗑️<br>Contaminated</h3>", unsafe_allow_html=True)
            files_contaminated = st.file_uploader("Upload Contaminated", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="up_c", label_visibility="collapsed")
        
        # Combine all files
        all_files = []
        if files_healthy: all_files.extend([(f, 0) for f in files_healthy]) # 0 = Healthy
        if files_dried: all_files.extend([(f, 1) for f in files_dried])   # 1 = Dried
        if files_contaminated: all_files.extend([(f, 2) for f in files_contaminated]) # 2 = Contaminated
        
        if all_files and st.button("Run Batch Statistics", type="primary"):
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, confusion_matrix
            import pandas as pd
            import seaborn as sns
            
            y_true = []
            y_pred = []
            y_scores = []
            color_stats_list = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_files = len(all_files)
            
            for i, (file, true_label) in enumerate(all_files):
                # Process image
                status_text.text(f"Processing ({i+1}/{total_files}): {file.name}")
                image = Image.open(file).convert('RGB')
                
                # Clean image: remove road signs before scene analysis
                clean_image = mask_road_signs(image)
                
                # Features - CLIP (uses cleaned image)
                image_tensor = preprocess(clean_image).unsqueeze(0)
                scene_emb, prompt_scores = scene_extractor.extract_scene_features(image_tensor)
                
                # DINO features
                if dino_preprocess and scene_extractor.use_dino:
                    dino_tensor = dino_preprocess(clean_image).unsqueeze(0)
                    dino_scene_emb = scene_extractor.extract_dino_scene_features(dino_tensor)
                else:
                    dino_scene_emb = None
                
                # Detect vegetation
                detections = veg_detector.detect_vegetation(image)
                boxes = [d['box'] for d in detections]
                _, veg_emb, _ = veg_extractor.extract_crop_embeddings(image, boxes)
                
                # DINO vegetation features
                if dino_preprocess and boxes:
                    # Unpack tuple: (crop_embeddings_list, mean_embedding_tensor)
                    _, dino_veg_emb = veg_extractor.extract_dino_crop_embeddings(image, boxes)
                else:
                    dino_veg_emb = None
                
                if veg_emb is None: veg_emb = torch.zeros_like(scene_emb)
                color_stats = color_analyzer.aggregate_stats(image, boxes, None)
                color_stats_list.append(color_stats)
                
                # Concatenate all features
                feature_parts = []
                feature_parts.append(scene_emb.numpy().flatten())
                feature_parts.append(veg_emb.numpy().flatten())
                
                # Add DINO if available
                if dino_scene_emb is not None:
                    feature_parts.append(dino_scene_emb.cpu().numpy().flatten())
                
                # BUGFIX: Always append DINO veg features to maintain vector size (1795)
                if scene_extractor.use_dino:
                    if dino_veg_emb is not None:
                        feature_parts.append(dino_veg_emb.cpu().numpy().flatten() if torch.is_tensor(dino_veg_emb) else dino_veg_emb.flatten())
                    else:
                        # Append zeros if no vegetation detected to match scaler expectation
                        dims = scene_extractor.dino_extractor.embedding_dim
                        feature_parts.append(np.zeros(dims, dtype=np.float32))
                
                color_vec = np.array([
                    color_stats.get('mean_green_ratio', 0),
                    color_stats.get('mean_edge_density', 0),
                    color_stats.get('vegetation_coverage', 0)
                ])
                feature_parts.append(color_vec)
                features = np.concatenate(feature_parts).reshape(1, -1)
                features_scaled = scaler.transform(features)
                
                # === Prediction logic (identical to single image tab) ===
                
                # RF prediction
                rf_prediction = classifier.predict(features_scaled)[0]
                if hasattr(classifier, "predict_proba"):
                    rf_probs = classifier.predict_proba(features_scaled)[0]
                else:
                    rf_probs = np.eye(3)[rf_prediction]
                
                # Handle case where classifier wasn't trained on all 3 classes
                if len(rf_probs) < 3:
                    full_probs = np.zeros(3)
                    known_classes = classifier.classes_
                    for ci, cls in enumerate(known_classes):
                        full_probs[cls] = rf_probs[ci]
                    rf_probs = full_probs
                
                # CLIP Prompt-based voting
                all_prompts = Config().scene_prompts
                prompt_class_map = []
                for pi, _ in enumerate(all_prompts):
                    if pi <= 9:
                        prompt_class_map.append(0)
                    elif pi <= 19:
                        prompt_class_map.append(1)
                    else:
                        prompt_class_map.append(2)
                
                scores = prompt_scores[0].cpu().numpy()
                clip_class_scores = np.zeros(3)
                for si, score in enumerate(scores):
                    clip_class_scores[prompt_class_map[si]] += score
                clip_class_scores /= clip_class_scores.sum() if clip_class_scores.sum() > 0 else 1
                
                # Check for consensus in top 3 matches
                top_3_indices = np.argsort(scores)[::-1][:3]
                top_3_classes = [prompt_class_map[ti] for ti in top_3_indices]
                
                full_consensus = len(set(top_3_classes)) == 1
                consensus_class = top_3_classes[0] if full_consensus else -1
                
                from collections import Counter
                class_counts = Counter(top_3_classes)
                majority_class, majority_count = class_counts.most_common(1)[0]
                has_majority = (majority_count >= 2) and not full_consensus
                
                # Ensemble: 60% RF + 40% CLIP
                ensemble_probs = 0.6 * rf_probs + 0.4 * clip_class_scores
                
                # Boost for #1 top prompt class (+5%)
                top_1_class = prompt_class_map[top_3_indices[0]]
                ensemble_probs[top_1_class] += 0.05
                ensemble_probs = ensemble_probs / ensemble_probs.sum()
                
                # Decision Hierarchy
                if full_consensus:
                    pred = consensus_class
                    probs = np.zeros(3)
                    probs[pred] = 0.98
                    others = [c for c in [0,1,2] if c != pred]
                    for c in others: probs[c] = 0.01
                elif has_majority:
                    boosted_probs = ensemble_probs.copy()
                    boosted_probs[majority_class] += 0.07
                    boosted_probs = boosted_probs / boosted_probs.sum()
                    pred = np.argmax(boosted_probs)
                    probs = boosted_probs
                elif rf_probs[rf_prediction] < 0.65:
                    pred = np.argmax(ensemble_probs)
                    probs = ensemble_probs
                else:
                    pred = rf_prediction
                    probs = rf_probs
                
                y_pred.append(pred)
                y_true.append(true_label)
                y_scores.append(probs)
                
                progress_bar.progress((i + 1) / total_files)
            
            status_text.text("Calculating metrics...")
            st.success(f"Processed {total_files} images.")
            
            # Metrics
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_scores = np.vstack(y_scores)
            
            acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            kappa = cohen_kappa_score(y_true, y_pred)
            
            # ROC-AUC requires at least 2 classes
            try:
                if len(np.unique(y_true)) > 1:
                    auc = roc_auc_score(y_true, y_scores, multi_class='ovr', average='weighted', labels=[0, 1, 2])
                else:
                    auc = 0.0 # Undefined for single class
                    st.warning("⚠️ **Note**: ROC-AUC and Kappa are 0.0 because you only uploaded images for one class. To calculate these discrimination metrics, please upload images for at least two classes (e.g., Healthy AND Dried).")
            except Exception as e:
                st.error(f"AUC Calculation Error: {e}")
                auc = 0.0
            
            # Display Metrics Grid
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{acc:.2%}")
            m2.metric("F1-Score", f"{f1:.3f}")
            m3.metric("ROC-AUC", f"{auc:.3f}")
            
            m4, m5, m6 = st.columns(3)
            m4.metric("Precision", f"{precision:.3f}")
            m5.metric("Recall", f"{recall:.3f}")
            m6.metric("Cohen's Kappa", f"{kappa:.3f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
            df_cm = pd.DataFrame(cm, index=['True Healthy', 'True Dried', 'True Contam.'], columns=['Pred Healthy', 'Pred Dried', 'Pred Contam.'])
            st.table(df_cm)
            
            # Detailed Results Table
            st.subheader("📋 Detailed Image Results")
            class_names = ['Healthy', 'Dried', 'Contaminated']
            results_data = []
            for i, (file, true_idx) in enumerate(all_files):
                res = {
                    "Filename": file.name,
                    "True Label": class_names[true_idx],
                    "Predicted": class_names[y_pred[i]],
                    "Confidence": f"{y_scores[i][y_pred[i]]:.2%}",
                    "Veg. Coverage": f"{color_stats_list[i].get('vegetation_coverage', 0):.1%}"
                }
                results_data.append(res)
            st.dataframe(pd.DataFrame(results_data))
        
        elif not all_files:
            st.info("Please upload images to at least one category above.")

                
