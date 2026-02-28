"""
Greenspace Quality Classifier - Training Script with DINOv2 Integration

This script extracts features from the dataset using both CLIP and DINOv2,
trains a RandomForestClassifier with hyperparameter optimization, and
evaluates the model on test data.

Usage:
    python train.py --use-dino --optimize-hyperparams
    python train.py --compare-baseline  # Compare CLIP-only vs CLIP+DINO
"""
import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, roc_auc_score, confusion_matrix,
    classification_report
)

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from scene_features import SceneFeatureExtractor
from vegetation_detector import VegetationDetector, mask_road_signs
from vegetation_features import VegetationFeatureExtractor, ColorTextureAnalyzer
from dataset import get_clip_preprocess


def extract_features_from_dataset(
    dataset_path: str,
    config: Config,
    use_dino: bool = True,
    save_features: bool = True
):
    """
    Extract CLIP + DINO + color features from all images in dataset.
    
    Args:
        dataset_path: Path to dataset split (train/val/test)
        config: Configuration object
        use_dino: Whether to extract DINOv2 features
        save_features: Whether to save extracted features to disk
    
    Returns:
        X: Feature matrix (N, D)
        y: Labels (N,)
        file_paths: List of image file paths
    """
    device = config.device
    
    # Initialize extractors
    print(f"\nInitializing feature extractors (DINO={use_dino})...")
    scene_extractor = SceneFeatureExtractor(
        device=device,
        scene_prompts=config.scene_prompts,
        use_dino=use_dino,
        dino_model_name=config.dino_model_name if use_dino else None,
        dino_image_size=config.dino_image_size if use_dino else None
    )
    veg_detector = VegetationDetector(device=device)
    veg_extractor = VegetationFeatureExtractor(scene_extractor, config.clip_image_size)
    color_analyzer = ColorTextureAnalyzer()
    preprocess = get_clip_preprocess(config.clip_image_size)
    
    # Get DINO preprocess if needed
    dino_preprocess = None
    if use_dino and scene_extractor.dino_extractor:
        dino_preprocess = scene_extractor.dino_extractor.preprocess_image
    
    # Collect image paths
    image_paths = []
    labels = []
    
    for class_name in config.class_names:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        class_idx = config.get_class_index(class_name)
        
        # Load both JPG and PNG images
        jpg_files = list(Path(class_dir).glob("*.jpg"))
        png_files = list(Path(class_dir).glob("*.png"))
        all_files = jpg_files + png_files
        
        for img_file in all_files:
            image_paths.append(str(img_file))
            labels.append(class_idx)
    
    print(f"Found {len(image_paths)} images in {dataset_path}")
    
    # Extract features
    X_list = []
    
    for img_path in tqdm(image_paths, desc="Extracting features"):
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Resize if too large
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # 0. Clean image: remove road signs before scene analysis
            clean_image = mask_road_signs(image)
            
            # 1. CLIP Scene Features (uses cleaned image)
            image_tensor = preprocess(clean_image).unsqueeze(0)
            scene_emb, _ = scene_extractor.extract_scene_features(image_tensor)
            scene_vec = scene_emb.cpu().numpy().flatten()
            
            # 2. DINO Scene Features (uses cleaned image)
            if use_dino and dino_preprocess:
                dino_tensor = dino_preprocess(clean_image).unsqueeze(0)
                dino_scene_emb = scene_extractor.extract_dino_scene_features(dino_tensor)
                dino_scene_vec = dino_scene_emb.cpu().numpy().flatten()
            else:
                dino_scene_vec = np.array([])
            
            # 3. Vegetation Detection
            detections = veg_detector.detect_vegetation(image)
            boxes = [d['box'] for d in detections]
            
            # 4. CLIP Vegetation Features
            _, veg_emb, _ = veg_extractor.extract_crop_embeddings(image, boxes)
            if veg_emb is None:
                veg_emb = torch.zeros(scene_extractor.embedding_dim, device=device)
            veg_vec = veg_emb.cpu().numpy().flatten()
            
            # 5. DINO Vegetation Features (if enabled)
            if use_dino:
                _, dino_veg_emb = veg_extractor.extract_dino_crop_embeddings(image, boxes)
                if dino_veg_emb is not None:
                    dino_veg_vec = dino_veg_emb.cpu().numpy().flatten()
                else:
                    dino_veg_vec = np.zeros(scene_extractor.dino_extractor.embedding_dim) if scene_extractor.dino_extractor else np.array([])
            else:
                dino_veg_vec = np.array([])
            
            # 6. Color/Texture Features
            color_stats = color_analyzer.aggregate_stats(image, boxes, None)
            color_vec = np.array([
                color_stats.get('mean_green_ratio', 0),
                color_stats.get('mean_edge_density', 0),
                color_stats.get('vegetation_coverage', 0)
            ])
            
            # Concatenate all features
            feature_parts = [scene_vec, veg_vec]
            if use_dino:
                feature_parts.extend([dino_scene_vec, dino_veg_vec])
            feature_parts.append(color_vec)
            
            features = np.concatenate(feature_parts)
            X_list.append(features)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Add zero features as fallback
            if X_list:
                X_list.append(np.zeros_like(X_list[0]))
            else:
                # Create zero feature vector with expected size
                expected_size = 512 + 512  # CLIP scene + veg
                if use_dino:
                    expected_size += 384 + 384  # DINO scene + veg
                expected_size += 3  # color features
                X_list.append(np.zeros(expected_size))
    
    X = np.array(X_list)
    y = np.array(labels)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Save features if requested
    if save_features:
        split_name = os.path.basename(dataset_path)
        features_dir = config.output_dir
        os.makedirs(features_dir, exist_ok=True)
        
        suffix = "_dino" if use_dino else "_clip_only"
        np.save(os.path.join(features_dir, f"X_{split_name}{suffix}.npy"), X)
        np.save(os.path.join(features_dir, f"y_{split_name}{suffix}.npy"), y)
        print(f"Saved features to {features_dir}")
    
    return X, y, image_paths


def train_classifier(
    X_train, y_train, X_val, y_val,
    optimize_hyperparams: bool = True,
    config: Config = None
):
    """
    Train RandomForestClassifier with optional hyperparameter optimization.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        optimize_hyperparams: Whether to run GridSearchCV
        config: Configuration object
    
    Returns:
        classifier: Trained classifier
        scaler: Fitted StandardScaler
        best_params: Best hyperparameters (if optimized)
    """
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    if optimize_hyperparams:
        print("\nOptimizing hyperparameters with GridSearchCV...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1_weighted',
            verbose=2, n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        classifier = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"\nBest parameters: {best_params}")
        print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
    else:
        print("\nTraining with default parameters...")
        best_params = {}
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        classifier.fit(X_train_scaled, y_train)
    
    # Evaluate on validation
    y_val_pred = classifier.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")
    
    return classifier, scaler, best_params


def evaluate_model(classifier, scaler, X_test, y_test, config: Config, save_path: str):
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Args:
        classifier: Trained classifier
        scaler: Fitted scaler
        X_test, y_test: Test data
        config: Configuration object
        save_path: Directory to save results
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred = classifier.predict(X_test_scaled)
    y_proba = classifier.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    kappa = cohen_kappa_score(y_test, y_pred)
    
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    except:
        auc = 0.0
    
    # Print results
    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("="*50)
    
    print("\nPer-Class Metrics:")
    try:
        print(classification_report(y_test, y_pred, target_names=config.class_names, labels=[0,1,2]))
    except Exception as e:
        print(f"Could not generate full report: {e}")
        print(f"Unique labels in test: {np.unique(y_test)}")
        print(f"Unique labels in predictions: {np.unique(y_pred)}")
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'cohen_kappa': float(kappa),
        'roc_auc': float(auc)
    }
    
    with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.class_names,
                yticklabels=config.class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300)
    print(f"\nSaved confusion matrix to {save_path}/confusion_matrix.png")
    
    # Feature Importance
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        top_k = min(20, len(importances))
        indices = np.argsort(importances)[-top_k:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_k), importances[indices])
        plt.yticks(range(top_k), [f"Feature {i}" for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_k} Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'feature_importance.png'), dpi=300)
        print(f"Saved feature importance to {save_path}/feature_importance.png")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Greenspace Classifier')
    parser.add_argument('--use-dino', action='store_true', help='Use DINOv2 features')
    parser.add_argument('--optimize-hyperparams', action='store_true', help='Run GridSearchCV')
    parser.add_argument('--compare-baseline', action='store_true', help='Compare CLIP-only vs CLIP+DINO')
    parser.add_argument('--data-root', type=str, default=None, help='Override data root directory')
    args = parser.parse_args()
    
    config = Config()
    if args.data_root:
        config.data_root = args.data_root
    
    # Paths
    train_path = os.path.join(config.data_root, 'train')
    val_path = os.path.join(config.data_root, 'val')
    test_path = os.path.join(config.data_root, 'test')
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    if args.compare_baseline:
        print("BASELINE COMPARISON MODE: Training both CLIP-only and CLIP+DINO models")
        results = {}
        
        for use_dino in [False, True]:
            model_type = "CLIP+DINO" if use_dino else "CLIP-only"
            print(f"\n{'='*60}\nTraining {model_type} Model\n{'='*60}")
            
            # Extract features
            X_train, y_train, _ = extract_features_from_dataset(train_path, config, use_dino)
            X_val, y_val, _ = extract_features_from_dataset(val_path, config, use_dino, save_features=False)
            X_test, y_test, _ = extract_features_from_dataset(test_path, config, use_dino, save_features=False)
            
            # Train
            classifier, scaler, _ = train_classifier(X_train, y_train, X_val, y_val, False, config)
            
            # Evaluate
            save_subdir = os.path.join(models_dir, 'dino' if use_dino else 'baseline')
            os.makedirs(save_subdir, exist_ok=True)
            metrics = evaluate_model(classifier, scaler, X_test, y_test, config, save_subdir)
            results[model_type] = metrics
        
        # Print comparison
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        for model_type, metrics in results.items():
            print(f"\n{model_type}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
        
        improvement = results["CLIP+DINO"]["accuracy"] - results["CLIP-only"]["accuracy"]
        print(f"\nAccuracy Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        
    else:
        # Standard training mode
        print(f"\nTraining with DINO={'enabled' if args.use_dino else 'disabled'}")
        
        # Extract features
        X_train, y_train, _ = extract_features_from_dataset(train_path, config, args.use_dino)
        X_val, y_val, _ = extract_features_from_dataset(val_path, config, args.use_dino, save_features=False)
        X_test, y_test, _ = extract_features_from_dataset(test_path, config, args.use_dino, save_features=False)
        
        # Train
        classifier, scaler, best_params = train_classifier(
            X_train, y_train, X_val, y_val,
            args.optimize_hyperparams, config
        )
        
        # Evaluate
        metrics = evaluate_model(classifier, scaler, X_test, y_test, config, models_dir)
        
        # Save models
        suffix = "_dino" if args.use_dino else ""
        classifier_path = os.path.join(models_dir, f'best_classifier{suffix}.pkl')
        scaler_path = os.path.join(models_dir, f'scaler{suffix}.pkl')
        
        with open(classifier_path, 'wb') as f:
            pickle.dump(classifier, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"\nSaved classifier to {classifier_path}")
        print(f"Saved scaler to {scaler_path}")
        
        # Save hyperparameters
        if best_params:
            with open(os.path.join(models_dir, 'best_params.json'), 'w') as f:
                json.dump(best_params, f, indent=2)
        
        print(f"\n✅ Training complete! Test accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
