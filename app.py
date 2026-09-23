import gradio as gr
import torch as tch
import numpy as np
import myModel as mynet
import os

# 1. Define the exact features used during training
CANCER_TYPES = [
    'ACC', 'ALL', 'BLCA', 'BRCA', 'CESC', 'CLL', 'COAD/READ', 'DLBC', 'ESCA', 'GBM', 
    'HNSC', 'KIRC', 'LAML', 'LCML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MB', 'MESO', 'MM', 
    'NB', 'OV', 'PAAD', 'PRAD', 'SCLC', 'SKCM', 'STAD', 'THCA', 'UCEC', 'UNABLE TO CLASSIFY'
]

PATHWAYS = [
    'ABL signaling', 'Apoptosis regulation', 'Cell cycle', 'Chromatin histone acetylation', 
    'Chromatin histone methylation', 'Chromatin other', 'Cytoskeleton', 'DNA replication', 
    'EGFR signaling', 'ERK MAPK signaling', 'Genome integrity', 'Hormone-related', 
    'IGF1R signaling', 'JNK and p38 signaling', 'Metabolism', 'Mitosis', 'Other', 
    'Other, kinases', 'PI3K/MTOR signaling', 'Protein stability and degradation', 
    'RTK signaling', 'Unclassified', 'WNT signaling', 'p53 pathway'
]

N_FEATURES = len(CANCER_TYPES) + len(PATHWAYS)

# 2. Load the trained PyTorch model
MODEL_PATH = "dataset/model_output.FNN.cv_5best_model.pt"

print("Loading model from:", MODEL_PATH)
net = mynet.FNN(N_FEATURES)

if os.path.exists(MODEL_PATH):
    net.load_state_dict(tch.load(MODEL_PATH, map_location=tch.device('cpu')))
    net.eval()
else:
    print(f"Warning: Model file not found at {MODEL_PATH}. Using untrained weights.")

# 3. Prediction function
def predict_sensitivity(cancer_type, pathway):
    # Initialize zero vector
    features = np.zeros(N_FEATURES, dtype=np.float32)
    
    # Set the corresponding CancerType to 1
    if cancer_type in CANCER_TYPES:
        idx_cancer = CANCER_TYPES.index(cancer_type)
        features[idx_cancer] = 1.0
        
    # Set the corresponding Pathway to 1
    if pathway in PATHWAYS:
        idx_pathway = len(CANCER_TYPES) + PATHWAYS.index(pathway)
        features[idx_pathway] = 1.0
        
    # Convert to PyTorch tensor
    tensor_input = tch.from_numpy(features).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with tch.no_grad():
        prediction = net(tensor_input).item()
        
    return round(prediction, 4)

# 4. Build the Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# PathDSP: Drug Sensitivity Predictor")
    gr.Markdown("Select a cancer cell-line type and a target drug pathway to predict the log(IC50) sensitivity score. Lower IC50 means the cancer is more sensitive to the drug.")
    
    with gr.Row():
        cancer_dropdown = gr.Dropdown(choices=CANCER_TYPES, label="Cancer Type")
        pathway_dropdown = gr.Dropdown(choices=PATHWAYS, label="Target Pathway")
        
    predict_btn = gr.Button("Predict Sensitivity", variant="primary")
    
    output_score = gr.Number(label="Predicted LN_IC50 Score")
    
    # Map button click to function
    predict_btn.click(
        fn=predict_sensitivity, 
        inputs=[cancer_dropdown, pathway_dropdown], 
        outputs=output_score
    )

if __name__ == "__main__":
    print("Starting interface...")
    interface.launch(share=False)
