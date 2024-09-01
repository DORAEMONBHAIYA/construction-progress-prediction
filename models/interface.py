import torch
from model import get_model

def predict_progress(floors, laborers, image_features):
    model = get_model()
    model.load_state_dict(torch.load('construction_model.pth'))
    model.eval()
    
    input_data = torch.tensor([[floors, laborers, image_features]], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_data)
    
    progress, time_remaining = output[0]
    return progress.item(), time_remaining.item()

# Example usage
floors = 5
laborers = 50
image_features = 0.8  # Example image feature extraction
progress, time_remaining = predict_progress(floors, laborers, image_features)
print(f"Predicted Progress: {progress*100:.2f}%")
print(f"Estimated Time Remaining: {time_remaining:.2f} days")
