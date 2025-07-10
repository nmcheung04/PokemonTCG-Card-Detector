from ultralytics import YOLO

# Load model (nano for speed)
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='datasets/pokemon/pokemon.yaml',
    epochs=50,
    imgsz=640,
    project='runs/detect',
    name='pokemon',
    save=True,  # Save best model
    save_period=10,  # Save every 10 epochs
)

print('Training complete! Results:')
print(results)

# Print model file locations
print('\nModel files saved:')
print('- Best model: runs/detect/pokemon/weights/best.pt')
print('- Last checkpoint: runs/detect/pokemon/weights/last.pt')
print('- Use best.pt for inference in your card_matcher pipeline') 