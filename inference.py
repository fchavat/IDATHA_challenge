import io
import torch
from torchvision import transforms
from PIL import Image
from CNN import CNN

model = CNN(29)
MODEL_NAME = "model_CNN-propio_2204111849_e30"
checkpoint = torch.load(f"checkpoints/{MODEL_NAME}.pt")
ind_to_class = list(checkpoint["class_to_ind"].keys())
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

model_epoch = checkpoint['epoch']
model_stats = checkpoint['stats'][model_epoch-1]
train_acc = float(model_stats['train']['correct']) / model_stats['train']['total']
val_acc = float(model_stats['val']['correct']) / model_stats['val']['total']
def transform_image(img_bytes):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    img = Image.open(io.BytesIO(img_bytes))
    return transform(img).unsqueeze(0)

def prediction(img_bytes):
    try:
        tensor = transform_image(img_bytes)
        output = model.forward(tensor)
    except Exception:
        return 'error'
    preds = [(ind, val.item()) for ind, val in enumerate(output[0])]
    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    best_3 = preds[:3]
    return str([(ind_to_class[b[0]], round(b[1],3)) for b in best_3])

def model_info():
    return {
        'model_name': MODEL_NAME,
        'train_accuracy': round(train_acc, 3),
        'val_accuracy': round(val_acc, 3),
    }
