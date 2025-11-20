import torch
import torchvision.transforms as T
from models_common import *  # or import specific model classes
from datasets_fer2013 import EMOTIONS

#Global variables for smoothing
smooth_age = None
smooth_gender = None
smooth_expr = None

# Pick device (Apple Silicon, CUDA, or CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

########################################
# 1. Transforms – COPY FROM TRAINING
########################################
# TODO: open train_age_utk.py / train_gender_utk.py / train_expr_fer.py
# and copy the exact transforms (Resize, ToTensor, Normalize...)
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225])
])

########################################
# 2. Model loading
########################################
# TODO: replace AgeModelClass / GenderModelClass / ExprModelClass
# with the actual class names from models_common.py

# Paths to your trained checkpoints
age_ckpt     = "ckpts_age_r18/age_best.pth"
gender_ckpt  = "ckpts_gender_r18/gender_best.pth"
expr_ckpt    = "ckpts_expr_r18/expr_best.pth"

def load_models(device=None):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    # Use ResNet18 backbone to match the checkpoint
    age_model = AgeNet(backbone="resnet18", pretrained=False).to(device)
    gender_model = GenderNet(backbone="resnet18", pretrained=False).to(device)
    expr_model = ExprNet(num_classes=7, backbone="resnet18", pretrained=False).to(device)

    age_state = torch.load(age_ckpt, map_location=device)
    gender_state = torch.load(gender_ckpt, map_location=device)
    expr_state = torch.load(expr_ckpt, map_location=device)

    age_model.load_state_dict(age_state)
    gender_model.load_state_dict(gender_state)
    expr_model.load_state_dict(expr_state)

    age_model.eval()
    gender_model.eval()
    expr_model.eval()

    return age_model, gender_model, expr_model


age_model, gender_model, expr_model = load_models()

########################################
# 3. Single function to call from cv2
########################################
# face_np: numpy array (H, W, 3) RGB uint8

def predict_all(face_np):
    global smooth_age, smooth_gender, smooth_expr
    
    # face_np: (H, W, 3), RGB uint8

    # Apply transforms (Resize, Normalize, etc.)
    x = transform(face_np)              # (3, 224, 224)
    x = x.unsqueeze(0).to(device)       # (1, 3, 224, 224)

    with torch.no_grad():
        age_out = age_model(x)          # (1,)
        gender_out = gender_model(x)    # (1,2)
        expr_out = expr_model(x)        # (1,7)

    # ----- Age with smoothing -----
    age = age_out.item()
    alpha = 0.3  # smoothing factor
    if smooth_age is None:
        smooth_age = age
    else:
        smooth_age = alpha * age + (1 - alpha) * smooth_age
    age = round(float(smooth_age))

    # ----- Gender with smoothing -----
    g_probs = torch.softmax(gender_out, dim=1)[0]  # (2,)
    if smooth_gender is None:
        smooth_gender = g_probs.cpu().numpy()
    else:
        smooth_gender = alpha * g_probs.cpu().numpy() + (1 - alpha) * smooth_gender
    g_idx = smooth_gender.argmax()
    gender = "Male" if g_idx == 0 else "Female"

    # ----- Expression with smoothing -----
    e_probs = torch.softmax(expr_out, dim=1)[0]  # (7,)
    if smooth_expr is None:
        smooth_expr = e_probs.cpu().numpy()
    else:
        smooth_expr = alpha * e_probs.cpu().numpy() + (1 - alpha) * smooth_expr
    e_idx = smooth_expr.argmax()
    expr = EMOTIONS[e_idx]

    return age, gender, expr