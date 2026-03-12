import torch
import torch.nn.functional as F
from math import log10

def calcular_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    return 20 * log10(max_val / torch.sqrt(mse).item())

def calcular_sam(pred, target):
    # SAM = arccos( (A dot B) / (||A|| ||B||) )
    # Aplanamos las dimensiones espaciales para operar sobre los espectros
    pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
    target_flat = target.view(target.shape[0], target.shape[1], -1)
    
    dot_product = torch.sum(pred_flat * target_flat, dim=1)
    norm_pred = torch.norm(pred_flat, dim=1)
    norm_target = torch.norm(target_flat, dim=1)
    
    # Evitamos divisiones por cero
    val = dot_product / (norm_pred * norm_target + 1e-8)
    val = torch.clamp(val, -1.0 + 1e-8, 1.0 - 1e-8) # Estabilidad numérica
    
    sam_angulos = torch.acos(val)
    return torch.mean(sam_angulos).item() * (180.0 / 3.14159265) # Convertimos a grados