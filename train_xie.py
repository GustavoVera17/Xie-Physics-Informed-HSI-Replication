import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import matplotlib
matplotlib.use('Agg') # Modo Headless para no trabar el entrenamiento
import matplotlib.pyplot as plt

# Importamos nuestros módulos rigurosamente armados
from dataset_dual import CASSIDualDataset
from sir_cnn_W48 import SIR_CNN
from physics_loss import SSL_Grayscale_Loss, SSL_Color_Loss
from metricas import calcular_psnr, calcular_sam
from skimage.metrics import structural_similarity as ssim_metric

def calcular_ssim(pred, target):
    pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0) 
    target_np = target.detach().cpu().numpy().transpose(1, 2, 0)
    ssim_val = ssim_metric(target_np, pred_np, data_range=1.0, channel_axis=-1)
    return float(ssim_val)

def train_xie():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando Entrenamiento Self-Supervised (Xie et al.) en: {device}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta_train = os.path.join(BASE_DIR, "dataset", "fortrain")
    ruta_test = os.path.join(BASE_DIR, "dataset", "fortest")
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints_xie")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    ruta_checkpoint_latest = os.path.join(checkpoint_dir, "xie_hrnet_latest.pth")

    # =========================================================================
    # ⚙️ HIPERPARÁMETROS EXACTOS DEL PAPER
    # =========================================================================
    BATCH_SIZE = 4 # Ajustado para evitar Out Of Memory. Súbelo si tu GPU aguanta.
    MAX_EPOCHS = 1200     
    INITIAL_LR = 0.001
    FACTOR_REDUCCION = 0.2
    EPOCAS_REDUCCION = 300
    FRECUENCIA_DASHBOARD = 10 # Exportar imagen cada 10 épocas
    FRECUENCIA_SAVE_50 = 50   # Guardar un checkpoint duro cada 50 épocas
    # =========================================================================

    print("Cargando datasets duales a la RAM...")
    dataset_train = CASSIDualDataset(root_dir=ruta_train, patch_size=256, num_patches_per_img=10, is_train=True)
    dataset_test = CASSIDualDataset(root_dir=ruta_test, patch_size=256, num_patches_per_img=1, is_train=False) 
    
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # 1. Instanciamos la Red y Jueces
    modelo = SIR_CNN(num_bands=31).to(device)
    mascara_sistema = dataset_train.get_mask().to(device)
    juez_cassi = SSL_Grayscale_Loss(mask=mascara_sistema).to(device)
    juez_rgb = SSL_Color_Loss(device=device)
    
    optimizador = optim.Adam(modelo.parameters(), lr=INITIAL_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizador, step_size=EPOCAS_REDUCCION, gamma=FACTOR_REDUCCION)

    # 2. SISTEMA DE RESUME (Checkpoints)
    start_epoch = 0
    mejor_psnr_val = 0.0
    
    # Históricos para el Dashboard
    hist_loss_total, hist_loss_cassi, hist_loss_rgb = [], [], []
    hist_psnr, hist_ssim, hist_sam = [], [], []

    if os.path.exists(ruta_checkpoint_latest):
        print(f"[*] Se encontró un Checkpoint previo. Reanudando entrenamiento...")
        checkpoint = torch.load(ruta_checkpoint_latest, map_location=device)
        modelo.load_state_dict(checkpoint['model_state_dict'])
        optimizador.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        mejor_psnr_val = checkpoint['mejor_psnr']
        
        # Recuperar historiales para que las gráficas no se reinicien
        hist_loss_total = checkpoint.get('hist_loss_total', [])
        hist_loss_cassi = checkpoint.get('hist_loss_cassi', [])
        hist_loss_rgb = checkpoint.get('hist_loss_rgb', [])
        hist_psnr = checkpoint.get('hist_psnr', [])
        hist_ssim = checkpoint.get('hist_ssim', [])
        hist_sam = checkpoint.get('hist_sam', [])
        print(f"[*] Reanudando desde la Época {start_epoch+1}")
    else:
        print("\n¡Iniciando entrenamiento desde cero!")

    for epoch in range(start_epoch, MAX_EPOCHS):
        modelo.train()
        loss_epoch_total, loss_epoch_cassi, loss_epoch_rgb = 0.0, 0.0, 0.0
        
        loop_entrenamiento = tqdm(loader_train, desc=f"Época [{epoch+1}/{MAX_EPOCHS}]", leave=True)
        
        for batch_idx, (cassi_real, rgb_real, gt_cube) in enumerate(loop_entrenamiento):
            cassi_real = cassi_real.to(device)
            rgb_real = rgb_real.to(device)
            
            optimizador.zero_grad()
            
            prediccion_raw = modelo(cassi_real) 
            prediccion_3d = prediccion_raw[:, :, :, :256].contiguous() # Contiguous asegura la memoria
            
            loss_gb, cassi_sim = juez_cassi(prediccion_3d, cassi_real)
            loss_cb, rgb_sim = juez_rgb(prediccion_3d, rgb_real)
            
            loss_total = loss_gb + loss_cb
            
            loss_total.backward()
            optimizador.step()
            
            loss_epoch_total += loss_total.item()
            loss_epoch_cassi += loss_gb.item()
            loss_epoch_rgb += loss_cb.item()
            loop_entrenamiento.set_postfix(LossT=f"{loss_total.item():.4f}")

        scheduler.step()
        lr_actual = scheduler.get_last_lr()[0]
        
        avg_loss_total = loss_epoch_total / len(loader_train)
        hist_loss_total.append(avg_loss_total)
        hist_loss_cassi.append(loss_epoch_cassi / len(loader_train))
        hist_loss_rgb.append(loss_epoch_rgb / len(loader_train))

        # -- VALIDACIÓN Y MÉTRICAS (Con Ground Truth) --
        modelo.eval()
        val_psnr_total, val_ssim_total, val_sam_total = 0.0, 0.0, 0.0
        
        with torch.no_grad(): 
            for val_cassi, val_rgb, val_gt in loader_test:
                val_cassi, val_gt = val_cassi.to(device), val_gt.to(device)
                val_pred_raw = modelo(val_cassi)
                val_pred_3d = val_pred_raw[:, :, :, :256].contiguous()
                
                val_psnr_total += float(calcular_psnr(val_pred_3d, val_gt))
                val_sam_total += float(calcular_sam(val_pred_3d, val_gt))
                val_ssim_total += float(calcular_ssim(val_pred_3d[0], val_gt[0]))
                
        avg_psnr = val_psnr_total / len(loader_test)
        avg_ssim = val_ssim_total / len(loader_test)
        avg_sam = val_sam_total / len(loader_test)
        
        hist_psnr.append(avg_psnr)
        hist_ssim.append(avg_ssim)
        hist_sam.append(avg_sam)

        # 🚨 GUARDADO DE CHECKPOINT "LATEST" (Para reanudar)
        estado_checkpoint = {
            'epoch': epoch,
            'model_state_dict': modelo.state_dict(),
            'optimizer_state_dict': optimizador.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'mejor_psnr': mejor_psnr_val,
            'hist_loss_total': hist_loss_total,
            'hist_loss_cassi': hist_loss_cassi,
            'hist_loss_rgb': hist_loss_rgb,
            'hist_psnr': hist_psnr,
            'hist_ssim': hist_ssim,
            'hist_sam': hist_sam
        }
        torch.save(estado_checkpoint, ruta_checkpoint_latest)

        # 🚨 GUARDADO DEL MEJOR MODELO
        if avg_psnr > mejor_psnr_val:
            mejor_psnr_val = avg_psnr
            torch.save(modelo.state_dict(), os.path.join(checkpoint_dir, "xie_hrnet_best.pth"))
            print(f"⭐ ¡Nuevo récord! Modelo guardado. (PSNR: {avg_psnr:.2f}dB)")
            
        # 🚨 GUARDADO CADA 50 ÉPOCAS
        if (epoch + 1) % FRECUENCIA_SAVE_50 == 0:
            torch.save(estado_checkpoint, os.path.join(checkpoint_dir, f"xie_hrnet_epoch_{epoch+1}.pth"))
            print(f"💾 Checkpoint de Época {epoch+1} guardado.")

        print(f"Resumen Época {epoch+1} | LR: {lr_actual:.6f} | Loss(Cassi+RGB): {hist_loss_cassi[-1]:.4f} + {hist_loss_rgb[-1]:.4f}")
        print(f"Métricas (VS GT) | PSNR: {avg_psnr:.2f}dB | SSIM: {avg_ssim:.4f} | SAM: {avg_sam:.2f}°\n")

        # =====================================================================
        # EXPORTACIÓN DEL DASHBOARD PNG
        # =====================================================================
        if (epoch + 1) % FRECUENCIA_DASHBOARD == 0 or epoch == 0:
            fig = plt.figure(figsize=(18, 9))
            
            # Tomamos el último lote de prueba para graficar
            img_cassi_plot = val_cassi[0, 0].cpu().numpy()
            img_rgb_real_plot = val_rgb[0].cpu().permute(1, 2, 0).numpy()
            img_rgb_real_plot = np.clip(img_rgb_real_plot / img_rgb_real_plot.max(), 0, 1)
            
            # Pasamos la predicción por el Juez RGB de nuevo para sacar la imagen a color simulada
            _, rgb_simulado = juez_rgb(val_pred_3d, val_rgb.to(device))
            img_rgb_sim_plot = rgb_simulado[0].detach().cpu().permute(1, 2, 0).numpy()
            img_rgb_sim_plot = np.clip(img_rgb_sim_plot / img_rgb_sim_plot.max(), 0, 1)
            
            banda_idx = 15
            img_gt_band = val_gt[0, banda_idx].cpu().numpy()
            img_pred_band = val_pred_3d[0, banda_idx].detach().cpu().numpy()

            ax1 = fig.add_subplot(2, 4, 1)
            ax1.set_title("Medición CASSI PAN")
            ax1.imshow(img_cassi_plot, cmap='gray')
            ax1.axis('off')

            ax2 = fig.add_subplot(2, 4, 2)
            ax2.set_title("RGB Real (Ground Truth)")
            ax2.imshow(img_rgb_real_plot)
            ax2.axis('off')

            ax3 = fig.add_subplot(2, 4, 3)
            ax3.set_title("RGB Simulado (Predicción)")
            ax3.imshow(img_rgb_sim_plot)
            ax3.axis('off')

            ax4 = fig.add_subplot(2, 4, 4)
            ax4.set_title(f"Banda {banda_idx} - Real vs Pred")
            ax4.imshow(np.concatenate((img_gt_band, img_pred_band), axis=1), cmap='gray')
            ax4.axis('off')

            ax5 = fig.add_subplot(2, 4, 5)
            ax5.set_title("Evolución de Pérdidas (Loss)")
            ax5.plot(hist_loss_total, label='Total', color='black', linewidth=2)
            ax5.plot(hist_loss_rgb, label='Loss Color', color='red', linestyle='--')
            ax5.plot(hist_loss_cassi, label='Loss Grayscale', color='blue', linestyle='--')
            ax5.legend()
            ax5.grid(True)

            ax6 = fig.add_subplot(2, 4, 6)
            ax6.set_title("PSNR (Validación)")
            ax6.plot(hist_psnr, color='blue')
            ax6.grid(True)

            ax7 = fig.add_subplot(2, 4, 7)
            ax7.set_title("SSIM (Validación)")
            ax7.plot(hist_ssim, color='green')
            ax7.grid(True)
            
            ax8 = fig.add_subplot(2, 4, 8)
            ax8.set_title("SAM (Validación)")
            ax8.plot(hist_sam, color='purple')
            ax8.grid(True)

            fig.suptitle(f"Dashboard Xie et al. | Época: {epoch+1}", fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            ruta_dashboard = os.path.join(BASE_DIR, "dashboard_xie.png")
            plt.savefig(ruta_dashboard, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"📸 Dashboard exportado exitosamente.")

if __name__ == "__main__":
    train_xie()