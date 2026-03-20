import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.metrics import structural_similarity as ssim_metric

# Importamos los módulos específicos de la replicación de Xie
from dataset_dual import CASSIDualDataset
from sir_cnn_W48SB import SIR_CNN
from metricas import calcular_psnr, calcular_sam

def calcular_ssim(pred, target):
    pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0) 
    target_np = target.detach().cpu().numpy().transpose(1, 2, 0)
    ssim_val = ssim_metric(target_np, pred_np, data_range=1.0, channel_axis=-1)
    return float(ssim_val)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando Inferencia SIR-CNN (Xie et al.) en: {device}")

    # =========================================================================
    # ⚙️ CONFIGURACIÓN: Cambia las rutas según lo que quieras evaluar
    # =========================================================================
    
    # Ruta del modelo .pth que generaste en el entrenamiento de Xie
    RUTA_MODELO_PTH = r"C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep\checkpoints_xie\xie_hrnet_best.pth"
    
    # Ruta exacta de la carpeta de la imagen de prueba (Misma que usabas antes)
    RUTA_CARPETA_IMAGEN = r"C:\CODE2026\CASSIproy2026\CASSIproy2026\Xie_Rep\dataset\fortest\balloons_ms"

    # =========================================================================

    if not os.path.exists(RUTA_MODELO_PTH):
        raise FileNotFoundError(f"No se encontró el modelo en: {RUTA_MODELO_PTH}")
    if not os.path.exists(RUTA_CARPETA_IMAGEN):
        raise FileNotFoundError(f"No se encontró la imagen en: {RUTA_CARPETA_IMAGEN}")

    ruta_padre = os.path.dirname(RUTA_CARPETA_IMAGEN)
    nombre_imagen_objetivo = os.path.basename(RUTA_CARPETA_IMAGEN)

    print(f"Cargando el entorno de validación desde: {ruta_padre}...")
    dataset_test = CASSIDualDataset(root_dir=ruta_padre, patch_size=256, num_patches_per_img=1, is_train=False)

    idx_elegido = None
    for i, folder in enumerate(dataset_test.image_folders):
        if os.path.basename(folder) == nombre_imagen_objetivo:
            idx_elegido = i
            break
            
    if idx_elegido is None:
        raise ValueError(f"No se pudo encontrar '{nombre_imagen_objetivo}' en el dataset.")

    # Inicializar la Red SIR-CNN (HRNet-W18)
    modelo = SIR_CNN(num_bands=31).to(device)
    modelo.load_state_dict(torch.load(RUTA_MODELO_PTH, map_location=device))
    modelo.eval()

    with torch.no_grad():
        # El Dataset Dual devuelve 3 tensores, extraemos CASSI y el GT
        cassi_real, rgb_real, gt_cube = dataset_test[idx_elegido]
        
        # Añadimos la dimensión del Batch para la red
        cassi_real = cassi_real.unsqueeze(0).to(device)
        gt_cube = gt_cube.unsqueeze(0).to(device)
        
        # Pasamos la imagen por la red
        pred_raw = modelo(cassi_real)
        
        # EL RECORTE CRÍTICO: Quitamos la dispersión para alinear con el GT de 256x256
        pred_3d = pred_raw[:, :, :, :256].contiguous()
        
        psnr_val = calcular_psnr(pred_3d, gt_cube)
        ssim_val = calcular_ssim(pred_3d[0], gt_cube[0])
        sam_val = calcular_sam(pred_3d, gt_cube)

    # Preparamos las imágenes para graficar
    img_pan = cassi_real[0, 0].cpu().numpy()
    cubo_gt = gt_cube[0].cpu().numpy()     
    cubo_pred = pred_3d[0].cpu().numpy() 
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.canvas.manager.set_window_title(f'Inferencia Xie et al. (HRNet)')
    plt.subplots_adjust(bottom=0.25, top=0.85)

    fig.suptitle(f"Imagen: {nombre_imagen_objetivo} | SIR-CNN | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | SAM: {sam_val:.2f}°", 
                 fontsize=14, fontweight='bold')

    longitudes_onda = np.arange(400, 710, 10)
    banda_inicial = 15 # Empezamos por una banda intermedia (aprox 550nm)

    axes[0].set_title("Medición CASSI Compresiva")
    axes[0].imshow(img_pan, cmap='gray')
    axes[0].axis('off')

    titulo_gt = axes[1].set_title(f"Verdad Original ({longitudes_onda[banda_inicial]}nm)")
    img_gt_plot = axes[1].imshow(cubo_gt[banda_inicial], cmap='gray', vmin=0, vmax=1)
    axes[1].axis('off')

    titulo_pred = axes[2].set_title(f"Predicción SIR-CNN ({longitudes_onda[banda_inicial]}nm)")
    img_pred_plot = axes[2].imshow(cubo_pred[banda_inicial], cmap='gray', vmin=0, vmax=1)
    axes[2].axis('off')

    # Escala de Error Absoluto FIJA de 0 a 1 con mapa arcoíris
    error_inicial = np.abs(cubo_gt[banda_inicial] - cubo_pred[banda_inicial])
    axes[3].set_title("Mapa de Error Absoluto (Fijo 0-1)")
    
    img_error_plot = axes[3].imshow(error_inicial, cmap='rainbow', vmin=0, vmax=1.0) 
    fig.colorbar(img_error_plot, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].axis('off')

    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor=axcolor)
    slider_banda = Slider(
        ax=ax_slider,
        label='Banda Espectral',
        valmin=0,
        valmax=30,
        valinit=banda_inicial,
        valstep=1
    )

    def update(val):
        b = int(slider_banda.val)
        wl = longitudes_onda[b]
        
        img_gt_plot.set_data(cubo_gt[b])
        img_pred_plot.set_data(cubo_pred[b])
        
        # El mapa se actualiza, el límite se mantiene estricto en 0-1
        nuevo_error = np.abs(cubo_gt[b] - cubo_pred[b])
        img_error_plot.set_data(nuevo_error)
        
        titulo_gt.set_text(f"Verdad Original ({wl}nm)")
        titulo_pred.set_text(f"Predicción SIR-CNN ({wl}nm)")
        
        fig.canvas.draw_idle()

    slider_banda.on_changed(update)
    plt.show()

if __name__ == "__main__":
    main()