import torch
import matplotlib.pyplot as plt
import numpy as np

# Importamos nuestros módulos recién creados
from dataset_dual import CASSIDualDataset
from sir_cnn_W48 import SIR_CNN

def main():
    print("Iniciando Prueba Visual de la Red No Entrenada...")
    
    # 1. Cargamos UNA imagen de prueba real
    # Asegúrate de poner tu ruta correcta
    RUTA_PRUEBA = r"C:\CODE2026\CASSIproy2026\CASSIproy2026\Yamawaki_Rep\dataset\fortest" 
    dataset = CASSIDualDataset(root_dir=RUTA_PRUEBA, patch_size=256, num_patches_per_img=1)
    
    cassi_tensor, rgb_tensor, gt_cube = dataset[0]
    # Añadimos la dimensión del "Batch" (Lote) para la red neuronal: [1, 1, 256, 286]
    cassi_input = cassi_tensor.unsqueeze(0) 
    
    # 2. Instanciamos nuestra red VIRGEN (con pesos aleatorios)
    modelo = SIR_CNN(num_bands=31)
    modelo.eval() # Modo evaluación para no alterar nada
    
    # 3. Pasamos la imagen por la red
    with torch.no_grad():
        pred_cube_raw = modelo(cassi_input) # Salida: [1, 31, 256, 286]
        
        # RECORTAMOS la dispersión extra para volver al 256x256 original
        pred_cube = pred_cube_raw[:, :, :, :256] 
        
    # 4. Preparamos las imágenes para graficar
    img_cassi = cassi_tensor[0].numpy()
    img_rgb_real = rgb_tensor.permute(1, 2, 0).numpy()
    img_rgb_real = np.clip(img_rgb_real / img_rgb_real.max(), 0, 1) # Normalizar
    
    # Seleccionamos una banda al azar de la predicción (ej. la banda 15, aprox 550nm Verde)
    banda_predicha = pred_cube[0, 15].numpy()
    banda_real = gt_cube[15].numpy()

    # 5. Visualización
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Respuesta Inicial de SIR-CNN (SIN ENTRENAR / PESOS ALEATORIOS)", fontsize=14, fontweight='bold')
    
    axes[0].set_title("Entrada: CASSI Compresiva\n(Hardware)")
    axes[0].imshow(img_cassi, cmap='gray')
    axes[0].axis('off')
    
    axes[1].set_title("Referencia: Cámara RGB\n(Hardware)")
    axes[1].imshow(img_rgb_real)
    axes[1].axis('off')
    
    axes[2].set_title("Ground Truth (Banda 15)\n(Oculto a la red)")
    axes[2].imshow(banda_real, cmap='gray')
    axes[2].axis('off')
    
    axes[3].set_title("Salida Red No Entrenada\n(Pura alucinación matemática)")
    # Usamos un mapa de calor para ver mejor el ruido
    axes[3].imshow(banda_predicha, cmap='viridis') 
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()