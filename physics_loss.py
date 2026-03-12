import torch
import torch.nn as nn

class SSL_Grayscale_Loss(nn.Module):
    """
    JUEZ 1 (SSL-GB): Simula el proceso físico de la cámara CASSI.
    Aplica la máscara codificada, simula la dispersión del prisma y suma las bandas.
    """
    def __init__(self, mask):
        super(SSL_Grayscale_Loss, self).__init__()
        # La máscara debe tener forma [1, 1, H, W]
        self.mask = mask
        self.mse = nn.MSELoss()

    def forward(self, hsi_pred, cassi_real):
        # hsi_pred: [Batch, 31, 256, 256] -> La "adivinanza" de la red
        B, C, H, W = hsi_pred.shape
        
        # 1. Modulación Espacial: Multiplicamos por la máscara T(x,y)
        hsi_enmascarado = hsi_pred * self.mask
        
        # 2. Simulación de la Dispersión (Shift) y Colapso (Sumatoria)
        # La imagen será más ancha en el eje horizontal debido al prisma (W + C - 1)
        cassi_simulado = torch.zeros((B, 1, H, W + C - 1), device=hsi_pred.device)
        
        for i in range(C):
            # Desplazamos cada banda 'i' píxeles hacia la derecha y la sumamos
            cassi_simulado[:, 0, :, i : i + W] += hsi_enmascarado[:, i, :, :]
            
        # 3. Calculamos el Error Cuadrático Medio
        loss_grayscale = self.mse(cassi_simulado, cassi_real)
        
        return loss_grayscale, cassi_simulado

class SSL_Color_Loss(nn.Module):
    """
    JUEZ 2 (SSL-CB): Simula el proceso de la cámara RGB.
    Integra las 31 bandas espectrales usando las curvas de eficiencia cuántica (K) de CAVE.
    """
    def __init__(self, device="cpu"):
        super(SSL_Color_Loss, self).__init__()
        
        # Transcripción exacta de la tabla CAVE (400nm a 700nm, pasos de 10nm)
        # Matriz de [3, 31] -> [R, G, B]
        cave_crf = [
            # Canal ROJO
            [0.0073, 0.0326, 0.1146, 0.2238, 0.2319, 0.1408, 0.0545, 0.0063, 0.0016, 0.0016,
             0.0020, 0.0049, 0.0163, 0.0458, 0.1065, 0.2088, 0.3516, 0.5284, 0.7042, 0.8359,
             0.8876, 0.8407, 0.7186, 0.5513, 0.3840, 0.2458, 0.1472, 0.0844, 0.0470, 0.0253, 0.0135],
            # Canal VERDE
            [0.0001, 0.0004, 0.0016, 0.0039, 0.0069, 0.0125, 0.0232, 0.0438, 0.0841, 0.1417,
             0.2114, 0.3160, 0.4578, 0.6127, 0.7303, 0.7711, 0.7259, 0.6094, 0.4566, 0.2974,
             0.1691, 0.0863, 0.0410, 0.0188, 0.0084, 0.0037, 0.0016, 0.0007, 0.0003, 0.0001, 0.0000],
            # Canal AZUL
            [0.0336, 0.1558, 0.5847, 1.2588, 1.4883, 1.1396, 0.6908, 0.3546, 0.1584, 0.0658,
             0.0267, 0.0099, 0.0038, 0.0015, 0.0005, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        ]
        
        # Convertimos a tensor y lo enviamos al dispositivo (CPU/GPU)
        self.crf = torch.tensor(cave_crf, dtype=torch.float32, device=device)
        self.mse = nn.MSELoss()

    def forward(self, hsi_pred, rgb_real):
        # hsi_pred: [Batch, 31, 256, 256]
        B, C, H, W = hsi_pred.shape
        
        # Aplanamos espacialmente: de [B, 31, 256, 256] a [B, 31, 65536]
        hsi_flat = hsi_pred.reshape(B, C, -1)
        
        # Expandimos la matriz CRF para el tamaño del lote: [B, 3, 31]
        crf_batch = self.crf.unsqueeze(0).expand(B, -1, -1)
        
        # Multiplicación matricial pura de tensores (Batch Matrix Multiplication)
        # [B, 3, 31] @ [B, 31, 65536] = [B, 3, 65536]
        rgb_simulado_flat = torch.bmm(crf_batch, hsi_flat)
        
        # Restauramos la dimensión espacial: [B, 3, 256, 256]
        rgb_simulado = rgb_simulado_flat.reshape(B, 3, H, W)
        
        # Calculamos el MSE
        loss_color = self.mse(rgb_simulado, rgb_real)
        
        return loss_color, rgb_simulado

# ==========================================
# ZONA DE PRUEBA (EJECUTABLE)
# ==========================================
if __name__ == "__main__":
    print("--- INICIANDO PRUEBA DE BLOQUES SSL (FÍSICA) ---")
    
    # Configuramos parámetros
    BATCH = 1
    BANDAS = 31
    H, W = 256, 256
    device = torch.device("cpu") # Usamos CPU para la prueba
    
    # 1. Creamos Tensores "Falsos" para simular la realidad
    print("\n1. Generando tensores de prueba...")
    # La cámara CASSI captura una imagen más ancha por la dispersión (W + Bandas - 1)
    cassi_real = torch.rand(BATCH, 1, H, W + BANDAS - 1) 
    rgb_real = torch.rand(BATCH, 3, H, W)
    mascara = torch.randint(0, 2, (1, 1, H, W)).float() # Máscara binaria (0s y 1s)
    
    # 2. Simulamos la "Adivinanza" de la Red Neuronal (El Cubo HSI predicho)
    hsi_predicho = torch.rand(BATCH, BANDAS, H, W)
    
    # 3. Instanciamos los Jueces (Funciones de Pérdida)
    juez_cassi = SSL_Grayscale_Loss(mask=mascara).to(device)
    juez_rgb = SSL_Color_Loss(device=device)
    
    print("\n2. Evaluando en el Juez CASSI (SSL-GB)...")
    loss_gb, cassi_sim = juez_cassi(hsi_predicho, cassi_real)
    print(f" -> Forma del CASSI simulado: {cassi_sim.shape}")
    print(f" -> MSE Loss Grayscale     : {loss_gb.item():.4f}")
    
    print("\n3. Evaluando en el Juez RGB (SSL-CB)...")
    loss_cb, rgb_sim = juez_rgb(hsi_predicho, rgb_real)
    print(f" -> Forma del RGB simulado : {rgb_sim.shape}")
    print(f" -> MSE Loss Color         : {loss_cb.item():.4f}")
    
    print("\n--- PRUEBA COMPLETADA CON ÉXITO ---")