import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. BLOQUES BASE (Los "ladrillos" de la red)
# =========================================================================
class BasicBlock(nn.Module):
    """Bloque Residual Estándar de HRNet"""
    def __init__(self, channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class HRModule(nn.Module):
    """
    Módulo de Fusión Multiescala: 
    Procesa las ramas en paralelo y luego cruza la información (Upsample / Downsample)
    """
    def __init__(self, num_branches, in_channels):
        super(HRModule, self).__init__()
        self.num_branches = num_branches
        self.in_channels = in_channels
        
        # Creamos los bloques de convolución para cada rama paralela
        self.branches = nn.ModuleList([
            BasicBlock(self.in_channels[i]) for i in range(self.num_branches)
        ])
        
        # Creamos las capas de fusión (flechas cruzadas en el diagrama de Xie)
        self.fusion_layers = nn.ModuleList()
        for i in range(self.num_branches):
            fusion_dest = nn.ModuleList()
            for j in range(self.num_branches):
                if j == i:
                    fusion_dest.append(nn.Identity()) # Misma resolución: no hace nada
                elif j < i:
                    # De Alta Resolución a Baja Resolución (Downsampling con Stride 2)
                    fusion_dest.append(nn.Sequential(
                        nn.Conv2d(self.in_channels[j], self.in_channels[i], kernel_size=3, stride=2**(i-j), padding=1, bias=False),
                        nn.BatchNorm2d(self.in_channels[i])
                    ))
                else:
                    # De Baja Resolución a Alta Resolución (Upsampling Bilineal + Conv 1x1)
                    fusion_dest.append(nn.Sequential(
                        nn.Conv2d(self.in_channels[j], self.in_channels[i], kernel_size=1, bias=False),
                        nn.BatchNorm2d(self.in_channels[i])
                    ))
            self.fusion_layers.append(fusion_dest)
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1. Pasamos la información por las ramas (horizontal)
        out_branches = [self.branches[i](x[i]) for i in range(self.num_branches)]
        
        # 2. Cruzamos la información (diagonal)
        out_fusion = []
        for i in range(self.num_branches):
            fusion_i = []
            for j in range(self.num_branches):
                if j == i:
                    fusion_i.append(out_branches[j])
                elif j < i:
                    fusion_i.append(self.fusion_layers[i][j](out_branches[j]))
                else:
                    # Escalamos el tensor pequeño para que coincida con el grande
                    upsampled = F.interpolate(out_branches[j], size=out_branches[i].shape[2:], mode='bilinear', align_corners=False)
                    fusion_i.append(self.fusion_layers[i][j](upsampled))
            
            # Sumamos toda la información cruzada y aplicamos ReLU
            out_fusion.append(self.relu(sum(fusion_i)))
            
        return out_fusion

# =========================================================================
# 2. LA RED PRINCIPAL: SIR-CNN (HRNet-W18)
# =========================================================================
class SIR_CNN(nn.Module):
    def __init__(self, num_bands=31):
        super(SIR_CNN, self).__init__()
        
        # Anchos de canal estándar para HRNet-W18
        self.channels = [18, 36, 72, 144] 
        
        # ---------------------------------------------------------
        # FASE 1: Ascenso Dimensional (Dimensional Ascension)
        # ---------------------------------------------------------
        # Expande 1 canal a 31 bandas usando un kernel de 1x1
        self.dimensional_ascension = nn.Sequential(
            nn.Conv2d(1, num_bands, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # Ajusta las 31 bandas al ancho de la primera rama (18 canales)
        self.layer0 = nn.Sequential(
            nn.Conv2d(num_bands, self.channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # ---------------------------------------------------------
        # FASE 2: La Cascada Multiescala (Parallel-multiscale net)
        # ---------------------------------------------------------
        # Transiciones para crear nuevas ramas más gruesas hacia abajo
        self.trans1 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, stride=2, padding=1)
        self.trans2 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, stride=2, padding=1)
        self.trans3 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=3, stride=2, padding=1)
        
        # Las 4 Etapas del Diagrama
        self.stage1 = HRModule(num_branches=1, in_channels=[self.channels[0]])
        self.stage2 = HRModule(num_branches=2, in_channels=[self.channels[0], self.channels[1]])
        self.stage3 = HRModule(num_branches=3, in_channels=[self.channels[0], self.channels[1], self.channels[2]])
        self.stage4 = HRModule(num_branches=4, in_channels=[self.channels[0], self.channels[1], self.channels[2], self.channels[3]])

        # ---------------------------------------------------------
        # FASE 3: La Cola (Reconstrucción del Cubo HSI)
        # ---------------------------------------------------------
        # Concatenaremos todas las ramas (18+36+72+144 = 270 canales)
        total_channels = sum(self.channels)
        self.head = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels, num_bands, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # 1. Ascenso Dimensional
        x = self.dimensional_ascension(x)
        x0 = self.layer0(x)
        
        # 2. Etapa 1 (1 Rama)
        x_s1 = self.stage1([x0])
        
        # 3. Etapa 2 (2 Ramas)
        x1 = self.trans1(x_s1[0])
        x_s2 = self.stage2([x_s1[0], x1])
        
        # 4. Etapa 3 (3 Ramas)
        x2 = self.trans2(x_s2[1])
        x_s3 = self.stage3([x_s2[0], x_s2[1], x2])
        
        # 5. Etapa 4 (4 Ramas)
        x3 = self.trans3(x_s3[2])
        x_s4 = self.stage4([x_s3[0], x_s3[1], x_s3[2], x3])
        
        # 6. Fusión Final (Upsampling de todas las ramas a 256x256)
        h0 = x_s4[0]
        h1 = F.interpolate(x_s4[1], size=h0.shape[2:], mode='bilinear', align_corners=False)
        h2 = F.interpolate(x_s4[2], size=h0.shape[2:], mode='bilinear', align_corners=False)
        h3 = F.interpolate(x_s4[3], size=h0.shape[2:], mode='bilinear', align_corners=False)
        
        out = torch.cat([h0, h1, h2, h3], dim=1) # Las pegamos todas juntas
        out = self.head(out)                     # Volvemos a las 31 bandas puras
        
        return out

# =========================================================================
# 3. ESCÁNER DE RENDIMIENTO Y ZONA DE PRUEBA
# =========================================================================
def analizar_modelo(modelo, entrada):
    total_params = sum(p.numel() for p in modelo.parameters())
    trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    macs_list = []

    def hook_conv(module, input, output):
        out_c, in_c = module.out_channels, module.in_channels
        k_h, k_w = module.kernel_size
        groups = module.groups
        out_h, out_w = output.shape[2], output.shape[3]
        macs = (in_c // groups) * k_h * k_w * out_c * out_h * out_w
        macs_list.append(macs)

    hooks = [capa.register_forward_hook(hook_conv) for capa in modelo.modules() if isinstance(capa, nn.Conv2d)]
    
    modelo.eval()
    with torch.no_grad():
        modelo(entrada)
        
    for h in hooks: h.remove()
    
    total_flops = sum(macs_list) * 2 
    return total_params, trainable_params, total_flops

if __name__ == "__main__":
    print("--- INICIANDO ESCÁNER SIR-CNN (HRNet-W18) ---")
    
    # Simulamos la imagen CASSI comprimida de 1 canal y 256x256
    dummy_input = torch.randn(1, 1, 256, 256)
    
    modelo = SIR_CNN(num_bands=31)
    
    total, entrenables, flops = analizar_modelo(modelo, dummy_input)
    peso_mb = (total * 4) / (1024 ** 2)
    giga_flops = flops / (10**9)
    
    print(f"Arquitectura             : SIR-CNN (Basada en HRNet-W18)")
    print(f"Entrada (Cámara Mono)    : {dummy_input.shape}")
    print(f"Salida (Cubo Reconstruido): [1, 31, 256, 256]")
    print(f"-"*50)
    print(f"Parámetros Totales       : {total:,}")
    print(f"Peso Estimado del Modelo : {peso_mb:.2f} MB")
    print(f"Rendimiento (GFLOPs)     : {giga_flops:.4f} GFLOPs")
    print(f"-"*50)
    print("¡Arquitectura lista para el entrenamiento Físico!")