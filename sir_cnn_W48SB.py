import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. MÓDULO SHIFT-BACK (Reversión de Dispersión)
# =========================================================================
class ShiftBack(nn.Module):
    """
    Toma la medición CASSI dispersa [B, 1, 256, 286] y recorta 31 ventanas 
    desplazadas para pre-alinear el cubo a [B, 31, 256, 256].
    """
    def __init__(self, step=1):
        super(ShiftBack, self).__init__()
        self.step = step

    def forward(self, x, num_bands=31):
        # x shape: [B, 1, H, W_disp]
        B, C, H, W_disp = x.shape
        W = W_disp - (num_bands - 1) * self.step # 286 - 30 = 256
        
        cubo_alineado = []
        for i in range(num_bands):
            # Recortamos la ventana exacta donde cayó esta longitud de onda
            cubo_alineado.append(x[:, :, :, i : i + W])
            
        return torch.cat(cubo_alineado, dim=1) # Salida natural: [B, 31, 256, 256]

# =========================================================================
# 2. BLOQUES BASE (HRNet con Reflect Padding)
# =========================================================================
class BasicBlock(nn.Module):
    def __init__(self, channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class HRModule(nn.Module):
    def __init__(self, num_branches, in_channels):
        super(HRModule, self).__init__()
        self.num_branches = num_branches
        self.in_channels = in_channels
        
        self.branches = nn.ModuleList([
            BasicBlock(self.in_channels[i]) for i in range(self.num_branches)
        ])
        
        self.fusion_layers = nn.ModuleList()
        for i in range(self.num_branches):
            fusion_dest = nn.ModuleList()
            for j in range(self.num_branches):
                if j == i:
                    fusion_dest.append(nn.Identity()) 
                elif j < i:
                    fusion_dest.append(nn.Sequential(
                        nn.Conv2d(self.in_channels[j], self.in_channels[i], kernel_size=3, stride=2**(i-j), padding=1, padding_mode='reflect', bias=False),
                        nn.BatchNorm2d(self.in_channels[i])
                    ))
                else:
                    fusion_dest.append(nn.Sequential(
                        nn.Conv2d(self.in_channels[j], self.in_channels[i], kernel_size=1, bias=False),
                        nn.BatchNorm2d(self.in_channels[i])
                    ))
            self.fusion_layers.append(fusion_dest)
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_branches = [self.branches[i](x[i]) for i in range(self.num_branches)]
        out_fusion = []
        for i in range(self.num_branches):
            fusion_i = []
            for j in range(self.num_branches):
                if j == i:
                    fusion_i.append(out_branches[j])
                elif j < i:
                    fusion_i.append(self.fusion_layers[i][j](out_branches[j]))
                else:
                    upsampled = F.interpolate(out_branches[j], size=out_branches[i].shape[2:], mode='bilinear', align_corners=False)
                    fusion_i.append(self.fusion_layers[i][j](upsampled))
            
            out_fusion.append(self.relu(sum(fusion_i)))
        return out_fusion

# =========================================================================
# 3. LA RED PRINCIPAL: SIR-CNN (HRNet-W48 + ShiftBack)
# =========================================================================
class SIR_CNN(nn.Module):
    def __init__(self, num_bands=31):
        super(SIR_CNN, self).__init__()
        self.channels = [48, 96, 192, 384] 
        
        # Inyectamos el ShiftBack en la entrada
        self.shift_back = ShiftBack()
        
        # Layer0 ahora recibe 31 canales del ShiftBack
        self.layer0 = nn.Sequential(
            nn.Conv2d(num_bands, self.channels[0], kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True)
        )
        
        self.trans1 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.trans2 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.trans3 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        
        self.stage1 = HRModule(num_branches=1, in_channels=[self.channels[0]])
        self.stage2 = HRModule(num_branches=2, in_channels=[self.channels[0], self.channels[1]])
        self.stage3 = HRModule(num_branches=3, in_channels=[self.channels[0], self.channels[1], self.channels[2]])
        self.stage4 = HRModule(num_branches=4, in_channels=[self.channels[0], self.channels[1], self.channels[2], self.channels[3]])

        total_channels = sum(self.channels)
        self.head = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels, num_bands, kernel_size=3, padding=1, padding_mode='reflect')
        )

    def forward(self, x):
        # 1. Alineación espacial inicial (286 -> 256)
        x = self.shift_back(x) 
        
        # 2. Procesamiento puro a 256x256
        x0 = self.layer0(x)
        x_s1 = self.stage1([x0])
        
        x1 = self.trans1(x_s1[0])
        x_s2 = self.stage2([x_s1[0], x1])
        
        x2 = self.trans2(x_s2[1])
        x_s3 = self.stage3([x_s2[0], x_s2[1], x2])
        
        x3 = self.trans3(x_s3[2])
        x_s4 = self.stage4([x_s3[0], x_s3[1], x_s3[2], x3])
        
        # 3. Fusión final multiescala
        h0 = x_s4[0]
        h1 = F.interpolate(x_s4[1], size=h0.shape[2:], mode='bilinear', align_corners=False)
        h2 = F.interpolate(x_s4[2], size=h0.shape[2:], mode='bilinear', align_corners=False)
        h3 = F.interpolate(x_s4[3], size=h0.shape[2:], mode='bilinear', align_corners=False)
        
        out = torch.cat([h0, h1, h2, h3], dim=1) 
        out = self.head(out)                     
        
        return out # Salida final perfecta: [Batch, 31, 256, 256]

# =========================================================================
# 4. ESCÁNER DE RENDIMIENTO 
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
    print("--- INICIANDO ESCÁNER SIR-CNN (HRNet-W48 + ShiftBack) ---")
    
    # ¡ATENCIÓN! La entrada de prueba AHORA ES 286 de ancho.
    dummy_input = torch.randn(1, 1, 256, 286) 
    
    modelo = SIR_CNN(num_bands=31)
    
    total, entrenables, flops = analizar_modelo(modelo, dummy_input)
    peso_mb = (total * 4) / (1024 ** 2)
    giga_flops = flops / (10**9)
    
    print(f"Arquitectura             : SIR-CNN (HRNet-W48 + ShiftBack)")
    print(f"Entrada (CASSI Dispersa) : {dummy_input.shape}")
    print(f"Salida (Cubo Alineado)   : [1, 31, 256, 256]")
    print(f"-"*50)
    print(f"Parámetros Totales       : {total:,}")
    print(f"Peso Estimado del Modelo : {peso_mb:.2f} MB")
    print(f"Rendimiento (GFLOPs)     : {giga_flops:.4f} GFLOPs")
    print(f"-"*50)