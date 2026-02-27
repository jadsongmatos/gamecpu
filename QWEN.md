# Lavapipe Vibe - Projeto de Renderizador GPU

## Visão Geral do Projeto

**lavapipe_demo** é um renderizador experimental em Rust que utiliza a API **WGPU** para renderização via Vulkan, especificamente direcionado ao driver **Lavapipe** (renderizador Vulkan baseado em CPU da Mesa). O projeto implementa um pipeline de renderização **tile-based** com computação GPU para renderizar um modelo 3D de luminária (lamp).

### Arquitetura

O renderizador utiliza um pipeline baseado em **compute shaders** GLSL compilados para SPIR-V:

1. **Clear Compute** (`clear.comp`) - Limpa buffers de cor e profundidade
2. **Vertex Compute** (`vertex.comp`) - Transformação de vértices e projeção
3. **Setup Bin Compute** (`setup_bin.comp`) - Organização de triângulos por tile
4. **Raster Compute** (`raster.comp`) - Rasterização e interpolação por tile

### Tecnologias Principais

- **Linguagem:** Rust (edition 2021)
- **API Gráfica:** WGPU 24 (backend Vulkan)
- **Shaders:** GLSL 450 com SPIR-V
- **Driver Alvo:** Lavapipe/Llvmpipe (Vulkan CPU)
- **Formatos de Textura:** KTX2, Basis Universal

## Estrutura do Projeto

```
lavapipe_vibe/
├── Cargo.toml              # Configuração do projeto Rust
├── build.rs                # Build script que compila GLSL → SPIR-V
├── run.sh                  # Script de execução com variáveis de ambiente
├── src/
│   ├── main.rs             # Código principal do renderizador
│   ├── pbr.wgsl            # Shader PBR em WGSL (não utilizado atualmente)
│   ├── sincos_sin_q15.in   # Tabela de seno em ponto fixo Q15
│   └── sincos_cos_q15.in   # Tabela de cosseno em ponto fixo Q15
├── shaders/
│   ├── clear.comp          # Compute shader de limpeza
│   ├── vertex.comp         # Compute shader de transformação de vértices
│   ├── setup_bin.comp      # Compute shader de organização de tiles
│   └── raster.comp         # Compute shader de rasterização
├── lamp_int.json           # Metadados do modelo da luminária
├── lamp_int.bin            # Dados binários do modelo (vértices/índices)
├── *.ktx2                  # Texturas em formato KTX2
└── glTF-Sample-Assets/     # Assets de exemplo (submódulo git)
```

## Construindo e Executando

### Pré-requisitos

- **Rust** (cargo)
- **glslangValidator** (para compilar GLSL → SPIR-V)
- **Driver Lavapipe** instalado (`libvulkan-mesa`, `mesa-vulkan-drivers`)
- **Wayland** (`libwayland-client`)

### Comandos

```bash
# Build do projeto
cargo build

# Executar (usa o script com variáveis de ambiente necessárias)
./run.sh

# Ou executar diretamente com cargo
cargo run
```

### Variáveis de Ambiente Necessárias

O `run.sh` configura:

```bash
export LD_PRELOAD=/usr/lib/libwayland-client.so
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
```

Estas variáveis são necessárias para:
- Carregar a biblioteca Wayland corretamente
- Forçar o uso do driver Lavapipe (CPU)

## Detalhes de Implementação

### Sistema de Coordenadas e Matemática

- **Posições em ponto fixo Q16/Q15** para precisão em shaders
- **Tabelas de seno/cosseno** pré-calculadas (`sincos_*.in`)
- **Transformações:** Rotação Y, translação de câmera, projeção perspectiva

### Tile-Based Rendering

- **Tamanho do tile:** 8x8 pixels
- **Máximo de triângulos por tile:** 256
- **Pipeline:**
  1. Clear → 2. Vertex → 3. SetupBin → 4. Raster

### Formato de Vértices

Cada vértice contém 4 `u32`:
- `pos_xy`: Posição X e Y (int16 packed)
- `z`: Posição Z (int16)
- `uv`: Coordenadas de textura (uint16 packed)
- `pad`: Padding

### Buffers Principais

- **Globals Buffer:** Uniform buffer com parâmetros de renderização
- **Color/Depth Buffers:** Buffers de framebuffer em sistema de tiles
- **TileCounts/TileTriIds:** Estruturas para tile-based rendering

## Assets

O projeto carrega um modelo de luminária a partir de:
- `lamp_int.json`: Header com centro, escala, contagem de vértices/índices
- `lamp_int.bin`: Dados binários de vértices e índices
- Texturas KTX2 referenciadas no JSON

## Ferramentas Auxiliares

- `gltf2intmesh.py`: Script Python para converter glTF → formato interno
- `raw2png.py`: Script Python para converter screenshots RAW → PNG

## Observações de Desenvolvimento

### Warnings Conhecidos

O código possui warnings de deprecated da API WGPU 24:
- `wgpu::ImageCopyTexture` → `TexelCopyTextureInfo`
- `wgpu::ImageDataLayout` → `TexelCopyBufferLayout`

### Shader PBR

Existe um shader `pbr.wgsl` em WGSL que implementa PBR básico, mas atualmente não é utilizado no pipeline principal.

## Screenshot

O renderizador gera screenshots em formato RAW (`frame.raw`) que podem ser convertidas para PNG usando `raw2png.py`.
