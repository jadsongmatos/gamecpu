#!/bin/bash
# Script para rodar o demo Lavapipe com as variáveis de ambiente necessárias

# Força o carregamento da libwayland-client para evitar erro de "undefined symbol" no driver lavapipe
export LD_PRELOAD=/usr/lib/libwayland-client.so

# Força o loader do Vulkan a usar apenas o driver Lavapipe (CPU)
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

echo "Iniciando Demo Lavapipe (Rust + WGPU)..."
cargo run
