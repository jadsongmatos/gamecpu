use std::borrow::Cow;
use std::sync::Arc;
use std::fs;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use serde_json;
use basis_universal::{Transcoder, TranscoderTextureFormat, TranscodeParameters};

const TILE_W: u32 = 8;
const TILE_H: u32 = 8;
const MAX_TRIS_PER_TILE: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Globals {
    width: u32,
    height: u32,
    pitch_pixels: u32,
    tiles_x: u32,

    tiles_y: u32,
    num_tiles: u32,
    clear_color_bgra: u32,
    _pad0: u32,

    cx2: i32,
    cy2: i32,
    focal_x_q16: i32,
    focal_y_q16: i32,

    sin_y_q15: i32,
    cos_y_q15: i32,
    cam_z_q16: i32,
    near_q16: i32,

    // Center and scale for decoding quantized positions
    center_x_q16: i32,
    center_y_q16: i32,
    center_z_q16: i32,
    scale_x_q16: i32,
    scale_y_q16: i32,
    scale_z_q16: i32,
    _pad1: u32,
    _pad2: u32,
}

fn aligned_pitch_pixels(width: u32) -> u32 {
    let align_pixels = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT / 4;
    ((width + align_pixels - 1) / align_pixels) * align_pixels
}

fn sincos_q15(step: u32) -> (i32, i32) {
    const SIN: [i16; 256] = include!("sincos_sin_q15.in");
    const COS: [i16; 256] = include!("sincos_cos_q15.in");

    let i = (step & 255) as usize;
    (SIN[i] as i32, COS[i] as i32)
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,

    globals: Globals,
    globals_buf: wgpu::Buffer,

    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    num_vertices: u32,
    num_tris: u32,

    vertex_screen: wgpu::Buffer,
    tri_setup: wgpu::Buffer,

    tile_counts: wgpu::Buffer,
    tile_tri_ids: wgpu::Buffer,

    color_buf: wgpu::Buffer,
    depth_buf: wgpu::Buffer,
    
    texture: wgpu::Texture,
    sampler: wgpu::Sampler,
    texture_view: wgpu::TextureView,

    clear_pipe: wgpu::ComputePipeline,
    vertex_pipe: wgpu::ComputePipeline,
    setup_bin_pipe: wgpu::ComputePipeline,
    raster_pipe: wgpu::ComputePipeline,

    clear_bg: wgpu::BindGroup,
    vertex_bg: wgpu::BindGroup,
    setup_bg: wgpu::BindGroup,
    raster_bg: wgpu::BindGroup,
}

fn create_lamp_procedural(device: &wgpu::Device, queue: &wgpu::Queue, color: [u8; 4]) -> (wgpu::Texture, wgpu::TextureView) {
    let width = 1024;
    let height = 1024;
    let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("LampProcedural"), size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm, usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[],
    });
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let mut c = color;
            // Add some procedural detail (noise/grid)
            if (x % 64 < 2) || (y % 64 < 2) {
                c[0] = c[0].saturating_add(20);
                c[1] = c[1].saturating_add(20);
                c[2] = c[2].saturating_add(20);
            }
            data.extend_from_slice(&c);
        }
    }
    queue.write_texture(wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &data, wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(width * 4), rows_per_image: Some(height) }, size);
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn create_gradient_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::Texture, wgpu::TextureView) {
    let width = 256;
    let height = 256;
    let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Gradient"), size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm, usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[],
    });
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = x as u8;
            let g = y as u8;
            let b = 128;
            data.extend_from_slice(&[r, g, b, 255]);
        }
    }
    queue.write_texture(wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &data, wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(width * 4), rows_per_image: Some(height) }, size);
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn create_checkerboard(device: &wgpu::Device, queue: &wgpu::Queue, c1: [u8; 4], c2: [u8; 4]) -> (wgpu::Texture, wgpu::TextureView) {
    let width = 256;
    let height = 256;
    let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Checkerboard"), size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm, usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[],
    });
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let check = ((x / 16) + (y / 16)) % 2 == 0;
            if check { data.extend_from_slice(&c1); }
            else { data.extend_from_slice(&c2); }
        }
    }
    queue.write_texture(wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &data, wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(width * 4), rows_per_image: Some(height) }, size);
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn load_spv_u32(bytes: &'static [u8]) -> Cow<'static, [u32]> {
    if (bytes.as_ptr() as usize) % 4 == 0 {
        Cow::Borrowed(bytemuck::cast_slice(bytes))
    } else {
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        Cow::Owned(words)
    }
}

impl Gpu {
    async fn new(window: Arc<winit::window::Window>) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .into_iter()
            .find(|a| {
                let info = a.get_info();
                info.device_type == wgpu::DeviceType::Cpu
                    && (info.name.to_lowercase().contains("lavapipe")
                        || info.name.to_lowercase().contains("llvmpipe"))
            })
            .ok_or_else(|| anyhow::anyhow!("Lavapipe adapter not found"))?;

        let features =
            wgpu::Features::SHADER_INT64 |
            wgpu::Features::SPIRV_SHADER_PASSTHROUGH |
            wgpu::Features::TEXTURE_COMPRESSION_BC;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: features,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats
            .iter()
            .copied()
            .find(|f| matches!(f, wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Bgra8Unorm))
            .unwrap_or(caps.formats[0]);

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Load lamp mesh
        let json_data = fs::read_to_string("lamp_int.json")?;
        let header: serde_json::Value = serde_json::from_str(&json_data)?;
        let bin_data = fs::read("lamp_int.bin")?;

        let vertices_u32_count = header["vertices_u32_count"].as_u64().unwrap() as usize;
        let indices_u32_count = header["indices_u32_count"].as_u64().unwrap() as usize;
        // Vertices are 4 u32s each (pos_xy, z, uv, pad)
        let num_vertices = vertices_u32_count as u32 / 4;
        let num_tris = (indices_u32_count / 3) as u32;

        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PackedVertices"),
            contents: &bin_data[0..vertices_u32_count * 4],
            usage: wgpu::BufferUsages::STORAGE,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Indices"),
            contents: &bin_data[vertices_u32_count * 4..],
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Vertex screen: now 8 ints per vertex (pos_x, pos_y, z, valid, u, v, pad, pad)
        let vertex_screen = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VertexScreenCache"),
            size: num_vertices as u64 * 8 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let tri_setup = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TriSetupCache"),
            size: num_tris as u64 * 128,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Load Texture
        let texture_uri = header["texture_uri"].as_str();
        let (texture, texture_view) = if let Some(uri) = texture_uri {
            let data = fs::read(uri).unwrap_or_else(|_| vec![]);
            if !data.is_empty() {
                // Determine color based on URI keywords
                let color = if uri.contains("glass") {
                    [100, 200, 255, 200] // Glassy Blue
                } else if uri.contains("grill") {
                    [50, 50, 50, 255] // Dark Grey Metal
                } else if uri.contains("bulb") {
                    [255, 255, 150, 255] // Warm Glow
                } else if uri.contains("hardware") || uri.contains("base") {
                    [180, 130, 50, 255] // Bronze/Gold
                } else {
                    [150, 150, 150, 255] // Steel
                };
                create_lamp_procedural(&device, &queue, color)
            } else {
                 create_lamp_procedural(&device, &queue, [255, 0, 255, 255])
            }
        } else {
             create_lamp_procedural(&device, &queue, [0, 0, 255, 255])
        };
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let center = header["center_q16"].as_array().unwrap();
        let scale = header["scale_q16"].as_array().unwrap();

        let globals = Globals {
            width: config.width, height: config.height, pitch_pixels: 0, tiles_x: 0, tiles_y: 0, num_tiles: 0,
            clear_color_bgra: 0xFF334C19, _pad0: 0, cx2: config.width as i32, cy2: config.height as i32,
            focal_x_q16: (config.width as i32) << 16, focal_y_q16: (config.height as i32) << 16,
            sin_y_q15: 0, cos_y_q15: 32767, cam_z_q16: (-2i32) << 16, near_q16: (1i32) << 16, // Zoom in
            center_x_q16: center[0].as_i64().unwrap() as i32,
            center_y_q16: center[1].as_i64().unwrap() as i32,
            center_z_q16: center[2].as_i64().unwrap() as i32,
            scale_x_q16: scale[0].as_i64().unwrap() as i32,
            scale_y_q16: scale[1].as_i64().unwrap() as i32,
            scale_z_q16: scale[2].as_i64().unwrap() as i32,
            _pad1: 0, _pad2: 0,
        };

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: std::mem::size_of::<Globals>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let tc = dummy_buffer(&device);
        let tti = dummy_buffer(&device);
        let cb = dummy_buffer(&device);
        let db = dummy_buffer(&device);
        let cp = dummy_compute(&device);
        let vp = dummy_compute(&device);
        let sp = dummy_compute(&device);
        let rp = dummy_compute(&device);
        let cbg = dummy_bg(&device);
        let vbg = dummy_bg(&device);
        let sbg = dummy_bg(&device);
        let rbg = dummy_bg(&device);

        let mut gpu = Self {
            device, queue, surface, config: config.clone(),
            globals,
            globals_buf,
            vbuf, ibuf, num_vertices, num_tris, vertex_screen, tri_setup,
            tile_counts: tc, tile_tri_ids: tti, color_buf: cb, depth_buf: db,
            texture, sampler, texture_view,
            clear_pipe: cp, vertex_pipe: vp, setup_bin_pipe: sp, raster_pipe: rp,
            clear_bg: cbg, vertex_bg: vbg, setup_bg: sbg, raster_bg: rbg,
        };

        gpu.rebuild_resolution_dependent();
        gpu.build_pipelines_and_bgs();
        Ok(gpu)
    }

    fn rebuild_resolution_dependent(&mut self) {
        let w = self.config.width.max(1);
        let h = self.config.height.max(1);
        let pitch = aligned_pitch_pixels(w);
        let tiles_x = (w + TILE_W - 1) / TILE_W;
        let tiles_y = (h + TILE_H - 1) / TILE_H;
        let num_tiles = tiles_x * tiles_y;

        self.globals.width = w; self.globals.height = h; self.globals.pitch_pixels = pitch;
        self.globals.tiles_x = tiles_x; self.globals.tiles_y = tiles_y; self.globals.num_tiles = num_tiles;
        self.globals.cx2 = w as i32; self.globals.cy2 = h as i32;
        self.globals.focal_x_q16 = (w as i32) << 16; self.globals.focal_y_q16 = (h as i32) << 16;

        self.color_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ColorBuf"), size: pitch as u64 * h as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        self.depth_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DepthBuf"), size: pitch as u64 * h as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        self.tile_counts = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TileCounts"), size: num_tiles as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        self.tile_tri_ids = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TileTriIds"), size: num_tiles as u64 * MAX_TRIS_PER_TILE as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
    }

    fn build_pipelines_and_bgs(&mut self) {
        let clear_spv = include_bytes!(concat!(env!("OUT_DIR"), "/clear.comp.spv"));
        let vertex_spv = include_bytes!(concat!(env!("OUT_DIR"), "/vertex.comp.spv"));
        let setup_spv = include_bytes!(concat!(env!("OUT_DIR"), "/setup_bin.comp.spv"));
        let raster_spv = include_bytes!(concat!(env!("OUT_DIR"), "/raster.comp.spv"));

        let clear_mod = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("clear_spv"), source: wgpu::ShaderSource::SpirV(load_spv_u32(clear_spv)) });
        let vertex_mod = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("vertex_spv"), source: wgpu::ShaderSource::SpirV(load_spv_u32(vertex_spv)) });
        let setup_mod = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("setup_spv"), source: wgpu::ShaderSource::SpirV(load_spv_u32(setup_spv)) });
        let raster_mod = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("raster_spv"), source: wgpu::ShaderSource::SpirV(load_spv_u32(raster_spv)) });

        let clear_bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clear_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        self.clear_pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("clear_pipe"), layout: Some(&self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&clear_bgl], push_constant_ranges: &[] })), module: &clear_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });

        let vertex_bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vertex_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        self.vertex_pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("vertex_pipe"), layout: Some(&self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&vertex_bgl], push_constant_ranges: &[] })), module: &vertex_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });

        let setup_bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("setup_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        self.setup_bin_pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("setup_bin_pipe"), layout: Some(&self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&setup_bgl], push_constant_ranges: &[] })), module: &setup_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });

        let raster_bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raster_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        self.raster_pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("raster_pipe"), layout: Some(&self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&raster_bgl], push_constant_ranges: &[] })), module: &raster_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });

        self.clear_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("clear_bg"), layout: &clear_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.globals_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: self.color_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: self.depth_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: self.tile_counts.as_entire_binding() }] });
        self.vertex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("vertex_bg"), layout: &vertex_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.globals_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: self.vbuf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: self.vertex_screen.as_entire_binding() }] });
        self.setup_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("setup_bg"), layout: &setup_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.globals_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: self.ibuf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: self.vertex_screen.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: self.tri_setup.as_entire_binding() }, wgpu::BindGroupEntry { binding: 4, resource: self.tile_counts.as_entire_binding() }, wgpu::BindGroupEntry { binding: 5, resource: self.tile_tri_ids.as_entire_binding() }] });
        self.raster_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("raster_bg"), layout: &raster_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: self.globals_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: self.tri_setup.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: self.tile_counts.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: self.tile_tri_ids.as_entire_binding() }, wgpu::BindGroupEntry { binding: 4, resource: self.color_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 5, resource: self.depth_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.texture_view) }, wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&self.sampler) }] });
    }

    fn resize(&mut self, new_w: u32, new_h: u32) {
        if new_w == 0 || new_h == 0 { return; }
        self.config.width = new_w; self.config.height = new_h;
        self.surface.configure(&self.device, &self.config);
        self.rebuild_resolution_dependent(); self.build_pipelines_and_bgs();
    }

    fn draw(&mut self, frame_id: u32) -> Result<()> {
        let (sin, cos) = sincos_q15(frame_id);
        self.globals.sin_y_q15 = sin; self.globals.cos_y_q15 = cos;
        self.queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&self.globals));
        let frame = self.surface.get_current_texture()?;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let pixels = self.globals.pitch_pixels * self.globals.height;
            let n = pixels.max(self.globals.num_tiles);
            let groups = (n + 255) / 256;
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("clear"), timestamp_writes: None });
            cpass.set_pipeline(&self.clear_pipe); cpass.set_bind_group(0, &self.clear_bg, &[]); cpass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let groups = (self.num_vertices + 255) / 256;
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("vertex"), timestamp_writes: None });
            cpass.set_pipeline(&self.vertex_pipe); cpass.set_bind_group(0, &self.vertex_bg, &[]); cpass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let groups = (self.num_tris + 255) / 256;
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("setup_bin"), timestamp_writes: None });
            cpass.set_pipeline(&self.setup_bin_pipe); cpass.set_bind_group(0, &self.setup_bg, &[]); cpass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("raster"), timestamp_writes: None });
            cpass.set_pipeline(&self.raster_pipe); cpass.set_bind_group(0, &self.raster_bg, &[]); cpass.dispatch_workgroups(self.globals.tiles_x, self.globals.tiles_y, 1);
        }
        let bytes_per_row = self.globals.pitch_pixels * 4;
        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer { buffer: &self.color_buf, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(bytes_per_row), rows_per_image: Some(self.globals.height) } },
            wgpu::ImageCopyTexture { texture: &frame.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::Extent3d { width: self.globals.width, height: self.globals.height, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

fn dummy_buffer(device: &wgpu::Device) -> wgpu::Buffer { device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }) }
fn dummy_compute(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed("@compute @workgroup_size(1) fn main() {}")) });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[] });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] })), module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
}
fn dummy_bg(device: &wgpu::Device) -> wgpu::BindGroup {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[] });
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[] })
}

fn main() -> Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    let window = Arc::new(WindowBuilder::new().with_title("Lavapipe - compute engine").with_inner_size(winit::dpi::LogicalSize::new(800, 600)).build(&event_loop)?);
    let mut gpu = pollster::block_on(Gpu::new(window.clone()))?;
    let mut frame_id: u32 = 0;
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::Resized(sz), .. } => gpu.resize(sz.width, sz.height),
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => { let _ = gpu.draw(frame_id); frame_id = frame_id.wrapping_add(1); }
            _ => {}
        }
    })?;
    Ok(())
}