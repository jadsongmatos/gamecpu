struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.world_position = model.position;
    out.world_normal = model.normal;
    out.tex_coords = model.tex_coords;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    return out;
}

@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;
@group(1) @binding(2) var t_normal: texture_2d<f32>;
@group(1) @binding(3) var s_normal: sampler;
@group(1) @binding(4) var t_arm: texture_2d<f32>; // Ambient, Roughness, Metallic
@group(1) @binding(5) var s_arm: sampler;
@group(1) @binding(6) var t_emissive: texture_2d<f32>;
@group(1) @binding(7) var s_emissive: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_color = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let normal_map = textureSample(t_normal, s_normal, in.tex_coords).rgb * 2.0 - 1.0;
    let arm = textureSample(t_arm, s_arm, in.tex_coords);
    let emissive = textureSample(t_emissive, s_emissive, in.tex_coords);

    // Simplified lighting
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let N = normalize(in.world_normal); // Should use TBN if using normal map
    let diffuse = max(dot(N, light_dir), 0.1);
    
    let color = base_color.rgb * diffuse + emissive.rgb;
    
    return vec4<f32>(color, base_color.a);
}
