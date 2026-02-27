use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = PathBuf::from("shaders");

    let shaders = ["clear.comp", "vertex.comp", "setup_bin.comp", "raster.comp"];

    for name in shaders {
        let path = shader_dir.join(name);
        println!("cargo:rerun-if-changed={}", path.display());

        let out_path = out_dir.join(format!("{name}.spv"));
        
        let status = Command::new("glslangValidator")
            .arg("-V") // Create SPIR-V
            .arg("-o")
            .arg(&out_path)
            .arg(&path)
            .status()
            .expect("failed to execute glslangValidator");

        if !status.success() {
            panic!("glslangValidator failed for {}", name);
        }
    }
}