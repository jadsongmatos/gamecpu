#!/usr/bin/env python3
import argparse
import base64
import json
import os
import shutil
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

COMPONENT_DTYPE = {
    5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4),
}
TYPE_COMPONENTS = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
}

def load_gltf(path: str) -> Tuple[Dict[str, Any], List[bytes], str]:
    base_dir = os.path.dirname(os.path.abspath(path))
    if path.lower().endswith(".glb"):
        with open(path, "rb") as f:
            data = f.read()
        magic, version, length = struct.unpack_from("<4sII", data, 0)
        offset = 12
        json_chunk = None; bin_chunk = None
        while offset + 8 <= len(data):
            chunk_len, chunk_type = struct.unpack_from("<I4s", data, offset)
            offset += 8
            chunk_data = data[offset:offset + chunk_len]
            offset += chunk_len
            if chunk_type == b"JSON": json_chunk = chunk_data
            elif chunk_type == b"BIN\0": bin_chunk = chunk_data
        gltf = json.loads(json_chunk.decode("utf-8"))
        buffers = [bin_chunk] if bin_chunk else []
        return gltf, buffers, base_dir
    with open(path, "rb") as f:
        gltf = json.loads(f.read().decode("utf-8"))
    bufs = []
    for b in gltf.get("buffers", []):
        uri = b.get("uri")
        if uri.startswith("data:"): bufs.append(base64.b64decode(uri.split(",", 1)[1]))
        else: bufs.append(open(os.path.join(base_dir, uri), "rb").read())
    return gltf, bufs, base_dir

def accessor_read(gltf, buffers, acc_idx):
    acc = gltf["accessors"][acc_idx]; bv = gltf["bufferViews"][acc["bufferView"]]
    comp_type = acc["componentType"]; gltf_type = acc["type"]; count = acc["count"]
    ncomp = TYPE_COMPONENTS[gltf_type]; fmt_char, comp_size = COMPONENT_DTYPE[comp_type]
    buf = buffers[bv["buffer"]]; start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride", ncomp * comp_size)
    raw = memoryview(buf)[start:start + count * stride]
    if np:
        dtype = {5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}[comp_type]
        if stride == ncomp * comp_size:
            return np.frombuffer(raw, dtype=dtype).reshape((count, ncomp))
        else:
            out = np.empty((count, ncomp), dtype=dtype)
            for i in range(count): out[i] = np.frombuffer(raw[i*stride:i*stride+ncomp*comp_size], dtype=dtype)
            return out
    else:
        out = []
        unpack = struct.Struct("<" + fmt_char * ncomp).unpack_from
        for i in range(count): out.append(unpack(raw, i * stride))
    return out

def to_q16(x): return int(round(x * 65536.0))

def convert_gltf(in_path, out_prefix):
    gltf, buffers, _ = load_gltf(in_path)
    all_pos, all_uv, prim_data = [], [], []
    for mi, mesh in enumerate(gltf["meshes"]):
        for pi, prim in enumerate(mesh["primitives"]):
            pos = accessor_read(gltf, buffers, prim["attributes"]["POSITION"])
            uv = accessor_read(gltf, buffers, prim["attributes"].get("TEXCOORD_0", -1)) if "TEXCOORD_0" in prim["attributes"] else np.zeros((len(pos), 2))
            idx = accessor_read(gltf, buffers, prim["indices"]) if "indices" in prim else np.arange(len(pos))
            all_pos.append(pos); all_uv.append(uv)
            prim_data.append((mi, pi, pos, uv, idx, prim.get("material")))

    cat_pos = np.concatenate(all_pos); pmin, pmax = cat_pos.min(axis=0), cat_pos.max(axis=0)
    center, extent = (pmin + pmax)*0.5, (pmax - pmin)
    max_extent = extent.max()
    scale_val = max_extent * 0.5 / 32767.0 if max_extent > 0 else 1.0
    center_q16 = [to_q16(float(c)) for c in center]; scale_q16 = [to_q16(float(scale_val))]*3
    scale = np.array([scale_val, scale_val, scale_val])

    packed_v, packed_i, draws = [], [], []
    for (mi, pi, pos, uv, idx, mat) in prim_data:
        base_v = len(packed_v) // 4 # agora 4 u32 por vert
        q = np.round((pos - center) / scale).astype(np.int16)
        # UVs em u16 (0.0-1.0 -> 0-65535)
        uv_u16 = np.clip(np.round(uv * 65535.0), 0, 65535).astype(np.uint16)
        for i in range(len(pos)):
            # v0: [y:i16 | x:i16], v1: [pad:i16 | z:i16]
            # v2: [v:u16 | u:u16]
            # v3: pad
            xy = (int(np.uint16(q[i,0]))) | (int(np.uint16(q[i,1])) << 16)
            z_ = (int(np.uint16(q[i,2])))
            uv_ = (int(uv_u16[i,0])) | (int(uv_u16[i,1]) << 16)
            packed_v.extend([xy, z_, uv_, 0])
        draws.append({"first_index": len(packed_i), "index_count": len(idx), "base_vertex": base_v, "material": mat})
        if np is not None and isinstance(idx, np.ndarray):
            idx = idx.flatten()
        packed_i.extend((idx + base_v).astype(np.uint32).tolist() if np is not None else [i[0] + base_v for i in idx])

    with open(out_prefix + ".bin", "wb") as f:
        f.write(struct.pack(f"<{len(packed_v)}I", *packed_v))
        f.write(struct.pack(f"<{len(packed_i)}I", *packed_i))

    texture_uri = None
    if gltf.get("materials"):
        mat = gltf["materials"][0]
        if "pbrMetallicRoughness" in mat:
            pbr = mat["pbrMetallicRoughness"]
            if "baseColorTexture" in pbr:
                tex_idx = pbr["baseColorTexture"]["index"]
                tex_def = gltf["textures"][tex_idx]
                if "source" in tex_def:
                    img_idx = tex_def["source"]
                elif "extensions" in tex_def and "KHR_texture_basisu" in tex_def["extensions"]:
                    img_idx = tex_def["extensions"]["KHR_texture_basisu"]["source"]
                else:
                    img_idx = None
                
                if img_idx is not None:
                    uri = gltf["images"][img_idx]["uri"]
                    # Copy texture
                    src_tex = os.path.join(os.path.dirname(in_path), uri)
                    dst_tex = os.path.join(os.path.dirname(os.path.abspath(out_prefix)), os.path.basename(uri))
                    if os.path.exists(src_tex):
                        shutil.copy(src_tex, dst_tex)
                        texture_uri = os.path.basename(uri)
                    else:
                        print(f"Warning: Texture file not found: {src_tex}")

    with open(out_prefix + ".json", "w") as f:
        json.dump({"center_q16": center_q16, "scale_q16": scale_q16, "vertices_u32_count": len(packed_v), "indices_u32_count": len(packed_i), "draws": draws, "texture_uri": texture_uri}, f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("input"); ap.add_argument("output_prefix")
    args = ap.parse_args(); convert_gltf(args.input, args.output_prefix)