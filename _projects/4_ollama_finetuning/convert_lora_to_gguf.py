"""Convierte un LoRA (PEFT) entrenado a un modelo GGUF fusionado."""

import argparse
import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import hf_hub_download  # Necesario para el fix

# Nombre de la carpeta temporal para evitar errores de memoria (Offloading)
OFFLOAD_DIR = "./offload_convert_temp"

def ensure_llama_cpp_exists(llama_cpp_dir: Path):
    if llama_cpp_dir.exists():
        return
    print(f"[INFO] Clonando llama.cpp en {llama_cpp_dir} ...")
    try:
        url = "https://github.com/ggerganov/llama.cpp.git"
        subprocess.run(["git", "clone", "--depth", "1", url, str(llama_cpp_dir)], check=True)
        print("[INFO] llama.cpp clonado.")
    except FileNotFoundError:
        print("[ERROR] No se encontró el comando 'git'. Instala Git o descarga llama.cpp manualmente.")
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser(description="Fusiona LoRA + base y convierte a GGUF para Ollama (Modo Seguro).")
    ap.add_argument("--base", required=True, help="Modelo base HF (nombre o ruta local)")
    ap.add_argument("--lora", required=True, help="Ruta del LoRA entrenado")
    ap.add_argument("--out", required=True, help="Ruta de salida del GGUF")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Precisión")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Limpieza preventiva de offload
    if os.path.exists(OFFLOAD_DIR):
        shutil.rmtree(OFFLOAD_DIR)
    os.makedirs(OFFLOAD_DIR, exist_ok=True)

    # 1. Cargar Tokenizer
    print("🔹 Cargando tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.lora, trust_remote_code=True)
    except:
        print("⚠️ No se encontró tokenizer en el LoRA, cargando del base...")
        tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    # 2. Cargar Modelo Base con OFFLOAD
    print("🔹 Cargando modelo base (Modo Seguro con Offload)...")
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, 
        torch_dtype=torch_dtype,
        device_map="auto",           
        low_cpu_mem_usage=True,      
        offload_folder=OFFLOAD_DIR,  
        offload_state_dict=True,
        trust_remote_code=True       
    )

    # 3. Cargar LoRA y Fusionar
    print(f"🔹 Cargando LoRA desde {args.lora}...")
    model = PeftModel.from_pretrained(
        base_model, 
        args.lora,
        offload_folder=OFFLOAD_DIR
    )

    print("🔹 Fusionando LoRA con el modelo base (Merge & Unload)...")
    model = model.merge_and_unload()

    # 4. Guardar fusión temporalmente
    tmp_dir = Path(tempfile.mkdtemp(prefix="merged_hf_"))
    print(f"🔹 Guardando modelo HF fusionado en {tmp_dir} ...")
    model.save_pretrained(tmp_dir, safe_serialization=True)
    tokenizer.save_pretrained(tmp_dir)

    # ==================== FIX AUTOMÁTICO ====================
    # Phi-3.5 necesita 'tokenizer.model' físico, pero save_pretrained no siempre lo crea.
    # Lo descargamos o copiamos manualmente a la carpeta temporal.
    print("🔹 Verificando 'tokenizer.model' para llama.cpp...")
    dest_tokenizer = tmp_dir / "tokenizer.model"
    
    if not dest_tokenizer.exists():
        try:
            print(f"   ⚠️ No existe tokenizer.model en {tmp_dir}. Intentando recuperar...")
            
            # Si el modelo base es una ruta local, buscamos ahí
            if os.path.isdir(args.base):
                src_local = Path(args.base) / "tokenizer.model"
                if src_local.exists():
                    print(f"   📂 Copiando desde ruta local: {src_local}")
                    shutil.copy(src_local, dest_tokenizer)
            else:
                # Si es un ID de HuggingFace, lo descargamos a la carpeta temporal
                print(f"   ⬇️ Descargando de HuggingFace ({args.base})...")
                file_path = hf_hub_download(repo_id=args.base, filename="tokenizer.model")
                shutil.copy(file_path, dest_tokenizer)
                print("   ✅ Archivo tokenizer.model descargado y colocado.")
                
        except Exception as e:
            print(f"   ❌ ERROR CRÍTICO: No se pudo conseguir tokenizer.model: {e}")
            print("   (La conversión fallará si este archivo falta)")
    # ========================================================

    # Liberar memoria de Python antes de llamar a llama.cpp
    del model
    del base_model
    torch.cuda.empty_cache()

    # 5. Preparar llama.cpp
    script_dir = Path(__file__).resolve().parent
    llama_cpp_dir = script_dir / "llama.cpp"
    
    ensure_llama_cpp_exists(llama_cpp_dir)
    
    # Instalar dependencias de llama.cpp si faltan
    print("🔹 Verificando dependencias de llama.cpp...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(llama_cpp_dir / "requirements.txt")], check=False)

    candidates = [
        llama_cpp_dir / "convert_hf_to_gguf.py",
        llama_cpp_dir / "convert-hf-to-gguf.py",
    ]
    convert_script = None
    for c in candidates:
        if c.exists():
            convert_script = c
            break
            
    if convert_script is None:
        print(f"[ERROR] No se encontró convert_hf_to_gguf.py en {llama_cpp_dir}")
        sys.exit(1)

    # 6. Ejecutar Conversión
    outtype = "f16" if args.dtype == "float16" else "f32"
    print("🔹 Ejecutando conversión a GGUF...")
    
    cmd = [
        sys.executable,
        str(convert_script),
        str(tmp_dir),
        "--outfile", str(out_path),
        "--outtype", outtype,
    ]
    
    subprocess.run(cmd, check=True)

    # Limpieza final
    print("🧹 Limpiando archivos temporales...")
    try:
        shutil.rmtree(tmp_dir)
        if os.path.exists(OFFLOAD_DIR):
            shutil.rmtree(OFFLOAD_DIR)
    except Exception as e:
        print(f"⚠️ Nota: No se pudo borrar algún temporal ({e}), puedes borrarlos manualmente.")

    print(f"✅ ¡LISTO! Modelo guardado en: {out_path}")

if __name__ == "__main__":
    main()