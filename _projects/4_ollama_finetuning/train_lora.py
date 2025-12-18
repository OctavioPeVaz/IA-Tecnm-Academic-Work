import os
import json
import random
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model

# ==========================================
# ⚙️ CONFIGURACIÓN CENTRAL
# ==========================================
class Config:
    # Modelo Base (Usamos el ID de HuggingFace)
    base_model = "microsoft/Phi-3.5-mini-instruct" 
    
    # Archivos de datos
    source_file = "./datasets/tutor_algoritmos/algoritmos_fixed.jsonl"  # Archivo fuente
    train_file = "./datasets/tutor_algoritmos/train.jsonl"        # Se creará automático
    val_file = "./datasets/tutor_algoritmos/val.jsonl"            # Se creará automático
    
    # Directorio de salida
    out_dir = "./trained_models/lora/tutor_algoritmos_lora"
    
    # Parámetros de Entrenamiento
    epochs = 3              
    lr = 2e-4               
    batch_size = 1          
    grad_accum = 8       
    max_seq_len = 512      
    seed = 42
    
    # Parámetros LoRA
    lora_r = 16             
    lora_alpha = 32         
    lora_dropout = 0.05

# ==========================================
# 🛠️ FUNCIONES DE UTILIDAD
# ==========================================

def prepare_data():
    """
    Divide el archivo fuente en train y val si no existen.
    """
    # Crear directorios si no existen
    os.makedirs(os.path.dirname(Config.train_file), exist_ok=True)

    if os.path.exists(Config.train_file) and os.path.exists(Config.val_file):
        print(f"[INFO] Archivos de entrenamiento encontrados en {os.path.dirname(Config.train_file)}")
        return

    print(f"[INFO] Generando splits desde {Config.source_file}...")
    
    if not os.path.exists(Config.source_file):
        raise FileNotFoundError(f"❌ No se encuentra el archivo fuente: {Config.source_file}")

    with open(Config.source_file, 'r', encoding='utf-8') as f:
        # Leemos línea por línea (JSONL)
        data = [json.loads(line) for line in f if line.strip()]
    
    print(f"[INFO] Total registros encontrados: {len(data)}")
    
    # Mezclar y dividir
    random.seed(Config.seed)
    random.shuffle(data)
    
    split_idx = int(len(data) * 0.9) # 90% Train, 10% Val
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Guardar
    def save_jsonl(path, entries):
        with open(path, 'w', encoding='utf-8') as f:
            for entry in entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
                
    save_jsonl(Config.train_file, train_data)
    save_jsonl(Config.val_file, val_data)
    print(f"[OK] Creados: Train ({len(train_data)}) | Val ({len(val_data)})")

def detect_device_and_dtype():
    if torch.cuda.is_available():
        print("[INFO] 🚀 Modo: NVIDIA CUDA (GPU)")
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        print("[INFO] 🍎 Modo: Apple Metal (MPS)")
        return "mps", torch.float16
    else:
        print("[INFO] 🐌 Modo: CPU (Lento)")
        return "cpu", torch.float32

def suggest_target_modules(model) -> List[str]:
    module_names = set([name for name, _ in model.named_modules()])
    candidates = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "q_proj", "k_proj", "v_proj"]
    targets = [c for c in candidates if any(c in m for m in module_names)]
    return list(set(targets)) if targets else ["q_proj", "k_proj", "v_proj", "o_proj"]

# ==========================================
# 🚀 MAIN (EJECUCIÓN)
# ==========================================

def main():
    # 1. Preparar datos (Auto-split)
    prepare_data()

    # 2. Configurar Hardware
    device_name, dtype = detect_device_and_dtype()
    
    print(f"[INFO] Cargando modelo: {Config.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"[INFO] Cargando modelo en {device_name} (Forzado)...")
    model = AutoModelForCausalLM.from_pretrained(
        Config.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,  #  IMPORTANTE: Desactivamos "auto" para evitar el error 'meta device'
        attn_implementation="eager"
    )
    model.config.use_cache = False
    # Movemos el modelo manualmente a la GPU
    if device_name == "cuda":
        model.to("cuda")
    elif device_name == "mps":
        model.to("mps")

    # 3. Configurar LoRA
    target_modules = suggest_target_modules(model)
    print(f"[INFO] Módulos LoRA: {target_modules}")
    
    lora_config = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        lora_dropout=Config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Cargar y Procesar Datos
    dataset = load_dataset("json", data_files={"train": Config.train_file, "validation": Config.val_file})

    def build_prompt(instruction: str, context: str = "") -> str:
        # Prompt estilo Alpaca/Instruct
        if context:
            return f"### Instrucción:\n{instruction}\n\n### Contexto:\n{context}\n\n### Respuesta:\n"
        return f"### Instrucción:\n{instruction}\n\n### Respuesta:\n"

    def preprocess(example: Dict):
        instr = example.get("instruction", "") or ""
        inp = example.get("input", "") or ""     # Usamos input si existe
        out = example.get("output", "") or ""    # Usamos output (no response)

        prompt_text = build_prompt(instr, inp)
        full_text = prompt_text + out + tokenizer.eos_token

        # Tokenizamos prompt y full para crear máscaras
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=Config.max_seq_len, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, truncation=True, max_length=Config.max_seq_len, add_special_tokens=False)["input_ids"]

        # Labels: -100 en el prompt para no calcular error ahí
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        
        # Ajuste de longitud por seguridad
        if len(labels) > len(full_ids):
            labels = labels[:len(full_ids)]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels
        }

    print("[INFO] Tokenizando dataset...")
    tokenized = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

    # 5. Configurar Entrenamiento
    training_args = TrainingArguments(
        output_dir=Config.out_dir,
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=Config.grad_accum,
        num_train_epochs=Config.epochs,
        learning_rate=Config.lr,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_total_limit=2,
        fp16=(device_name == "cuda"), # Solo FP16 si es NVIDIA
        bf16=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # 6. Ejecutar
    print("🔥🔥🔥 INICIANDO FINE-TUNING 🔥🔥🔥")
    trainer.train()

    print(f"[DONE] Guardando adaptadores en {Config.out_dir}")
    model.save_pretrained(Config.out_dir)
    tokenizer.save_pretrained(Config.out_dir)

if __name__ == "__main__":
    main()