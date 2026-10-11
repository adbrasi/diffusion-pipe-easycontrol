source /venv/main/bin/activate
cd /workspace/models
for f in diffusion_models/krea2_raw_fp8_scaled.safetensors loras/krea2_turbo_lora_rank_64_bf16.safetensors vae/qwen_image_vae.safetensors text_encoders/qwen3vl_4b_bf16.safetensors diffusion_models/krea2_turbo_fp8_scaled.safetensors; do
  hf download Comfy-Org/Krea-2 $f --local-dir /workspace/models/krea2 || hf download Comfy-Org/Krea-2 split_files/$f --local-dir /workspace/models/krea2
done
echo DONE_MODELS
