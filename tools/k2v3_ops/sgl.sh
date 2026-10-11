cd /workspace
export PACK_RUSH_DEBUG_DIR=${PACK_RUSH_DEBUG_DIR:-}
export SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_pack_rush PACK_RUSH_UPSCALE_MODEL=/workspace/models/esrgan/realesr-animevideov3.pth HF_HUB_OFFLINE=1
exec /workspace/env-sgl/bin/sglang serve --model-type diffusion --model-path /workspace/models/circlestone-labs/Anima-Base-v1.0-Diffusers --pipeline-class-name AnimaPackPipeline --port 30010 --num-gpus 1 "$@"
