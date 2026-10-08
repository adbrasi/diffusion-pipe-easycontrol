"""Is per-block torch.compile numerically as good as eager? Real Anima block 5 + LoRA r64,
1024px shapes (B=4, T=2, 64x64 tokens). Error of eager-bf16 and compiled-bf16 vs fp32 reference,
for block output, input grad and LoRA A/B grads (what training actually updates)."""
import sys
sys.path.insert(0, '/workspace/diffusion-pipe-easycontrol')
import utils.common
import copy, torch
from models.anima_nextscene import install_nextscene_rope
from safetensors.torch import load_file
from peft import LoraConfig, inject_adapter_in_model
from models.cosmos_predict2_modeling import Block, VideoRopePosition3DEmb
torch.manual_seed(0)
sd = load_file('/workspace/models_anima/split_files/diffusion_models/anima-base-v1.0.safetensors')
p = 'net.blocks.5.'
w = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
blk = Block(x_dim=2048, context_dim=1024, num_heads=16, mlp_ratio=4.0, use_adaln_lora=True,
            adaln_lora_dim=256, self_attention_backend='torch', cross_attention_backend='torch')
print(blk.load_state_dict(w, strict=False))
cfg = LoraConfig(r=64, lora_alpha=64, target_modules=['q_proj','k_proj','v_proj','output_proj','layer1','layer2'])
blk = inject_adapter_in_model(cfg, blk)
for n, prm in blk.named_parameters():
    prm.requires_grad_('lora_' in n)
    if 'lora_B' in n:  # emulate a partially trained adapter
        torch.nn.init.normal_(prm, std=0.02)
B, T, H, W = 4, 2, 64, 64
dev = 'cuda'
x = torch.randn(B, T, H, W, 2048, device=dev)
emb = torch.randn(B, T, 2048, device=dev)
ctx = torch.randn(B, 512, 1024, device=dev)
ada = torch.randn(B, T, 6144, device=dev) * 0.1
pe = VideoRopePosition3DEmb(head_dim=128, len_h=128, len_w=128, len_t=16).to(dev)
install_nextscene_rope(pe, 'aligned', 1)
rope = pe(x, fps=None)
gout = torch.randn(B, T, H, W, 2048, device=dev)

def run(model, dtype, autocast):
    m = model
    xi = x.clone().to(dtype).requires_grad_(True)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
        y = m(xi, emb.to(dtype), ctx.to(dtype), rope_emb_L_1_1_D=rope, adaln_lora_B_T_3D=ada.to(dtype))
    y.float().backward(gout)
    grads = {n: p.grad.float().clone() for n, p in m.named_parameters() if p.grad is not None}
    for p in m.parameters(): p.grad = None
    return y.float().detach(), xi.grad.float(), grads

ref_m = copy.deepcopy(blk).to(dev, torch.float32)
torch.backends.cuda.matmul.allow_tf32 = False
ref = run(ref_m, torch.float32, False)
del ref_m
bf = copy.deepcopy(blk).to(dev, torch.bfloat16)
eager = run(bf, torch.bfloat16, True)
comp_m = copy.deepcopy(blk).to(dev, torch.bfloat16)
comp_m.compile(dynamic=False)
run(comp_m, torch.bfloat16, True)  # warmup/compile
comp = run(comp_m, torch.bfloat16, True)

rel = lambda a, b: ((a - b).norm() / b.norm()).item()
print(f"{'quantity':<28}{'eager_bf16 vs fp32':>20}{'compiled_bf16 vs fp32':>24}{'compiled vs eager':>20}")
print(f"{'block output':<28}{rel(eager[0],ref[0]):>20.3e}{rel(comp[0],ref[0]):>24.3e}{rel(comp[0],eager[0]):>20.3e}")
print(f"{'input grad':<28}{rel(eager[1],ref[1]):>20.3e}{rel(comp[1],ref[1]):>24.3e}{rel(comp[1],eager[1]):>20.3e}")
ge = torch.cat([eager[2][k].flatten() for k in sorted(ref[2])]); gc = torch.cat([comp[2][k].flatten() for k in sorted(ref[2])]); gr = torch.cat([ref[2][k].flatten() for k in sorted(ref[2])])
print(f"{'LoRA grads (all, %d)' % len(ref[2]):<28}{rel(ge,gr):>20.3e}{rel(gc,gr):>24.3e}{rel(gc,ge):>20.3e}")
worst = max(sorted(ref[2]), key=lambda k: rel(comp[2][k], ref[2][k]) / max(rel(eager[2][k], ref[2][k]), 1e-12))
print('worst tensor ratio compiled/eager error:', worst, rel(comp[2][worst], ref[2][worst]), rel(eager[2][worst], ref[2][worst]))
