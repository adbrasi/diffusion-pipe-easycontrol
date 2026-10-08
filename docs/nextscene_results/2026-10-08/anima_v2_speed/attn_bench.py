import torch, time
from torch.nn.attention import sdpa_kernel, SDPBackend
B,H,S,D=4,16,8192,128
torch.manual_seed(0)
q,k,v=[torch.randn(B,H,S,D,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
g=torch.randn(B,H,S,D,device='cuda',dtype=torch.bfloat16)
ref=torch.nn.functional.scaled_dot_product_attention(q.float(),k.float(),v.float())
for name,bk in [('default',None),('flash',SDPBackend.FLASH_ATTENTION),('cudnn',SDPBackend.CUDNN_ATTENTION),('efficient',SDPBackend.EFFICIENT_ATTENTION)]:
    try:
        ctx = sdpa_kernel([bk]) if bk else torch.autocast('cuda',enabled=False)
        with ctx:
            for _ in range(3):
                o=torch.nn.functional.scaled_dot_product_attention(q,k,v); o.backward(g)
            torch.cuda.synchronize(); t=time.time()
            for _ in range(10):
                o=torch.nn.functional.scaled_dot_product_attention(q,k,v); o.backward(g)
            torch.cuda.synchronize()
        err=((o.float()-ref).norm()/ref.norm()).item()
        print(f'{name:10s} fwd+bwd {1000*(time.time()-t)/10:7.1f} ms   rel err vs fp32 {err:.2e}')
    except Exception as e: print(name,'unavailable:',str(e)[:80])
