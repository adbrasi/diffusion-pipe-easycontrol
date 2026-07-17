# Omini-Grounded — dando mais força ao canal semântico (guia de calibração)

**Contexto:** no adapter validado, o canal semântico (Qwen3-VL grounded → txtfusion)
carrega só **~1,13% da energia** do adapter — quase toda a capacidade aprendida vai
para o canal de aparência (LoRA routado nos blocks). Isso é em parte design (o
built-in do Krea 2 já entrega grounding de graça; se o canal semântico fosse forte
demais ele poderia satisfazer a loss sozinho e matar de fome a aparência), mas
limita o quanto o modelo COMPREENDE relações novas entre referência e cena.
Este guia lista os knobs para inverter esse equilíbrio em treinos futuros,
do mais barato ao mais invasivo.

## Anatomia atual (onde cada coisa vive)

```text
caption + ref ──► Qwen3-VL 4B (CONGELADO) ──► stack 12 camadas (30720d)
                                                   │
                              TextFusionTransformer ◄── LoRA GLOBAL rank 64  (~1% energia)
                              (projector EXCLUÍDO)
                                                   │
                                  txtmlp ──► tokens de texto no DiT
28 SingleStreamBlocks ◄── LoRA ROUTADO às rows da ref (~99% energia)
```

- Targets do adapter: `models/krea2.py:21` — `adapter_target_modules = ['SingleStreamBlock', 'TextFusionTransformer']` (+ txtmlp), filtrados por `adapter_allowed_key_substrings = ('.blocks.', '.txtfusion.')` em `models/krea2_omini_grounded.py`.
- LoraConfig único (mesmo rank para tudo): `models/krea2.py:47`.
- Router só nos blocks (txtfusion fica global): `configure_adapter` de `models/krea2_omini_grounded.py` → `router.install(self.diffusion_model.blocks)`.
- Um único param group no otimizador (mesmo lr para os dois canais): `train.py` `get_optimizer`.

## Knobs, do mais barato ao mais invasivo

### 0. Sem retreino: `fusion_strength` > 1 na inferência
O node `CtxRushKrea2OminiGroundedApply` tem dials separados. Antes de retreinar,
teste `fusion_strength` 1.5–3.0 com `block_strength` fixo: como o delta do
txtfusion é pequeno, superamplificá-lo é seguro e mede se o problema é
capacidade aprendida ou só escala. Se 2–3× já melhorar a compreensão semântica,
talvez nem precise retreinar.

### 1. `rank_pattern` do PEFT: rank maior SÓ no txtfusion (fácil, recomendado)
`peft.LoraConfig` aceita `rank_pattern`/`alpha_pattern` (dict regex→valor).
Mudança em `models/krea2.py:47` (ou num override no pipeline grounded):

```python
peft_config = peft.LoraConfig(
    r=adapter_config['rank'],                      # 64 nos blocks
    lora_alpha=adapter_config['alpha'],
    rank_pattern={'txtfusion': adapter_config.get('txtfusion_rank', 128)},
    alpha_pattern={'txtfusion': adapter_config.get('txtfusion_rank', 128)},  # manter alpha=rank
    ...
)
```

Config: `[adapter] txtfusion_rank = 128` (ou 256). Custo de VRAM marginal
(o txtfusion é pequeno). O save/load e o node ComfyUI já leem shapes do
próprio tensor, então ranks mistos funcionam sem mudança na inferência.

### 2. LR maior para o txtfusion (fácil-médio)
Hoje todos os params treináveis dividem um lr. Para dar 2–5× ao canal semântico,
separar param groups no `get_optimizer` (`train.py`) — precedente já existe no
fork: `llm_adapter_lr` do caminho Anima. Esboço:

```python
fusion_params = [p for n, p in pipeline_model.named_parameters()
                 if p.requires_grad and 'txtfusion' in getattr(p, 'original_name', n)]
block_params  = [p for ... if 'txtfusion' not in ...]
optimizer = optim_cls([
    {'params': block_params},
    {'params': fusion_params, 'lr': config['optimizer'].get('txtfusion_lr', lr)},
], lr=lr, ...)
```

Config: `[optimizer] txtfusion_lr = 3e-4` (com lr base 1e-4). Atenção: warmup
do DeepSpeed scheduler aplica fator ao grupo — verificar que o SequentialLR
respeita lrs por grupo (respeita: LinearLR multiplica o lr inicial de cada grupo).

### 3. Incluir o projector no adapter (uma linha, risco médio)
`adapter_allowed_key_substrings = ('.blocks.', '.txtfusion.')` →
adicionar `'.projector.'` (e garantir que o módulo entra nos targets).
O projector comprime o stack 30720→6144 e é o gargalo de informação do canal;
adaptá-lo dá ao task acesso a informação do Qwen que o colapso padrão descarta.
Risco: é uma peça calibrada do built-in — usar rank baixo (16–32) e monitorar
se o s0 (built-in puro) não degrada.

### 4. Dropout assimétrico Composer-style (médio; ataca a competição na raiz)
Para forçar o canal semântico a trabalhar: dropar o canal de APARÊNCIA
(zerar os tokens VAE da ref) com prob p_app (ex. 0.3) mantendo o grounding —
na ausência da aparência, a loss só pode ser reduzida via semântica → gradiente
flui para o txtfusion. O inverso do que o Composer faz (drop maior no canal
dominante). Implementação: no `prepare_inputs`/packing do grounded, com prob
p_app substituir `control_latents` por zeros MAS manter a imagem no cache de
texto (que já é per-sample). Cuidado: o cache de texto é pré-computado — o drop
de aparência é em runtime (latents), então não exige recache.

### 5. Full-finetune do txtfusion (invasivo)
Em vez de LoRA, marcar os 4 blocos do txtfusion inteiros como treináveis
(requires_grad=True fora do PEFT) com lr próprio. Máxima capacidade semântica;
exige mudar o save_adapter (salvar os pesos full do txtfusion junto com o LoRA)
e o node (aplicar pesos full em runtime). Só se 1+2+4 não bastarem.

## Protocolo de calibração sugerido (ordem dos testes)

1. **Baseline sem retreino**: sweep `fusion_strength` {1, 1.5, 2, 3} no adapter atual.
2. **Retreino A**: `txtfusion_rank = 128` + `txtfusion_lr = 3e-4` (knobs 1+2), resto igual.
3. **Retreino B**: A + dropout assimétrico p_app = 0.3 (knob 4).
4. Comparar sempre com o E1 do plano de experimentos: `fusion_strength=0`
   (built-in puro) vs 1 — a DIFERENÇA entre eles é a contribuição real do canal
   semântico aprendido. O objetivo destes retreinos é aumentar essa diferença
   sem degradar a fidelidade de aparência (block_strength fixo no sweet spot).
5. Forense de pesos após cada retreino: % de energia do txtfusion e rank efetivo
   (o script da análise está na sessão de auditoria; alvo: sair de ~1% para
   5–15% sem colapso da aparência).

## Armadilhas conhecidas

- **Não** aumentar lr global para fortalecer o txtfusion — os blocks routados
  também aceleram e o equilíbrio não muda (só o overfit).
- `caption_dropout` (CFG) interage: o uncond grounded treina o caminho semântico
  com caption vazio. Com canal semântico mais forte, manter 0.1 e monitorar se o
  CFG não superamplifica semântica (baixar cfg na inferência se necessário).
- Metadata não substitui bisseção: qualquer mudança aqui deve ser validada com o
  protocolo de forward único / noise-paired da saga (o contrato REAL é o que o
  código executa).
