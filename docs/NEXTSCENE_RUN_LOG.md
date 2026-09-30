# Anima NextScene — execução na RTX 5090

## Estado em 2026-09-30 (UTC)

Execução iniciada às ~06:05 UTC. Branch `claude/elegant-ptolemy-8kywqy`.
Handoff e pesquisa lidos integralmente, nessa ordem, antes do setup.
Resultados privados: https://huggingface.co/AdwolfCzar/anima-nextscene-runs

### Marco 0 — ambiente

- RTX 5090, 32.607 MiB, compute capability 12.0; GPU inicialmente livre.
- Disco de 180 GiB sem volume persistente. Uploads são obrigatórios.
- Submodules inicializados. Os avisos de pointers LFS no HiDream não afetam Anima.
- Dependências instaladas em `/venv/main`; torch 2.14.0+cu130.
- Operação real bf16 na GPU passou (matmul). Driver host não foi alterado.
- `python -m pytest -q test/test_anima_nextscene.py test/test_anima_inference_contract.py`:
  **17 passed**, 7,03 s. Ainda não é smoke GPU do pipeline.
- HF autenticado como AdwolfCzar; repo privado criado.
- Download dos três arquivos Base v1.0 e dataset em andamento.
- Manifesto de versões: `setup/environment.txt` no HF.

### Plano imediato

Separar 24 held-out antes dos caches; auditar pares, calibrar filtros olhando
fronteiras; construir captions full/short. Smoke de 10 steps, save/audit,
resume, geração única. Medir batch 1/2/4 antes dos probes de 250–1.000 steps.
Nenhum treino final autorizado pela evidência ainda: primeiro medir leitura de
ref, cópia e qualidade visual em held-out.

### Como acompanhar/retomar agora

```bash
cd /workspace/diffusion-pipe-easycontrol
source /venv/main/bin/activate
tail -n 30 /workspace/nextscene_ops/download.log
python -m pytest -q test/test_anima_nextscene.py test/test_anima_inference_contract.py
```

Comandos exatos dos treinos serão adicionados ao iniciar cada run. Nunca parar
um trainer com SIGTERM; usar `touch <run_dir>/save_quit`.

## Marco 1 — smoke, integração, throughput (~06:25 UTC)

Smoke real: 10 steps @512, ref limpa, `disjoint_w`; loss finita, save e geração
passaram. Resume de 10 para 30 passou, com LR real 5e-5 no log. Adapter: 280
lineares, 112 cross_attn, 560 tensors LoRA; nenhum llm_adapter/adaln.
19 testes CPU passaram após adicionar regressões dos bugs encontrados.

Bugs encontrados antes de aprender qualquer hipótese:

1. Configs fornecidos continham `alpha=64`, rejeitado por `train.py`.
   Removido: o trainer define alpha=rank.
2. `PipelineDataLoader` desempacotava `(target, mask)`; NextScene fornece
   `(target, mask, diff_weight)`. Preservar e sincronizar todos os tensors da label.
3. **PEFT usa suffix matching em listas de targets.** `blocks.0.self_attn.wq`
   atingia também `llm_adapter.blocks.0.self_attn.wq`. O log prometia exclusão,
   mas havia 72 parâmetros treináveis indevidos no bridge. Substituído por regex
   de caminhos completos e guarda de parâmetros treináveis. Regressão reproduz
   o namespace real. Nenhum checkpoint contaminado foi usado em experimentos.
4. A/B não tinha seed inicial explícita no trainer. Adicionado suporte opcional
   a `seed`, usando 42 nos configs desta execução.

| smoke (512 quadrado) | mediana s/step | amostras/s | pico VRAM MiB |
|---|---:|---:|---:|
| micro 1, checkpointing | 0,276 | 3,62 | 9.176 |
| micro 2, checkpointing | 0,516 | 3,88 | 10.096 |
| micro 4, checkpointing | 1,056 | 3,79 | 11.892 |
| micro 4, sem checkpointing | OOM antes do step 1 | — | ~32.000 |

Medições curtas (~30 steps), não equivalem ao throughput de todos os buckets.
Manter activation checkpointing; micro 2 foi ligeiramente mais eficiente.
A geração de smoke é funcional, **não prova o método** (apenas 10 amostras).
Sample: `/workspace/nextscene_artifacts/smoke/sample/20260930-061935_42.png`.
HF: `artifacts/smoke/`; adapters em `checkpoints/smoke_b*/`.

Operação: trainer/eval em fila serial, serviço supervisord `nextscene_worker`;
backup contínuo em serviço `nextscene_sync` (a cada 60s, uploads incrementais de
artefatos e adapters completos). Samples e grids ficam sempre em `/workspace/`.
Fila e estados em `/workspace/nextscene_ops/jobs/`. Resume após reboot só usa
run com arquivo `latest`, nunca reinicia treino com estado completo do zero.

Dados: download HF snapshot entrou em 429 pelo número de HEADs/arquivos.
Migrei para Git LFS: clone sem smudge, reutilização por SHA256 de 12.662 objetos
já presentes e `lfs.concurrenttransfers=32`. Download completo em poucos minutos.
Auditoria DINO agora usa batches de 32 pares (64 imagens), mantendo relatórios
por par e reduzindo overhead de GPU. Bruto em `/workspace/ds`.

Held-out: 24 pares, 6/subset, candidatos com seed 20260930; seleção visual para
incluir câmera/pose novas. Ordem intercalada (limit12 = 3 de cada subset).
Manifesto: `/workspace/heldout/manifest.json` e `artifacts/data/heldout_manifest.json`.
Reservado antes de qualquer cache. Falta ainda remover equivalentes/mesmo vídeo
no build definitivo. A avaliação usa um recorte SFW para facilitar revisão.

Captions: chave OpenRouter existe, mas está **expirada** (HTTP401). Não houve
custo de geração. Meu erro: script de probe continuou 12 chamadas após o primeiro
401; deveria ter abortado ali. Usarei extração determinística de ação/framing,
permitida no handoff, registrando cobertura e limitações; sem bloquear por API.
