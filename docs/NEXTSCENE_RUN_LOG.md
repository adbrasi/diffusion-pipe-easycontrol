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
