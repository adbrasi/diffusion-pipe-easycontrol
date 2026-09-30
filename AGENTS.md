# AGENTS.md — regras de treino deste repositório (leia antes de configurar qualquer run)

Estas regras vêm do dono do projeto e de meses de saga (Anima, Krea 2, Ideogram 4).
Elas valem para qualquer agente, humano ou modelo. Se alguma regra impedir um
experimento, **pergunte antes**; não contorne.

## GPU e custo
- A GPU é alugada (RTX 5090 32 GB, ~US$0,62/h) e paga do bolso. Tempo de GPU parado,
  lento ou gasto em teste grande demais é dinheiro perdido.
- **Hipóteses se testam em escala pequena:** probes de 250–1.000 steps, **512 px**, com
  grids e métricas a cada save. Resolução alta (768/1024) só no acabamento de um método
  já validado.
- **Smoke de 10 steps antes de qualquer run:** medir s/step, amostras/s e VRAM, e checar
  audit das chaves e resume. Reporte **amostras/s**, não só s/step.

## Batch
- **Batch é `micro_batch_size_per_gpu`.** Use o maior micro batch que couber (4, depois 2).
- **Não use `gradient_accumulation_steps` > 1 para "simular" batch.** Não acelera nada,
  só muda o que conta como step e engana a leitura de progresso. Se não couber batch real,
  treine com o micro batch que cabe e diga isso.
- Compare braços por **amostras vistas**.

## Precisão numérica da base
- **Treine quantizado, com a escala.** Em `*_fp8_scaled`, use `[model] base_quant = 'fp8_scaled'`
  (`models/base.py::ScaledFP8Linear`). Isso guarda o fp8 + a escala do checkpoint e
  desquantiza por forward com o mesmo kernel do ComfyUI: bit-idêntico à inferência, rápido e
  sem block swap. É o equivalente do fp8 do quanto usado pelo ai-toolkit e pelo krea2edit-trainer.
- **Nunca `diffusion_model_dtype = 'float8'` num checkpoint `fp8_scaled`.** Isso
  re-quantiza sem a escala e zerou até ~27% dos pesos dos blocos do Krea 2 (causa da queda
  de qualidade e do patch `K2 Training Base`).
- bf16 com block swap não é o padrão numa 5090: é lento. Só use com motivo medido.

## Contrato treino ↔ inferência
- O contrato real é o que o código executa, não o que a config ou a metadata dizem.
- Todo pipeline de referência precisa de teste de paridade contra a inferência real (ex.:
  `tools/krea2_native_parity.py`, `test/test_anima_nextscene.py`).
- Smoke de 1 imagem antes de qualquer lote de geração.
- Avaliação: pares **held-out**, ref certa × ref trocada, contra o alvo real
  (`tools/nextscene_eval.py`, `tools/k2ab_metrics.py`). O veredito final é visual e do
  usuário. Métrica serve para triagem.

## Operação
- Parar treino: `touch <run_dir>/save_quit`. Nunca SIGTERM.
- Resume do DeepSpeed restaura o LR antigo: confira o LR no log.
- Disco pequeno: pode estados de resume já enviados; o cache de texto do Krea 2 custa
  ~24,5 MB por amostra.
- HF: repositório privado, upload contínuo, model card curto (é vitrine do usuário).
- Documente cada achado no run log do projeto, com commit e push na branch de trabalho.
  Isso inclui erros próprios. Nada de conclusão com n=1 sem avisar.
