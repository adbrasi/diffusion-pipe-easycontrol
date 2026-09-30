# Operação da pesquisa na instância GPU

Snapshot dos scripts usados em `/workspace/nextscene_ops/`, versionado para
reproduzir o experimento. São scripts desta instância: os caminhos `/workspace/`,
o repositório HF privado e os nomes dos jobs estão configurados explicitamente.
Não contêm tokens; autenticação usa o ambiente/cache do Hugging Face.

- `worker.py`: fila serial supervisionada em `/workspace/nextscene_ops/jobs/`,
  logs/monitoramento GPU e retomada do estado DeepSpeed mais recente.
- `sync.py`: backup contínuo de adapters e artefatos para HF privado.
- `build_data.py`: construção do dataset filtrado e recorte de probes, a partir
  dos audits e manifest de heldout salvos nos artefatos.
- `eval_latest.py`: avaliação do checkpoint solicitado (12 pares,512/20steps).
- `make_comparisons.py`, `expand_comparisons.py`: grids dos testes já avaliados.

Os serviços em execução continuam usando os arquivos de `/workspace/nextscene_ops/`.
Configurações de treino estão em `examples/anima_nextscene/gpu_20260930/`.
Para a revisão consolidada de3colunas, use `tools/nextscene_review_grid.py`.

Não reinicie o worker enquanto houver trainer ativo. Para interromper treino
sem perder estado, crie `save_quit` no diretório do run; não envie SIGTERM.
O worker registra jobs com falha e segue a fila; antes de enfileirar uma
continuação, confirme a existência do checkpoint completo esperado.
