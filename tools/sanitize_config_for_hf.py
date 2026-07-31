#!/usr/bin/env python3
"""Gera a copia de uma config para publicar no HF, sem o nome do modelo base.

Por que existe: o CLAUDE.md §9 manda subir a configuracao como realmente
usada, mas o usuario nao consegue criar repos privados e nao quer que bots
indexem o repositorio pela palavra-chave do modelo base. As duas coisas sao
conciliaveis: os hiperparametros (o que importa para reproduzir) ficam
intactos e so o identificador do base vira `****`.

O arquivo local NAO e tocado — ele precisa dos caminhos reais para rodar.
A saida e uma copia separada, que e o que o uploader envia.

Uso:
  python tools/sanitize_config_for_hf.py entrada.toml saida.toml [--token krea2]
"""
import argparse
import re
import sys
from pathlib import Path

AVISO = (
    '# NOTA: o identificador do modelo base foi substituido por **** nesta\n'
    '# copia publicada, a pedido do autor. Todos os hiperparametros estao\n'
    '# exatamente como usados no treino.\n'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('entrada')
    ap.add_argument('saida')
    ap.add_argument('--token', default='krea2',
                    help='string a redigir (case-insensitive)')
    args = ap.parse_args()

    texto = Path(args.entrada).read_text()
    limpo, n = re.subn(re.escape(args.token), '****', texto, flags=re.IGNORECASE)
    if n == 0:
        print(f'aviso: "{args.token}" nao aparece em {args.entrada}', file=sys.stderr)

    # guarda: se sobrou o token em qualquer forma, nao publicar
    if re.search(re.escape(args.token), limpo, flags=re.IGNORECASE):
        print('ERRO: token ainda presente apos a redacao', file=sys.stderr)
        sys.exit(1)

    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(AVISO + limpo)
    print(f'{args.entrada} -> {args.saida} ({n} ocorrencias redigidas)')


if __name__ == '__main__':
    main()
