"""Fatores de LR para torch.optim.lr_scheduler.LambdaLR (sem dependências de GPU)."""


def hold_linear_lambda(decay_start, decay_end, end_factor, warmup_steps=0):
    """Fator 1.0 até o step absoluto `decay_start`, linear até `end_factor` em
    `decay_end`, constante depois. Dentro de um SequentialLR o LambdaLR recebe o
    índice RELATIVO ao fim do warmup; `warmup_steps` converte para absoluto."""
    if not 0 < end_factor <= 1 or decay_end <= decay_start:
        raise ValueError('hold_linear: exige 0 < end_factor <= 1 e decay_end > decay_start')

    def factor(i):
        step = i + warmup_steps
        if step <= decay_start:
            return 1.0
        if step >= decay_end:
            return end_factor
        return 1.0 - (1.0 - end_factor) * (step - decay_start) / (decay_end - decay_start)

    return factor
