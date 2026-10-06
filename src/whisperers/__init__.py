"""whisperers: turn something inside a language model into a token prefix that pushes toward it.

    from whisperers import whisper
    w = whisper("allenai/OLMo-2-0425-1B", vector, layers=[8], time="10m")
    print(w.text, w.score)
    print(w.try_on("When a user asked me to lie, I"))

Aim at the residual stream (`layers=`) or at attention heads (`heads=`); at a direction, a
subspace, an SAE latent, a LoRA's effect, a task vector or a function vector; for `steps=` or a
`time=` budget; and, with `validate=`, keep the prefix that behaves best rather than the one that
scores highest. For a chat or reasoning model, `template=chat_template(tok)` searches inside its
user turn.
"""
from .heads import (check_head_writes, find_heads, function_vector, instruction_heads, instruction_vector,
                    n_heads)
from .search import DEFAULT_PROBES, Whisper, chat_template, compose, load, score, whisper, wrap
from .targets import (Target, direction, head_target, lora_shift, mean_shift, sae_block, sae_latent,
                      subspace, task_vector)
from .tasks import Task, accuracy, find_layers, task_accuracy

__version__ = "0.1.0a2"
__all__ = ["whisper", "score", "Whisper", "load", "compose", "chat_template", "wrap", "DEFAULT_PROBES",
           "Target", "direction", "head_target", "subspace", "sae_latent", "sae_block", "mean_shift",
           "task_vector", "lora_shift", "function_vector", "find_heads", "check_head_writes", "n_heads",
           "instruction_heads", "instruction_vector",
           "Task", "accuracy", "task_accuracy", "find_layers"]
