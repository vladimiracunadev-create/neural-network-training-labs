from __future__ import annotations

from typing import Any


def build_pretrained_classifier(
    model_name: str,
    classes: int,
    *,
    use_lora: bool = False,
    lora_rank: int = 8,
) -> Any:
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise RuntimeError('Instale el extra text-modern: pip install -e ".[text-modern]"') from exc
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=classes)
    if use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError('Para LoRA instale el extra text-modern que incluye PEFT.') from exc
        configuration = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
        )
        model = get_peft_model(model, configuration)
    return model
