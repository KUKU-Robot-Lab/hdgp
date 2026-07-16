from __future__ import annotations

from io import BytesIO
from typing import Any

_SYSTEM_PROMPT = """You ground a robot pouring task. Return exactly one JSON object and no other text.
The keys must be exactly: task, source_id, target_id, nominal_plan, allowed_skills.
task must be pour. Skills may only be approach, pre_grasp_bridge, grasp_lift,
pre_pour_bridge, bimanual_pour, or recovery. Never output joint commands, actions,
contact state, or control poses."""


class QwenBackend:
    """Lazy Hugging Face Qwen3-VL backend; construction does not touch CUDA."""

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct") -> None:
        self.model_id = model_id
        self._processor: Any | None = None
        self._model: Any | None = None
        self._image_type: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._processor is not None and self._model is not None

    def load(self) -> None:
        if self.loaded:
            return
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self._image_type = Image

    def generate(self, command: str, image: bytes) -> str:
        if not self.loaded:
            self.load()
        assert self._processor is not None
        assert self._model is not None
        assert self._image_type is not None
        rgb = self._image_type.open(BytesIO(image)).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb},
                    {"type": "text", "text": command},
                ],
            },
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(next(self._model.parameters()).device)
        generated = self._model.generate(**inputs, do_sample=False, max_new_tokens=256)
        trimmed = [
            output[len(prompt) :]
            for prompt, output in zip(inputs.input_ids, generated, strict=True)
        ]
        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
