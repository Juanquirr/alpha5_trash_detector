# venv: .transformers-5.X-venv
# STATUS: UNAVAILABLE — VideoLLaMA3 custom code imports `VideoInput` from
# transformers.image_utils which does not exist in any released version (as of 5.5.4).
# Requires transformers from git main (pre-release, unstable).
# Keeping implementation for future compatibility check.
from .base import BaseVLM


class VideoLLaMA3(BaseVLM):
    name = "videollama3"
    variant = "DAMO-NLP-SG/VideoLLaMA3-2B"

    def load(self) -> None:
        raise RuntimeError(
            "VideoLLaMA3 requires `VideoInput` from transformers which is not in any "
            "released version. Blocked until transformers adds `VideoInput` to image_utils. "
            "Use smolvlm or qwen_vl instead."
        )
        import torch  # noqa: unreachable — kept for future use
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(self.variant, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.variant,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()

    def describe(self, image_path: str, prompt: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": {"image_path": image_path}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=200)

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated[0], skip_special_tokens=True).strip()
