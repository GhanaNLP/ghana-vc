"""Run ghana-vc on Modal using the prebuilt image.

    modal run examples/modal_app.py --dataset org/audio-ds --output org/result

The image already has Seed-VC, the torch stack and the checkpoint baked in, so
the container starts converting immediately instead of resolving dependencies.
"""

import modal

IMAGE = "ghcr.io/ghanaopenai/ghana-vc:latest"

app = modal.App("ghana-vc")
image = modal.Image.from_registry(IMAGE, add_python=None)


@app.function(
    image=image,
    gpu="L4",                      # a T4 is enough; L4 is a little faster
    timeout=60 * 60 * 3,
    secrets=[modal.Secret.from_name("huggingface")],   # provides HF_TOKEN
)
def convert(dataset: str, output: str, num_samples: int | None = None,
            diffusion_steps: int = 50):
    from ghana_vc import convert_dataset

    convert_dataset(
        dataset,
        output,
        num_samples=num_samples,
        diffusion_steps=diffusion_steps,
        install_deps=False,        # already installed in the image
    )
    return output


@app.local_entrypoint()
def main(dataset: str, output: str, num_samples: int = 0,
         diffusion_steps: int = 50):
    result = convert.remote(
        dataset, output,
        num_samples=num_samples or None,
        diffusion_steps=diffusion_steps,
    )
    print(f"pushed: https://huggingface.co/datasets/{result}")
