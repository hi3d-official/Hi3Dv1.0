# Hi3D v1.0

**English** · <a href="README.zh-CN.md"><kbd>简体中文</kbd></a>

**High-resolution 3D geometry refinement built upon Sparc3D.**

**[Live Demo](https://huggingface.co/spaces/ilcve21/Sparc3D)** · [Hugging Face](https://huggingface.co/hi3d-official/Hi3Dv1.0) · [Run on swm cloud](https://cloud.swmgpu.com/w/samofvaani/hi3d-mesh-refiner-studio) · [International Website](http://hi3d.ai/) · [China Website](https://www.hi3d.cn/)

**[Technical Report (PDF)](https://github.com/hi3d-official/Hi3Dv1.0/blob/main/docs/Hi3D_v1.0_Technical_Report.pdf)** · [Download PDF](https://github.com/hi3d-official/Hi3Dv1.0/raw/refs/heads/main/docs/Hi3D_v1.0_Technical_Report.pdf)

Released on October 23, 2026.

## Overview

Hi3D v1.0 refines a coarse mesh into high-resolution geometry, conditioned on a reference image of the same object. Built upon the Sparc3D representation and refinement architecture, it progressively samples sparse latents, decodes them with a self-pruning sparse VAE, and extracts a mesh.

The project report describes training on approximately **two million curated 3D assets** and refinement up to an effective sparse extraction-grid resolution of **1536³**. The Python package and public API retain the names `sparc3d` and `Sparc3DPipeline` for compatibility.

This repository provides **inference code**. Training code and the paper's evaluation harness are not included. Model artifacts are available on [Hugging Face](https://huggingface.co/hi3d-official/Hi3Dv1.0). The examples load `hi3d-official/Hi3Dv1.0` directly; a compatible local model directory is also supported.

## System overview

[![Hi3D v1.0 system overview: reference image and coarse mesh, sparse refinement, and three mesh extraction levels](assets/framework.png)](assets/framework.png)

Image features condition sparse flow on coordinates derived from the coarse geometry. After decoding and mesh extraction, the mesh is revoxelized to provide the sparse support for the next pass.

| Pass | Nominal structural grid | Sparse extraction grid | Transformer |
| --- | --- | --- | --- |
| 1 | 64³ | 512³ | `transformer_512` |
| 2 | 128³ | 1024³ | `transformer_1024` |
| 3 | 192³ | 1536³ | `transformer_1024` |

A 512³ request runs one pass; 1024³ runs two; 1536³ runs all three. These labels describe nominal sparse coordinate and extraction grids, rather than a dense 1536³ latent tensor or a fixed triangle count. See [pipeline details](docs/pipeline.md).

## Visual comparisons

### Local geometry

[![Local geometry comparison across nine methods, with Hi3D v1.0 highlighted in teal](assets/local-detail-comparison.jpg)](assets/local-detail-comparison.jpg)

Red boxes mark the reference regions enlarged in each row. Shaded geometry and colored surface-normal renderings show local structures, including layered relief, architectural moldings, curved boundaries, scales, teeth, and concentric grooves. **Hi3D v1.0 is highlighted in teal in the rightmost column.** Click the figure to inspect the full-resolution image.

### Multi-view comparisons

[![Multi-view geometry comparisons for a vehicle, armored character, armchair, and handbag](assets/global-comparison-1.jpg)](assets/global-comparison-1.jpg)

Four objects are shown from the front, back, and bottom. Results combine shaded geometry and colored surface-normal renderings; the color does not represent generated textures. Hidden views illustrate plausible completion, rather than exact reconstruction of unseen surfaces.

<details>
<summary>More multi-view examples</summary>

[![Additional multi-view examples for a humanoid bust, mechanical robots, and a stylized character](assets/global-comparison-2.jpg)](assets/global-comparison-2.jpg)

Additional examples follow the same column order and rendering conventions as the preceding figure. Hi3D v1.0 remains the teal-highlighted rightmost column.

</details>

## Installation

Requirements: **Linux, Python 3.10, an NVIDIA GPU with compute capability 8.0 or newer, and a CUDA toolkit in the 12.0–12.5 range** matching the installed PyTorch build. There is no CPU inference path. The measured stack used Python 3.10.12, PyTorch 2.5.0+cu124, CUDA toolkit 12.4.131, and a **96 GB NVIDIA H20**. The 1536³ ladder has only been validated on that GPU; peak memory and feasibility on smaller cards have not been established.

Clone this repository, then run from its root in a Python 3.10 environment:

```bash
git clone https://github.com/hi3d-official/Hi3Dv1.0.git
cd Hi3Dv1.0

pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 \
  --extra-index-url https://download.pytorch.org/whl/cu124
pip install packaging ninja psutil
pip install --no-build-isolation flash-attn==2.7.4.post1
pip install -c docker/constraints.txt -e ".[inference]"
```

The constraints preserve the measured kernel stack, including `spconv-cu120==2.3.6` and `flash-attn==2.7.4.post1`. Install `flash-attn` separately after PyTorch: its source build needs the installed torch and CUDA toolkit, with build isolation disabled. Triton mesh kernels compile at first use. This project is installed from source; no PyPI release or prebuilt project wheel is supplied.

For background removal in the demos, additionally install `pip install -c docker/constraints.txt -e ".[demo]"`. This extra installs `rembg`, not a coarse generator. See [installation](docs/installation.md) for Docker, the complete verified stack, offline setup, and alternative CUDA/kernel builds. Other configurations have not been measured.

## Model artifacts

The [Hugging Face model repository](https://huggingface.co/hi3d-official/Hi3Dv1.0/tree/main) provides these eight runtime files at its root. A local copy must preserve this layout:

```text
model_index.json
scheduler/scheduler_config.json
vae/config.json
vae/model.safetensors
transformer_512/config.json
transformer_512/model.safetensors
transformer_1024/config.json
transformer_1024/model.safetensors
```

The loader consumes JSON and safetensors; it does not execute model-repository Python or unpickle checkpoints. It also accepts a Hugging Face model repository ID with the same artifact layout. Standard Hub loading defers the high-resolution transformer until needed; local directories must contain all files, and enabling sequential CPU offload materializes both transformers on the CPU.

The conditioning encoders, `facebook/dinov2-large` and `openai/clip-vit-large-patch14`, are loaded separately from Hugging Face on first use. They are not part of the model directory above. `--revision` selects the refinement artifact revision; `--cache-dir` points to the directory holding `models--*` trees, normally `$HF_HOME/hub`.

## Python API

```python
from PIL import Image
import trimesh
from sparc3d import Sparc3DPipeline

pipeline = Sparc3DPipeline.from_pretrained(
    "hi3d-official/Hi3Dv1.0",
).to("cuda")

result = pipeline(
    Image.open("input.png"),
    trimesh.load("coarse.glb", force="mesh", process=False),
    resolution=1024,
    num_inference_steps=25,
    seed=0,
)
result.mesh.export("refined.glb")
```

`result` is a `Sparc3DPipelineOutput` containing a geometry-only `trimesh.Trimesh` and the achieved `resolution`. Supported resolutions are **512, 1024, and 1536**, with 25 scheduler updates per pass by default. Supply `seed` or a local CPU `torch.Generator`, not both. For existing unique integer voxel coordinates shaped `[N, 3]`, the API also provides `pipeline.refine_voxels(image, voxel_coords, structure_resolution=64, resolution=1536, seed=0)`.

### Memory offload and progress

To reduce model residency on the GPU, enable Accelerate offload while the pipeline is still on the CPU:

```python
pipeline = Sparc3DPipeline.from_pretrained("hi3d-official/Hi3Dv1.0")
pipeline.enable_sequential_cpu_offload("cuda")
```

Use this instead of `.to("cuda")`; the two modes cannot be combined. Offload does not establish a minimum supported GPU memory size and does not eliminate decoded-feature memory costs.

Pass `progress_callback` to `from_pretrained` to receive a `Sparc3DProgressEvent` after every scheduler update:

```python
def report(event):
    print(f"pass {event.pass_index + 1}/{event.num_passes} "
          f"step {event.step}/{event.num_inference_steps}")

pipeline = Sparc3DPipeline.from_pretrained(
    "hi3d-official/Hi3Dv1.0", progress_callback=report,
).to("cuda")
```

Decoding, extraction, and postprocessing do not emit scheduler progress events.

## Command line

Refine an existing coarse mesh without any Hunyuan dependency:

```bash
python demo/refine_mesh.py \
  --model hi3d-official/Hi3Dv1.0 \
  --image input.png --coarse-mesh coarse.glb --output refined.glb \
  --resolution 1024 --num-inference-steps 25 --seed 0
```

Add `--sequential-cpu-offload` for model offload, or `--remove-background` after installing the demo extra. Background removal is opt-in; transparent cut-out inputs retain their alpha for subject framing.

For offline refinement, provide local copies of all three model components:

```bash
python demo/refine_mesh.py \
  --model /path/to/hi3d-v1.0-model \
  --image input.png --coarse-mesh coarse.glb --output refined.glb \
  --local-files-only \
  --dino-model /path/to/dinov2-large \
  --clip-model /path/to/clip-vit-large-patch14
```

## Optional Hunyuan coarse-generation example

`demo/coarse_stage.py` is an optional adapter for **Hunyuan3D-2 mini**, using `tencent/Hunyuan3D-2mini` and its `hunyuan3d-dit-v2-mini-turbo` subfolder. `demo/image_to_mesh.py` calls this adapter before refinement. Hunyuan is used only to demonstrate an upstream coarse generator; it is neither a required dependency of the core library nor a required product integration. You can replace the adapter or supply your own coarse mesh to `demo/refine_mesh.py`, which has no Hunyuan import.

Install the [Hunyuan3D-2 code](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) and obtain its weights separately before running:

```bash
python demo/image_to_mesh.py \
  --model hi3d-official/Hi3Dv1.0 \
  --image assets/example.png --output refined.glb \
  --resolution 1024 --seed 0
```

The Hunyuan code and weights are third-party releases, not code or weights released or relicensed by Hi3D. Follow the [upstream source license](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE) and the [Hunyuan3D-2 mini model license](https://huggingface.co/tencent/Hunyuan3D-2mini/blob/main/LICENSE). See [external models and optional examples](THIRD_PARTY_LICENSES.md#external-models-and-optional-examples) and the [coarse-stage installation notes](docs/installation.md#the-coarse-stage).

`--coarse-model` and `--coarse-subfolder` override the coarse model. Offline end-to-end use additionally needs a local Hunyuan model: the refinement flags `--local-files-only`, `--revision`, and `--cache-dir` do not configure the Hunyuan loader.

## Scope and reproducibility

- **Geometry only, batch size one.** No textures, materials, or vertex colors are generated. Output is recentered and normalized to unit maximum radius, so input world coordinates and absolute scale are not preserved.
- **Coarse structure matters.** Severe structural errors in the coarse mesh can persist. Geometry hidden from the image is a plausible completion.
- **GPU kernels are required.** `flash-attn` has no fallback attention path. Adaptive voxelization may use a coarser support grid on complex inputs and emits a warning. A request that cannot reach the target resolution raises instead of silently returning a lower resolution.
- **Seeds are not mesh hashes.** The refinement sampler uses a local CPU RNG, but GPU sparse convolution and attention are not bitwise deterministic. Compare meshes geometrically using the same inputs, artifact, scheduler, and environment.
- **Inference validation only.** Training code, the evaluation harness, benchmark splits, and comparison meshes are not included; this repository alone cannot reproduce the paper's comparative tables. Geometric detail and nominal resolution do not guarantee fitness for a downstream application.

## Documentation

- [Installation](docs/installation.md): verified stack, Docker, offline loading, and GPU dependencies.
- [Pipeline](docs/pipeline.md): refinement passes, conditioning, extraction, and memory behavior.
- [Reproduction](docs/reproduction.md): validation scope and meaningful run-to-run comparisons.

## License and acknowledgments

The Hi3D-owned or authorized portions of the second-stage implementation and modifications, documentation, configuration, and eight model artifact files identified in Schedule 2 are governed by the **[Hi3D Open License Agreement Version 1.0](LICENSE)**. See [license scope](THIRD_PARTY_LICENSES.md#license-scope) for the scope and exclusions; the Agreement controls.

- **Commercial use is permitted under the Agreement.** If a related product or service exceeds **200,000 monthly active users** or reaches **USD 1,000,000 in monthly revenue**, notify the Licensor and apply for a separate written license within **60 days**, subject to Section 5 and its aggregation and transition rules.
- **Commercial integrations must display `Powered by Hi3D 1.0`** as specified in Section 7, linking to the official website identified in Schedule 2 where supported. Distribution and use also remain subject to the Agreement's notice requirements and Acceptable Use Policy.
- **Standalone outputs have different treatment.** Obtaining, editing, displaying, selling, or distributing an output that contains no Hi3D Materials or Model Derivative does not by itself trigger the branding, Agreement/NOTICE distribution, or scale-license obligations. Applicable law and third-party rights still apply.

Existing **Apache-2.0 and MIT terms remain applicable to the original and third-party portions they cover**; those rights are not replaced by the Hi3D Agreement. The original Apache text is preserved in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md#original-apache-license). See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md), [NOTICE](NOTICE), and [AUTHORS](AUTHORS) for retained terms and attribution. The included Hunyuan adapter is distinct from the external Hunyuan implementation, weights, and outputs, which remain subject to their own terms, as do DINOv2 and CLIP.

For Hi3D license inquiries, contact [contact@hi3d.ai](mailto:contact@hi3d.ai).

Hi3D v1.0 builds upon [Sparc3D: Sparse Representation and Construction for High-Resolution 3D Shapes Modeling](https://arxiv.org/abs/2505.14521). We acknowledge the original Sparc3D work and its upstream research.
