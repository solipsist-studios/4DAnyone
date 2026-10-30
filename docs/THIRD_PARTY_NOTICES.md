# Third-party notices

The root Apache-2.0 license applies only to first-party 4DAnyone code. The following components and assets retain their own terms.

## Body pose: SAM 3D Body and MHR

This fork estimates body pose with [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body), which is governed by the SAM License, and uses the [Momentum Human Rig (MHR)](https://github.com/facebookresearch/MHR) keypoint layout, whose source is Apache-2.0. The caller estimates the pose and passes it in as `--sam3d_npz`; no SAM 3D Body weights or code are bundled here. `fdanyone/skeleton/sam3d.py` (gravity alignment and the motion record built from that pose), and the SAM 3D Body pose handling in `fdanyone/pipeline.py`, are taken from [ComfyUI-SplatKit](https://github.com/mickmumpitz/ComfyUI-SplatKit)'s vendored 4DAnyone fork, copyright 2026 mickmumpitz, under the MIT license copied at `third_party/licenses/SPLATKIT_LICENSE`.

The SAM License carries conditions that apply to anyone using this pipeline commercially, including exclusions for ITAR-controlled and military or warfare end uses, a clause allowing Meta to amend the terms, and an indemnity if you sue over the materials or their outputs. Read the license that ships with the weights you use.

The upstream GVHMR and SMPL-X pose path, HMR2, ViTPose, YOLO, and the PyTorch3D compatibility subset are no longer used or shipped in this fork. GVHMR is restricted to educational, research, and non-profit use and SMPL-X to non-commercial use; they were removed for that reason.

## BiRefNet

- Upstream source and model: <https://huggingface.co/ZhengPeng7/BiRefNet>
- Pinned model revision: `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`
- Upstream code: <https://github.com/ZhengPeng7/BiRefNet>
- License: MIT, copied at `third_party/licenses/BIREFNET_LICENSE`

The adaptive preprocessing path downloads the pinned BiRefNet configuration, inference source, and weights on demand. It uses the resulting foreground masks for source cropping and source-framing analysis. Those downloaded files retain their upstream terms.

## Pexels example media

- Source platform: <https://www.pexels.com/>
- Location: `data/source/pexels`
- Terms: <https://www.pexels.com/license/>

The companion Hugging Face repository distributes 20 modified 121-frame excerpts for CLI inference and 4DAnyone Space. They are installed under the location above on demand. Their filenames, source videos, extraction ranges, and checksums are recorded in the pinned [Pexels asset manifest](https://huggingface.co/AntResearch/4DAnyone/blob/4c80e87b805a5f8461cf339cdbe2fb4249e585aa/data/source/pexels/README.md). Pexels permits its media to be used and modified for free and does not require attribution. The source links are retained for provenance and creator credit.

## DiffSynth-Studio inference runtime

- Public source: <https://github.com/modelscope/DiffSynth-Studio>
- Public base revision: `04e39f7de53df7276a7b40ca1791c2a393e05ff3`
- Original experiment fork revision: `c00782d90c872c97bda4745a9e6a41a0a4a7c4db`
- Location: `fdanyone/vendor/diffsynth`
- License: Apache-2.0, copied at `fdanyone/vendor/diffsynth/LICENSE`

`UPSTREAM.md`, `UPSTREAM.patch`, and `VENDORED_FILES.txt` in that directory record the exact provenance, research patch, and retained file set.

One retained DiffSynth file carries additional code-level upstream attribution:

- `models/wan_video_pose_encoder.py` derives its pose network from [Tencent/MimicMotion](https://github.com/Tencent/MimicMotion/tree/c053153a1d124abae8c08568925ae88debc63001), Copyright Tencent, under Apache-2.0.

## Sapiens2-derived Goliath/MHR schema material

- Source project: <https://github.com/facebookresearch/sapiens2>
- Source revision: `0e51c12d7c7257d88431b2d50e523a7b03004854`
- Source file: `sapiens/pose/configs/_base_/keypoints308.py`
- Retained material: the exact MHR70 name/order and Goliath40 link closure needed to reproduce model conditioning
- First-party Apache-2.0 material: RGB palette, color assignment policy, and renderer implementation in `fdanyone/skeleton`
- License copy: `third_party/licenses/SAPIENS2_LICENSE.md`

No Sapiens2 model weights are distributed by this repository.

The retained schema material is conservatively treated as Sapiens2-derived and the complete agreement is included rather than assuming that the root Apache-2.0 license applies. The agreement contains explicit prohibited-use terms, including surveillance, biometric processing, identification or re-identification, deepfakes or deceptive content, and the other activities listed in section 1(b)(vi).

## Model and body assets

The 4DAnyone checkpoint is a first-party release artifact licensed under Apache-2.0, published at the Hugging Face revision frozen in `fdanyone/assets.py`. The MHR70 keypoint names and order used to interpret poses retain the Sapiens2 notice above. The Wan VAE and UMT5-derived frozen prompt conditioning remain third-party assets, mirrored by the companion Hugging Face repository; users remain responsible for every applicable license. The SMPL-X-to-MHR70 regressor that upstream publishes is no longer used here.

The frozen base-model assets come from the official [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) repository at revision `37685f96025fc1425edccdd4b2bca3836ae917ff` and the official [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) repository at revision `3f40b6dc4ca5c02dd23c9db74d9d2ccb82903b86`. The latter is used offline to produce the fixed prompt-conditioning tensor; its T5 weights and tokenizer are not part of reader downloads. Their official model cards license those model assets under Apache-2.0. The redistributed copies are pinned by the frozen Hugging Face revision.

### Wan2.2 TI2V 5B Turbo LoRA

`4danyone/Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors` is a byte-for-byte mirror of the Rank64 difference-LoRA published by [`Kijai/WanVideo_comfy`](https://huggingface.co/Kijai/WanVideo_comfy) at revision `86c2b0442e01eeee630b48fd7efc0cd37af03252`. Kijai identifies its source model as [`quanhaol/Wan2.2-TI2V-5B-Turbo`](https://huggingface.co/quanhaol/Wan2.2-TI2V-5B-Turbo), whose weights are published under [CC BY-NC-SA 4.0](https://github.com/quanhaol/Wan2.2-TI2V-5B-Turbo/blob/main/LICENSE.md). The adapter is redistributed under those upstream terms. **CC BY-NC-SA 4.0 is a non-commercial license, so `--enable_turbo=True` (the default) is not available for commercial use; run with `--enable_turbo=False` to use only the Apache-2.0 base model.** Kijai declares no separate adapter license, and the safetensors metadata is empty. Its exact SHA-256 remains pinned in `fdanyone/assets.py`; the model card retains the full upstream file identity and provenance.

The MHR70 asset is a sanitized inference artifact containing only the sparse support/weight tensors, keypoint names, and a path-free structural schema. The first-party regression tensors are redistributed under Apache-2.0. The embedded name/order metadata remains covered by the schema notice above.

That regressor was trained by the 4DAnyone project on DNA-Rendering frames, using outputs from [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) and the [Momentum Human Rig](https://github.com/facebookresearch/MHR) conversion tools. SAM 3D Body is governed by the SAM License, MHR source is Apache-2.0, and DNA-Rendering access requires a separate signed dataset agreement. The regressor is not bundled in the source repository. Its Hugging Face model card must retain these source-model and dataset citations alongside the split tensor/schema license declaration in this notice.

## VGG-19 perceptual reconstruction

- Loss reference: [Crowdsampling the Plenoptic Function](https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function/tree/f5216f312cf82d77f8d20454b5eeb3930324630a), released under MIT in its upstream README
- VGG-19 model: [Oxford Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/research/very_deep/)
- MatConvNet distribution: [imagenet-vgg-verydeep-19](https://www.vlfeat.org/matconvnet/pretrained/), source MD5 `106118b7cf60435e6d8e04f6a6dc3657`
- Model license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

The optional `splatfacto-perceptual` method uses a custom pixel-plus-feature objective rather than standard LPIPS. The companion Hugging Face repository contains a modified copy of the VGG-19 model: MatConvNet filters are transposed into PyTorch layout and serialized as safetensors, while the unused classifier and final two convolution layers are omitted. The retained tensor values are otherwise unchanged. Credit remains with Karen Simonyan and Andrew Zisserman. Users must preserve that attribution and the CC BY 4.0 license when redistributing the converted weights.
