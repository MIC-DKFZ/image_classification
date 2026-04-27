# Video Models

This document explains the current video-model integration in GloViTa.

## Current Scope

The current runtime supports:

- clip-level video encoders
- video encoders that can optionally expose intermediate feature stages
- a framewise decoder head at the model-construction level

The current runtime does not yet provide a fully mature end-to-end framewise
video training path across datasets, metrics, and losses. The model assembly
supports it; the broader task pipeline is mostly clip-level oriented.

## Code Layout

Relevant files:

- [../src/glovita/models/video_encoder](../src/glovita/models/video_encoder): active video encoder families
- [../src/glovita/models/heads/video](../src/glovita/models/heads/video): video-specific heads
- [../src/glovita/configs/model.py](../src/glovita/configs/model.py): video encoder and head config classes
- [../src/glovita/models/factory.py](../src/glovita/models/factory.py): encoder/head assembly

## Encoder Families

Current video encoder config families:

- `torchvision_video`
- `pytorchvideo`

These are wrappers around external model families, analogous to the image-side
encoders such as `timm` and `torchvision`.

Why this design is used:

- encoder families differ in model construction details
- the rest of the runtime should not need to know those details
- the same `encoder + head + PEFT` composition should work across image and video

## Clip-Level Vs Framewise

There are two different usage patterns.

### Clip-Level Prediction

The encoder returns one final feature tensor. Standard heads can then be used:

- `classification`
- `regression`

### Framewise Prediction

The encoder returns structured output containing:

- final features
- named intermediate stages

The decoder head consumes those intermediate stages to produce per-frame predictions.

## Structured Encoder Output

Video encoders can be configured with:

- `return_intermediates`
- `intermediate_names`

If enabled, `forward_features(...)` returns:

```python
{
  "features": final_features,
  "intermediates": {
    "stage_name": tensor,
    ...
  }
}
```

This is a deliberate generalization. It avoids hardcoding one single special
encoder/decoder pair.

Why this matters:

- standard heads can use `features`
- decoder-style heads can consume `intermediates`
- different video architectures can expose different stage names

## Default Stage Names

For ResNet-like video backbones, the common default stage names are:

- `stem`
- `layer1`
- `layer2`
- `layer3`
- `layer4`

If a backbone does not use those names, pass explicit `intermediate_names`.

## Framewise Decoder Head

The current decoder head is:

- [framewise_decoder_1d.py](../src/glovita/models/heads/video/framewise_decoder_1d.py)

Its config class is:

- `framewise_decoder_1d`

Important config fields:

- `num_clip_frames`
- `stem_key`
- `layer2_key`
- `layer3_key`
- `layer4_key`

These keys tell the decoder which intermediate feature maps to consume.

## Example Commands

### Clip-Level Video Classification

```bash
glovita_train \
  --data.dataset your_video_dataset \
  --data.data_root_dir /data/videos \
  --model.encoder.encoder_type torchvision_video \
  --model.encoder.type r3d_18 \
  --model.head.head_type classification \
  --peft.method full_finetuning
```

### Framewise Decoder Wiring

```bash
glovita_train \
  --data.dataset your_video_dataset \
  --data.data_root_dir /data/videos \
  --model.encoder.encoder_type torchvision_video \
  --model.encoder.type r3d_18 \
  --model.encoder.return_intermediates \
  --model.head.head_type framewise_decoder_1d
```

If you need non-default intermediate names:

```bash
--model.encoder.intermediate_names stem layer1 layer2 layer3 layer4
```

## PytorchVideo-Specific Notes

The `pytorchvideo` encoder config also exposes:

- `pathway_mode`
  - `auto`
  - `single`
  - `slowfast`
- `slowfast_alpha`

Use `model_kwargs` if a specific backbone constructor needs extra arguments that
are not yet promoted into the typed config.

## Current Limitations

- the framewise path is only cleanly wired at the model-construction level
- not every video backbone exposes useful intermediate stages automatically
- inference code is more mature for standard clip-level tensor inputs than
  for all structured video input/output cases

## Extension Path

To add a new video encoder family:

1. add a config class in [../src/glovita/configs/model.py](../src/glovita/configs/model.py)
2. add the implementation in `src/glovita/models/video_encoder`
3. wire it into [../src/glovita/models/factory.py](../src/glovita/models/factory.py)
4. optionally expose intermediate stages for decoder-style heads

To add a new video head:

1. add the head in `src/glovita/models/heads/video`
2. add the config class in [../src/glovita/configs/model.py](../src/glovita/configs/model.py)
3. wire it into [../src/glovita/models/factory.py](../src/glovita/models/factory.py)
