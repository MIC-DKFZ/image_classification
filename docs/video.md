# Video Models

This document describes the current video-model integration in GloViTa.

## Current Structure

Video support is split into:

- video encoders in [video_encoder](../src/glovita/models/video_encoder)
- video heads in [heads/video](../src/glovita/models/heads/video)

## Encoder Families

Current video encoder config families in [model.py](../src/glovita/configs/model.py):

- `torchvision_video`
- `pytorchvideo`

These are wired through [factory.py](../src/glovita/models/factory.py).

## Clip-Level vs Framewise

There are two different output styles:

- clip-level prediction:
  - encoder returns one final feature tensor
  - standard classification/regression heads can be used
- framewise prediction:
  - encoder returns structured output with intermediate stages
  - a video decoder head consumes those stages

## Intermediate Features

Video encoders now support:

- `return_intermediates`
- `intermediate_names`

If enabled, `forward_features(...)` returns:

```python
{
  "features": final_features,
  "intermediates": {
    "...": ...
  }
}
```

This keeps the interface general:

- standard heads use `features`
- decoder-style heads can consume `intermediates`

For ResNet-style backbones, the default intermediate stage names are:

- `stem`
- `layer1`
- `layer2`
- `layer3`
- `layer4`

## Framewise Decoder

The current framewise decoder head is:

- [framewise_decoder_1d.py](../src/glovita/models/heads/video/framewise_decoder_1d.py)

Config:

- `head_type=framewise_decoder_1d`

It expects intermediate features and is intended for per-frame prediction from a
video backbone.

## Example Shape

Example direction for a framewise setup:

```bash
python train.py \
  --model.encoder.encoder_type torchvision_video \
  --model.encoder.type r3d_18 \
  --model.encoder.return_intermediates \
  --model.head.head_type framewise_decoder_1d
```

## Limitations

- the framewise path is only wired at the model-construction level so far
- dataset, loss, and metric handling are still mostly clip-level classification oriented
- not every video backbone exposes useful intermediate stages automatically
- `pytorchvideo` intermediate capture is generic, but not every architecture will
  have stable stage names unless you set `intermediate_names` explicitly
