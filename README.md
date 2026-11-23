# Full Transformer From Scratch

Build a small **encoder-decoder Transformer** that translates short English
sentences into French. Everything is in one teaching notebook:
[transformer_from_scratch.ipynb](./transformer_from_scratch.ipynb).

The notebook follows the architecture in
[Attention Is All You Need](https://arxiv.org/abs/1706.03762): sinusoidal
positions, encoder self-attention, masked decoder self-attention, and
encoder-decoder cross-attention. It implements attention from scratch in
PyTorch, trains with teacher forcing, and generates translations one token at
a time. It uses no pretrained weights, tokenizer, or prebuilt attention layer.

The original **encoder-only** AG News classifier is unchanged on the
[`decoder-only` branch](https://github.com/abdoabozena7/transformer_from_scratch/tree/decoder-only).

## Run it

**Colab:** Upload the notebook to [Google Colab](https://colab.research.google.com/)
and choose **Runtime → Run all**. No other files are needed. A GPU is optional.

**Local Jupyter:**

```shell
python -m pip install -r requirements.txt
python -m notebook transformer_from_scratch.ipynb
```

In Section 01, leave `FAST_MODE = True` for the first run (6,000 training pairs,
8 epochs). Turn it off for a larger run (24,000 training pairs, 16 epochs).
The notebook downloads the short English-French pairs automatically from
the [PyTorch tutorial archive](https://download.pytorch.org/tutorial/data.zip)
(from [Tatoeba](https://tatoeba.org/) via [ManyThings](https://www.manythings.org/anki/)).

Section 06 checks the masks, cross-attention, and gradients before training.
Section 08 reports held-out loss, shows generated translations, and lets you
try your own sentence. The small model is for learning the architecture, not
for reliable translation.
