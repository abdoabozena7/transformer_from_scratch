# Transformer Encoder From Scratch for News Classification

Train a small Transformer to read a news article and classify it as **World**,
**Sports**, **Business**, or **Sci/Tech**. The project is a single, self-contained
teaching notebook. It uses [AG News](https://huggingface.co/datasets/fancyzhx/ag_news),
which it downloads automatically. No dataset files, pretrained weights, or
pretrained tokenizer are needed.

## What we build

```text
article -> word IDs + [CLS] -> token embeddings + sinusoidal positions
        -> Transformer Encoder blocks -> final [CLS] vector -> four class logits
```

In the notebook we write the word tokenizer, the Q/K/V projections and
multi-head attention math, the padding mask, the feed-forward layers, and the
residual connections followed by layer normalization. Q, K, and V come from
three learned transformations of the *same* input. The encoder reads all
retained tokens so one linear layer can score the categories. There is no
decoder, text generation, or fine-tuning.

We use PyTorch for basic layers, tensors, gradients, cross-entropy, and AdamW,
but not a prebuilt Transformer or attention layer. All model weights start
fresh. The notebook runs synthetic shape, masking, gradient, and tiny-batch
checks before training on news.

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/), choose **File →
   Upload notebook**, and upload `transformer_from_scratch.ipynb`. That is the
   only file you need to upload.
2. Optional: select a GPU under **Runtime → Change runtime type**. The
   notebook detects CUDA automatically and uses CPU if CUDA is unavailable.
3. Keep the Section 01 defaults for the first run. Choose **Runtime →
   Run all**. The first code cell installs the required libraries, including
   PyTorch 2.1 or newer if needed. It does not request upgrades, so a
   compatible preinstalled build such as Colab's CUDA build stays in place.
   The dataset cell downloads AG News.
4. In Section 13, edit the example sentence and rerun the prediction call.
   The result contains a predicted category and a probability for each class.
   Section 14 lets you choose which attention layer and head to view.
   Section 16 saves the selected model and shows how to reload it.

For a local Jupyter run, use a Python environment with
`python -m pip install -r requirements.txt` and open the notebook with
`python -m notebook transformer_from_scratch.ipynb`. Run all cells in order.

For checkpoint inference in a fresh session, run Section 01 and the first
code cell in Section 03 to define preprocessing without downloading AG News.
Use the vocabulary saved in the checkpoint; there is no need to build it again.

## Settings to try

| Section 01 setting | Default | What it controls |
|---|---|---|
| `FAST_MODE` | `True` | 6,000 training and 1,000 validation examples, two epochs. `False` uses the full 90/10 partition of official training data and five epochs. |
| `RUN_BONUS_EXPERIMENTS` | `False` | Skip the four extra fits unless you explicitly enable them. |
| `USE_POSITIONAL_ENCODING` | `True` | Give the baseline information about token order. |
| `MAX_LENGTH`, `MIN_WORD_FREQUENCY` | `64`, `2` | Total token slots, including `[CLS]`; minimum training count for regular vocabulary words. |
| `BATCH_SIZE`, `SEED` | `64`, `42` | Examples per batch and repeatable split, initialization, and shuffle. |
| `D_MODEL`, `NUM_HEADS`, `NUM_LAYERS` | `128`, `4`, `2` | Embedding width, attention heads per block, and number of stacked blocks. The width must be divisible by the number of heads. |
| `FF_DIM`, `DROPOUT` | `256`, `0.1` | Hidden width of each feed-forward layer and dropout probability. |
| `LEARNING_RATE`, `WEIGHT_DECAY` | `0.001`, `0.01` | AdamW learning rate and weight decay. |

To run the **bonus**, set `RUN_BONUS_EXPERIMENTS = True` in Section 01 and rerun
the notebook. Keep quick mode for a first comparison. Section 15 trains a
fresh shared reference (four heads, positions on, two layers) and three fresh
variants, changing only one factor at a time: one head, no positions, or one
layer. The runs use the same split, seed, optimizer, batch size, and epoch
budget. `D_MODEL` must also be divisible by four for this comparison.

## Read the results

- **Loss curves:** Lower is better. If training loss falls but validation
  loss starts rising, the model may be overfitting. The baseline restores
  the lowest validation-loss epoch, not necessarily the last one.
- **Test evaluation:** The official test split is reserved for final
  evaluation. Accuracy is the fraction correct; macro F1 weights each class
  equally. The classification report breaks scores down by class. A
  confusion matrix shows true classes in rows and predictions in columns;
  off-diagonal cells reveal which categories get confused.
- **Attention:** The heatmap uses actual attention weights for the selected
  layer and head. A row is the token doing the looking; a column is a token
  it can attend to. Bright cells show stronger attention, not necessarily
  why the model made its final decision.
- **Bonus:** Compare validation loss and accuracy alongside parameter count
  and training time, not held-out test scores. These are exploratory
  single-seed comparisons, not evidence that one design always wins.

The notebook creates `outputs/` automatically. Its files are:

| File | When it appears |
|---|---|
| `loss_curves.png`, `confusion_matrix.png` | After baseline evaluation in Section 12 |
| `attention_heatmap.png` | After drawing attention in Section 14 |
| `architecture_experiments.csv`, `architecture_experiments.png` | Only generated by the enabled bonus in Section 15 |
| `transformer_checkpoint.pt` | After saving the model weights, vocabulary, settings, and class names in Section 16 |

Colab can reset its temporary runtime. Before leaving, use the **Files** pane
to download the checkpoint and any plots or CSV you want to keep. The quick
run is for checking the pipeline, not for estimating the best possible AG News
score.
