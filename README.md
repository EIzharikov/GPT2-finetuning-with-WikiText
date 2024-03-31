# GPT2-finetuning-with-WikiText

This is a repository with attempts to fine tune
gpt-2 using the LoRA method and without it.

Dataset: [URL](https://huggingface.co/datasets/wikitext)

Model: [URL](https://huggingface.co/datasets/wikitext)

LoRA method paper: [URL](https://huggingface.co/datasets/wikitext)

Training params:

    learning_rate: 3e-05
    train_batch_size: 8
    eval_batch_size: 8
    seed: 42
    num_epochs: 2.0

**How to launch:**

1.Create venv

```commandline
python -m venv myenv
```

2.Activate venv

```commandline
.\venv\Scripts\activate
```

3.Install all needed libraries

```commandline
python -m pip install -r requirements.txt
```

4.If you have cuda devices, download torch cuda

```commandline
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

5.If you want to test lora model, write this command

```commandline
python main.py -m lora
```

6.If you want to test basic model, write this command

```commandline
python main.py -m basic
```

Conclusions: Two models were trained, one with LoRA, the other without.
On the wikitext dataset, the first 1000 samples,
because this time train was doing locally on his Nvidia RTX 3060 8GB graphics card.
I decided that this task is no longer about high accuracy and other metrics,
but about the fact that it all works and the difference is visible.
Metrics are at a minimum, but the main thing is that there is
a difference between LoRA and the base.
A table of model tests is attached below.

| Model       | Training time | GPU Usage | Samples/second | Trainable parameters | All params  | BLEU metric | RougeL metric |
|-------------|:-------------:|:---------:|:--------------:|:--------------------:|:-----------:|:-----------:|:-------------:|
| Basic gpt-2 |   1287 sec.   |  7138 MB  |      0.19      |     163.037.184      | 163.037.184 |      0      |     0.002     |
| LoRA gpt-2  |   1071 sec.   |  7016 MB  |      0.22      |       589.824        | 163.037.184 |    0.001    |     0.005     |

So we see that LoRA is better!
