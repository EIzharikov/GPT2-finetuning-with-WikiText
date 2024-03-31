"""
Implementation of LoRA training
"""
from pathlib import Path

import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from basic_training import BasicInferencePipeline, BasicTrainPipeline
from config.constants import LORA_MODEL_PATH, LORA_MODEL_PRED_PATH
from helpers import (DataImporter, DataPreprocessor, Evaluator, get_device, set_seed_in_everything,
                     TrainDataset)


class FullLoraTrainPipeline(BasicTrainPipeline):
    """
    Class for LoRA training
    """

    def __init__(self, model_name: str, current_device: str):
        """
        Initialize an instance of BasicTrainPipeline.

        Args:
             model_name (str): Name of the model
             current_device (str): Device for model and tokenizer
        """
        super().__init__(model_name, current_device)
        self.freeze_weights()
        self.set_lora_adapters()
        self._model.to(current_device)

    def freeze_weights(self) -> None:
        """
        Freeze weights of model.

        """
        for param in self._model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

        self._model.gradient_checkpointing_enable()
        self._model.enable_input_require_grads()

        class CastOutputToFloat(nn.Sequential):
            def forward(self, x): return super().forward(x).to(torch.float32)

        self._model.lm_head = CastOutputToFloat(self._model.lm_head)

    def set_lora_adapters(self):
        """
        Set LoRA adapters

        """
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            fan_in_fan_out=True
        )

        self._model = get_peft_model(self._model, config)


class FullInferenceLoraPipeline(BasicInferencePipeline):
    """
    Class for inference with LoRA model
    """
    def __init__(self, model_path: Path, device: str, dataset: TrainDataset | None, batch_size: int,
                 max_len: int):
        """
        Initialize an instance of FullInferenceLoraPipeline.

        Args:
            model_path (Path): The path to the pre-trained model
            device (str): The device for inference
            dataset (TrainedDataset): The dataset used
            batch_size (int): The size of the batch
            max_len (int): The maximum length of generated sequence
        """
        super().__init__(model_path, device, dataset, batch_size, max_len)
        clear_model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
        lora_config = LoraConfig.from_pretrained(str(model_path))
        lora_config.use_cache = True
        self._model = get_peft_model(clear_model, lora_config).to(device)
        self._model.eval()
        self._tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2',
                                                        padding_side='left')
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._dataset = dataset
        self._batch_size = batch_size
        self._max_length = max_len
        self._device = device


def main() -> None:
    """
    Training
    """
    batch_size = 8
    epochs = 2.0
    max_len = 10
    model = 'openai-community/gpt2'
    device = get_device()
    print(f'DEVICE IS {device}')
    seed = 42
    set_seed_in_everything(seed)
    print(f'SEED IS {seed}')

    train_dataset = load_dataset('wikitext',
                                 'wikitext-2-raw-v1',
                                 split='train')
    eval_dataset = load_dataset('wikitext',
                                'wikitext-2-raw-v1',
                                split='validation')

    raw_dataset = DataImporter('wikitext')
    raw_dataset.obtain()
    processed_dataset = DataPreprocessor(raw_dataset.raw_data)
    processed_dataset.transform()
    test_dataset = TrainDataset(processed_dataset.data)

    lora_pipeline = FullLoraTrainPipeline(model, device)
    lora_pipeline.train(batch_size, epochs, train_dataset, eval_dataset, LORA_MODEL_PATH)
    print(lora_pipeline.get_stats())

    full_infer_pipeline = FullInferenceLoraPipeline(LORA_MODEL_PATH,
                                                    device,
                                                    test_dataset,
                                                    batch_size,
                                                    max_len=max_len)
    predictions_df = full_infer_pipeline.infer_dataset()
    predictions_df.to_csv(LORA_MODEL_PRED_PATH, index=False, encoding='utf-8')

    evaluator = Evaluator(LORA_MODEL_PRED_PATH)
    results = evaluator.run()
    print(results)


if __name__ == '__main__':
    main()
