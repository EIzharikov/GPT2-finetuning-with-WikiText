"""
Implementation of basic training
"""
from pathlib import Path

import pandas as pd
import torch
import transformers
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torchinfo import summary, ModelStatistics
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config.constants import BASIC_MODEL_PATH, BASIC_MODEL_PRED_PATH
from helpers import (DataImporter, DataPreprocessor, Evaluator, get_device, print_summary,
                     set_seed_in_everything, TrainDataset)


class BasicTrainPipeline:
    """
    Class for basic training
    """

    def __init__(self, model_name: str, current_device: str) -> None:
        """
        Initialize an instance of BasicTrainPipeline.

        Args:
             model_name (str): Name of the model
             current_device (str): Device for model and tokenizer
        """
        self._tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self._model = GPT2LMHeadModel.from_pretrained(model_name)

        self._device = current_device
        self._model.to(self._device)

    def train(self, batch_size: int, num_epochs: float,
              train_data: Dataset, eval_data: Dataset, save_dir: Path | str) -> None:
        # pylint: disable=too-many-arguments
        """
        Train the language model.

        Args:
            batch_size (int): The size of the batch for training
            num_epochs (float): The number of epochs to train the model
            train_data: The training dataset
            eval_data: The evaluation dataset
            save_dir (str): The directory to save the trained model
        """
        train_data = train_data.map(lambda samples: self._tokenizer(samples['text']))
        eval_data = eval_data.map(lambda samples: self._tokenizer(samples['text']))

        self._tokenizer.pad_token = self._tokenizer.eos_token
        trainer = transformers.Trainer(
            model=self._model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                save_total_limit=1,
                num_train_epochs=num_epochs,
                gradient_checkpointing=True,
                gradient_accumulation_steps=1,
                warmup_steps=20,
                max_steps=30,
                save_steps=10,
                learning_rate=2e-1,
                fp16=True,
                logging_steps=1,
                evaluation_strategy="steps",
                save_strategy="steps",
                logging_strategy="steps",
                output_dir=save_dir
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self._tokenizer, mlm=False)
        )
        self._model.config.use_cache = False

        result = trainer.train()
        print_summary(result)

        self._save_model(trainer, save_dir)

    @staticmethod
    def _save_model(trainer: transformers.Trainer, path: str) -> None:
        """
        Save the trained model.

        Args:
            trainer(transformers.Trainer): The model trainer
            path (str): The directory to save the model
        """
        model_to_save = trainer.model.module if hasattr(trainer.model,
                                                        'module') else trainer.model
        model_to_save.save_pretrained(path)

    def get_stats(self) -> ModelStatistics:
        """
        Get statistics about the model.

        Returns:
            dict: A dictionary containing statistics about the model.
        """
        config = self._model.config

        embeddings_length = config.max_position_embeddings

        ids = torch.ones(1, embeddings_length, dtype=torch.long)
        tokens = {
            'input_ids': ids,
            'attention_mask': ids
        }

        return summary(self._model,
                       input_data=tokens,
                       device='cuda',
                       verbose=0)


class BasicInferencePipeline:
    """
    Class for inference with basic model
    """

    def __init__(self, model_path: Path, device: str, dataset: TrainDataset, batch_size: int,
                 max_len: int) -> None:
        # pylint: disable=too-many-arguments
        """
        Initialize an instance of BasicInferencePipeline.

        Args:
            model_path (Path): The path to the pre-trained model
            device (str): The device for inference
            dataset: The dataset used
            batch_size (int): The size of the batch
            max_len (int): The maximum length of generated sequence
        """
        self._tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2',
                                                        padding_side='left')
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        self._model.eval()
        self._model.config.use_cache = True
        self._dataset = dataset
        self._batch_size = batch_size
        self._max_length = max_len
        self._device = device

    def infer_dataset(self) -> None:
        """
        Infer the model on the entire dataset.

        Returns:
            pd.DataFrame: A DataFrame containing predictions.
        """
        predictions = []
        dataset_loader = DataLoader(self._dataset, batch_size=self._batch_size)

        for batch_data in dataset_loader:
            predictions.extend(self._infer_batch(batch_data))

        preds_df: pd.DataFrame = self._dataset.data.loc[:, ['target']]
        preds_df['prediction'] = predictions

        return preds_df

    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """

        res = self._infer_batch(sample)
        return str(res[0])

    def _infer_batch(self, sample_batch: tuple[str, ...]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        ids = self._tokenizer(
            sample_batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        ids.to(self._device)
        generated_ids = self._model.generate(
            **ids, max_new_tokens=self._max_length)

        pred = self._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)
        return list(str(i) for i in pred)


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
    set_seed_in_everything(42)
    print(f'SEED IS {42}')

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

    basic_pipeline = BasicTrainPipeline(model, device)
    basic_pipeline.train(batch_size, epochs, train_dataset, eval_dataset, BASIC_MODEL_PATH)
    print(basic_pipeline.get_stats())

    basic_infer_pipeline = BasicInferencePipeline(BASIC_MODEL_PATH,
                                                  device,
                                                  test_dataset,
                                                  batch_size,
                                                  max_len=max_len)
    predictions_df = basic_infer_pipeline.infer_dataset()
    predictions_df.to_csv(BASIC_MODEL_PRED_PATH, index=False, encoding='utf-8')

    evaluator = Evaluator(BASIC_MODEL_PRED_PATH)
    results = evaluator.run()
    print(results)


if __name__ == '__main__':
    main()
