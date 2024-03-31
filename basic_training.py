import pandas as pd
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchinfo import summary
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from config.constants import BASIC_MODEL_PATH, BASIC_MODEL_PRED_PATH
from helpers import print_summary, get_device, set_seed_in_everything, DataImporter, \
    DataPreprocessor, TrainDataset, Evaluator


class BasicTrainPipeline:
    def __init__(self, model_name: str, current_device: str):
        self._tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self._model = GPT2LMHeadModel.from_pretrained(model_name)

        self._device = current_device
        self._model.to(self._device)

    def train(self, batch_size: int, num_epochs: float, train_data, eval_data, save_dir: str):
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
    def _save_model(trainer, path):
        model_to_save = trainer.model.module if hasattr(trainer.model,
                                                        'module') else trainer.model
        model_to_save.save_pretrained(path)

    def get_stats(self):
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
    def __init__(self, model_path, device, dataset, batch_size, max_len):
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

    def infer_dataset(self):
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


def main():
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
