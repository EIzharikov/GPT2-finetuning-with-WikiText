import argparse

from basic_training import BasicInferencePipeline
from config.constants import LORA_MODEL_PATH, BASIC_MODEL_PATH
from helpers import get_device
from lora_training import FullInferenceLoraPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="The chosen model to use. "
                                                        "Write 'lora' if you want to use LoRA "
                                                        "model, 'basic' you want use basic one.")

    args = parser.parse_args()

    batch_size = 8
    max_len = 10
    device = get_device()
    test_dataset = None
    if args.model == 'lora':
        pipe = FullInferenceLoraPipeline(LORA_MODEL_PATH,
                                         device,
                                         test_dataset,
                                         batch_size,
                                         max_len=max_len)
    elif args.model == 'basic':
        pipe = BasicInferencePipeline(BASIC_MODEL_PATH,
                                      device,
                                      test_dataset,
                                      batch_size,
                                      max_len=max_len)
    else:
        raise AttributeError('Choose LoRA or basic model!')

    while True:
        prompt = input("Prompt ")
        if not prompt:
            break

        output_text = pipe.infer_sample((prompt,))
        print(output_text, end='\n\n')


if __name__ == "__main__":
    main()
