import os

from functools import partial
from pathlib import Path

from datasets import load_dataset
from transformers import (
    Trainer, 
    TrainingArguments, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    GenerationConfig
)

from .training_args import CONFIG
from ..model.TexTeller import TexTeller
from ..utils.functional import tokenize_fn, collate_fn, img_train_transform, img_inf_transform, filter_fn
from ..utils.metrics import bleu_metric
from ...globals import MAX_TOKEN_SIZE, MIN_WIDTH, MIN_HEIGHT   

import os

# Restrict TensorFlow or PyTorch to see only GPU 1
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def train(model, tokenizer, train_dataset, eval_dataset, collate_fn_with_tokenizer):
    training_args = TrainingArguments(**CONFIG)
    #output_dir = "./scratch/santosh/checkpoints-6" 
    trainer = Trainer(
        model,
        training_args,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        tokenizer=tokenizer, 
        data_collator=collate_fn_with_tokenizer,
         
    )

    trainer.train(resume_from_checkpoint=None) ### check what is does 


def evaluate(model, tokenizer, eval_dataset, collate_fn):
    eval_config = CONFIG.copy()
    eval_config['predict_with_generate'] = True
    generate_config = GenerationConfig(
        max_new_tokens=MAX_TOKEN_SIZE,
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    eval_config['generation_config'] = generate_config
    seq2seq_config = Seq2SeqTrainingArguments(**eval_config)

    trainer = Seq2SeqTrainer(
        model,
        seq2seq_config,

        eval_dataset=eval_dataset,
        tokenizer=tokenizer, 
        data_collator=collate_fn,
        compute_metrics=partial(bleu_metric, tokenizer=tokenizer)
    )

    eval_res = trainer.evaluate()
    print(eval_res)
    

if __name__ == '__main__':
    script_dirpath = Path(__file__).resolve().parent
    os.chdir(script_dirpath)

    dataset = load_dataset(str(Path('./dataset/loader.py').resolve()))['train']
    dataset = dataset.filter(lambda x: x['image'].height > MIN_HEIGHT and x['image'].width > MIN_WIDTH)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.flatten_indices()

    tokenizer = TexTeller.get_tokenizer()
    # If you want use your own tokenizer, please modify the path to your tokenizer
    #+tokenizer = TexTeller.get_tokenizer('/path/to/your/tokenizer')
    filter_fn_with_tokenizer = partial(filter_fn, tokenizer=tokenizer)
    dataset = dataset.filter(
        filter_fn_with_tokenizer,
        num_proc=4
    )

    map_fn = partial(tokenize_fn, tokenizer=tokenizer)
    tokenized_dataset = dataset.map(map_fn, batched=True, remove_columns=dataset.column_names, num_proc=8)

    # Split dataset into train and eval, ratio 9:1
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)    
    train_dataset, eval_dataset = split_dataset['train'], split_dataset['test']
    train_dataset = train_dataset.with_transform(img_train_transform)
    eval_dataset  = eval_dataset.with_transform(img_inf_transform)
    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)

    # Train from scratch
    #model = TexTeller()
    model = TexTeller.from_pretrained()#'/home/santoshpalaskar77/IE_643/TexTeller/src/models/ocr_model/train/train_result/test5/checkpoint-130930')#'/home/santoshpalaskar77/IE_643/TexTeller/src/models/ocr_model/train/train_result/test5/checkpoint-130930') #'/home/santoshpalaskar77/IE_643/TexTeller/src/models/ocr_model/train/train_result/test3/checkpoint-23850') #'/home/santoshpalaskar77/IE_643/TexTeller/src/models/ocr_model/train/train_result/test2/checkpoint-30490')
    #model = TexTeller.from_pretrained('/home/santoshpalaskar77/IE_643/TexTeller/src/models/ocr_model/train/train_result/test2/checkpoint-4310')
    
    # for i in range(9):  # Adjusting to freeze layers 0 through 8
    #     for param in model.encoder.encoder.layer[i].parameters():
    #         param.requires_grad = False
     # # Freeze the weights of the encoder
    # Freeze all parameters in the encoder
    # for param in model.encoder.encoder.parameters():
    #     param.requires_grad = False

    # Freeze the first 10 layers (layers 0 to 9) of the decoder
    # for i in range(10):  # Adjusting to freeze layers 0 through 9
    #     for param in model.decoder.model.decoder.layers[i].parameters():
    #         param.requires_grad = False
    # Check which encoder parameters are frozen
    # print("Encoder parameters frozen status:")
    # for name, param in model.encoder.encoder.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")

    # # Check the freeze status of decoder layers
    # print("\nDecoder layers freeze status:")
    # for i, layer in enumerate(model.decoder.model.decoder.layers):
    #     for name, param in layer.named_parameters():
    #         print(f"Layer {i}, {name}: requires_grad={param.requires_grad}")

    

    # # Freeze the weights of the decoder
    # for param in model.decoder.parameters():
    #     param.requires_grad = False  # Set requires_grad to False


    # If you want to train from pre-trained model, please modify the path to your pre-trained checkpoint
    #+e.g.
    #+model = TexTeller.from_pretrained(
    #+    '/path/to/your/model_checkpoint'
    #+)
    
    

    enable_train = True
    enable_evaluate = True #False
    if enable_train:
        train(model, tokenizer, train_dataset, eval_dataset, collate_fn_with_tokenizer)  
    if enable_evaluate and len(eval_dataset) > 0:
        evaluate(model, tokenizer, eval_dataset, collate_fn_with_tokenizer)
