import os
from transformers import Trainer, TrainingArguments

import torch
from datasets import Dataset, DatasetDict
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from peft import LoraConfig

from transformers import RobertaTokenizer, RobertaModel

from utils import load_jsonl, CustomDataset 

languages = ['ruby', 'go', 'php', 'python', 'java', 'javascript']
root_path = "../dataset/CSN"

def get_dataset(root_path, languages, split):
    for lang in languages:
        data_path = os.path.join(root_path, lang, f"{split}.jsonl")
        data_list = load_jsonl(data_path)

    torch_dataset = CustomDataset(data_list)

    return torch_dataset

def get_model(model_name='microsoft/codebert-base'):
    model = RobertaModel.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=['query', 'value'],
        lora_dropout=0.1
    )
    
    # model = get_peft_model(model, lora_config)
    model.add_adapter(lora_config, adapter_name="text2code-r64")
    model.set_adapter("text2code-r64")
    return model, tokenizer

def collate_fn(batch, tokenizer):
    anchor_codes = [item[0] for item in batch]
    positive_codes = [item[1] for item in batch]

    anchor_codes_tensor = tokenizer(
        anchor_codes,
        truncation=True,
        max_length=512,
        padding="longest",
        return_tensors="pt",
    )
    positive_codes_tensor = tokenizer(
        positive_codes,
        truncation=True,
        max_length=512,
        padding="longest",
        return_tensors="pt",
    )

    collated_batch = {
        "anchor": anchor_codes_tensor,
        "positive": positive_codes_tensor,
        "labels": torch.tensor(
            range(len(anchor_codes)), dtype=torch.long
        ),
    }

    return collated_batch

def _get_pooled_embeds(model, batch, field):
    ids = batch[field]["input_ids"]
    mask = batch[field]["attention_mask"]
    embeds = model(ids, attention_mask=mask)[0]
    in_mask = mask.unsqueeze(-1).expand(embeds.size()).float()
    pooled_embeds = torch.sum(embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-6
    )
    return pooled_embeds

class ContrastiveTrainer(Trainer):

    def compute_loss(self, model, batch, return_outputs=False):
        a = _get_pooled_embeds(model, batch, field="anchor")
        p = _get_pooled_embeds(model, batch, field="positive")
        # assert a.shape == (16, 768)
        # assert p.shape == (16, 768)
        scores = torch.stack(
            [F.cosine_similarity(a_i.reshape(1, a_i.shape[0]), p, eps=1e-6) for a_i in a]
        )
        # assert scores.shape == (16,16)
        print("Shapes for pooled embeds: ", a.shape, p.shape, scores.shape)
        loss = F.cross_entropy(scores * 5, batch["labels"])
        return (loss, scores) if return_outputs else loss

def run(model, tokenizer):
    training_args = TrainingArguments(
        "contrastive_trainer",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=400,
        num_train_epochs=1,
        evaluation_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        warmup_steps=4000,
        save_strategy="epoch"
    )
    trainer = ContrastiveTrainer(
        model,
        training_args,
        train_dataset=get_dataset(root_path=root_path, languages=languages, split="train"),
        eval_dataset=get_dataset(root_path=root_path, languages=languages, split="valid"),
        data_collator=lambda x: collate_fn(x, tokenizer),
    )
    trainer.train()

for model_name in ['microsoft/codebert-base', 'microsoft/graphcodebert-base', 'microsoft/unixcoder-base']:
    model, tokenizer = get_model(model_name)
    run(model, tokenizer)
    print(f"\n\n Training completed with {model_name}. \n\n")
    model.push_to_hub("text2code-r64")
