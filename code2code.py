from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModel
import sys
from code_search import get_dataset, collate_fn, ContrastiveTrainer
from transformers import TrainingArguments

def get_model():
    model = RobertaModel.from_pretrained('bigcode/starencoder', trust_remote_code=True)
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/unixcoder-base', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # named_modules/layers inside model
    for name, module in model.named_modules():
        print(name)
    print("\nNamed Parameters\n\n")
    for name, param in model.named_parameters():
        print(name, param.data)

    # print(model.config)
    model.print_trainable_parameters()
    model = model.eval().cuda()
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=['query', 'value'],
        lora_dropout=0.1
    )
    
    # model = get_peft_model(model, lora_config)
    model.add_adapter(lora_config, adapter_name="starencoder-code2code-r64")
    model.set_adapter("starencoder-code2code-r64")

    model.print_trainable_parameters()
    return model, tokenizer

model, tokenizer = get_model()

languages = ["C", "PHP", "Java", "C++", "C#", "Javascript", "Python"]
root_path = "../XLCoST_data"

dataset = get_dataset(root_path=root_path, languages=languages)

training_args = TrainingArguments(
    "contrastive_trainer",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=200,
    num_train_epochs=1,
    evaluation_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    warmup_steps=1000,
    save_strategy="epoch"
)
trainer = ContrastiveTrainer(
    model,
    training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    data_collator=lambda x: collate_fn(x, tokenizer),
)

print("active adapter before training: ", model.active_adapters())

trainer.train()

model.push_to_hub("starencoder-code2code-r64")
