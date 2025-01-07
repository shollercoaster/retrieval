import numpy as np
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModel
from data_utils import create_dataset, create_loader
from peft import PeftModel, PeftConfig

from torch import Tensor

@torch.no_grad()
def get_feats(model, tokenizer, data_loader, max_length, device, desc='Get feats'):
    embeds = []

    for text in tqdm(data_loader, total=len(data_loader), desc=desc):
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=max_length,
                               return_tensors="pt").to(device)
        ids = text_input["input_ids"]
        mask = text_input["attention_mask"]
        embed = model(ids, attention_mask=mask)[0]
        in_mask = mask.unsqueeze(-1).expand(embed.size()).float() # embed = model(text_input.input_ids, attention_mask=text_input.attention_mask)
        pooled_embeds = torch.sum(embed * in_mask, 1) / torch.clamp(
                in_mask.sum(1), min=1e-6
        )
        embeds.append(pooled_embeds)
    print("before cat operation: ", type(embeds), type(embeds[0]), embeds[0].shape)
    embeds = torch.cat(embeds, dim=0)
    print("after cat operation: ", type(embeds), type(embeds[0]), embeds[0].shape)
    return embeds


@torch.no_grad()
def contrast_evaluation(text_embeds, code_embeds, img2txt):
    print(text_embeds.shape, text_embeds.t().shape, code_embeds.shape, code_embeds.t().shape)
    text_embeds = torch.nn.functional.normalize(text_embeds, dim=1)  # Shape: [num_queries, embedding_dim]
    code_embeds = torch.nn.functional.normalize(code_embeds, dim=1)
    score_matrix_i2t = text_embeds @ code_embeds.t() # torch.nn.functional.cosine_similarity(text_embeds.t(), code_embeds.t())
    scores_i2t = score_matrix_i2t.cpu().numpy()


    ranks = np.ones(scores_i2t.shape[0]) * -1
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == img2txt[index])[0][0]

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    mrr = 100.0 * np.mean(1 / (ranks + 1))

    eval_result = {'r1': f"{tr1:.2f}",
                   'r5': f"{tr5:.2f}",
                   'r10': f"{tr10:.2f}",
                   'mrr': f"{mrr:.2f}"}
    return eval_result

def get_model_and_dataset(model_name, language, peft_eval=False):
    print("\nCreating retrieval dataset")
    _, _, test_dataset, code_dataset = create_dataset('../../dataset/CSN', language)

    test_loader, code_loader = create_loader([test_dataset, code_dataset], [None, None],
                                                batch_size=[256, 256],
                                                num_workers=[4, 4], is_trains=[False, False], collate_fns=[None, None])

    tokenizer = RobertaTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = RobertaModel.from_pretrained(model_name, trust_remote_code=True)
    
    if peft_eval:
        peft_model = PeftModel.from_pretrained(model, "schaturv/microsoft-text2code-r32", adapter_name="text2code")
        peft_model.eval()  # Set to evaluation mode
        peft_model.set_adapter("text2code")

        print(peft_model)

        print("Active adapters: ", peft_model.active_adapters)
        return peft_model, tokenizer, test_loader, code_loader
    return model, tokenizer, test_loader, code_loader

def evaluation_script(model, tokenizer, test_loader, code_loader):
    print('\nStart zero-shot evaluation...')
    device = torch.device('cuda')
    model.to(device)
    model.eval()

    text_embeds = get_feats(model, tokenizer, test_loader, 512, device, desc='Get text feats')
    code_embeds = get_feats(model, tokenizer, code_loader, 512, device, desc='Get code feats')
    test_result = contrast_evaluation(text_embeds, code_embeds, test_loader.dataset.text2code)

    print(f'\n====> zero-shot test result: ', test_result)
    return test_result

file = open('text2code_combined_results.txt', "a")

for model_name in ["bigcode/starencoder"]: #'microsoft/unixcoder-base', 'microsoft/graphcodebert-base', 'microsoft/codebert-base']:
    file.write(f"{model_name} results: ----------\n")
    file.write("-----------------\n")
    for language in ['ruby', 'go', 'php', 'python', 'java', 'javascript']:
        file.write(f"{language} results: \n")
        model, tokenizer, test_loader, code_loader = get_model_and_dataset(model_name, language, False)
        test_result = evaluation_script(model, tokenizer, test_loader, code_loader)
        file.write("Base Model Results ------------\n")
        file.write(f"zero-shot test result: {test_result}\n")
        model, tokenizer, test_loader, code_loader = get_model_and_dataset(model_name, language, True)
        test_result = evaluation_script(model, tokenizer, test_loader, code_loader)
        file.write("PEFT Model Results ------------\n")
        file.write(f"zero-shot test result: {test_result}")
        file.write('\n--------------\n')