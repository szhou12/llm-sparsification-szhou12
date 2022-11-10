from torch.nn.utils import prune
import copy
from transformers import RobertaModel, GPT2Model, BartModel
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

SPARSITY_LIST = [.1, .5, .9, .95, .99]

def get_weight_parameters(layer):
    '''
    Get all parameters/modules identified as 'weight'
    '''
    weight_parameters = []
    if len(list(layer.children())) > 0:
        for child in layer.children():
            for param in child.named_parameters():
                if 'weight' == param[0]:
                    # print(param)
                    weight_parameters.append((child, param[0]))
            weight_parameters.extend(get_weight_parameters(child))
    
    
    return weight_parameters


def prune_weight_parameters(model, prune_amount):
    '''
    Global pruning
    '''
    params_to_prune = get_weight_parameters(model)

    prune.global_unstructured(
        params_to_prune, 
        pruning_method=prune.L1Unstructured, 
        amount=prune_amount,
    )

    for module, name in params_to_prune:
        try:
            prune.remove(module, name)
        except Exception as e:
            print(e)
    return model



def save_pruned_models():
    roberta_model = RobertaModel.from_pretrained("roberta-large")
    gpt2_model = GPT2Model.from_pretrained("gpt2")
    bart_model = BartModel.from_pretrained("facebook/bart-large")

    model_list = [("roberta", roberta_model), ("gpt2", gpt2_model), ("bart", bart_model)]

    for name, model in model_list:
        for sparsity in SPARSITY_LIST:
            model_to_prune = copy.deepcopy(model)
            pruned_model = prune_weight_parameters(model_to_prune, sparsity)
            pruned_model.save_pretrained(f"models/{name}_{sparsity}")


def save_pruned_models_glue():
    # Define a padding token and save
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(f"models/gpt2-glue-tokenizer")

    # Add padding to gpt2 in order to do GLUE task
    # num_labels=2 bc GLUE-mrpc is a binary classification dataset
    gpt2_glue = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
    gpt2_glue.config.pad_token_id = tokenizer.pad_token_id
    # Save gpt2 model with padding
    gpt2_glue.save_pretrained("models/gpt2-glue")

    for sparsity in SPARSITY_LIST:
        model_to_prune = copy.deepcopy(gpt2_glue)
        pruned_model = prune_weight_parameters(model_to_prune, sparsity)
        pruned_model.save_pretrained(f"models/gpt2-glue_{sparsity}")

if __name__ == "__main__":
    # save_pruned_models()
    save_pruned_models_glue()