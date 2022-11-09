from torch.nn.utils import prune
import copy

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