import torch
import torchvision.models as models


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_exact, use_pretrained=True):
    model_ft = None

    if model_name == "resnet":
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_exact)

        num_features = model_ft.fc.in_features
        model_ft.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )
    return model_ft


def parameter_to_update(model, feature_exact):
    print("Params to learn")
    param_array = model.parameters()

    if feature_exact:
        param_array = []
        for name, param, in model.named_parameters():
            if param.requires_grad:
                param_array.append(param)
                print("\t", name)
    else:
        for name, param, in model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    return param_array
