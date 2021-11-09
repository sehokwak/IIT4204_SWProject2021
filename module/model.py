import importlib


def get_model(args):
    method = getattr(importlib.import_module(args.network), 'Net')
    model = method(args.num_classes)
    return model
