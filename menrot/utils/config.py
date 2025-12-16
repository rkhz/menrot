import json

def load_model_from_config(model_class, config_path, device=None):
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_config = config['model']['config']

    print(f'Instantiating Model: {model_class.__name__}')
    model = model_class(**model_config).to(device)

    return model, model_config
