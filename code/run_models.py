from EconModels import *
from RBFDQN import *

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    env = GrowthModel(δ=1)
    params = {'env_name': 'solow',
            'env': env,
            'max_episode': 300,
            'num_layers': 2,
            'layer_size': 256,
            'num_layers_action_side': 1,
            'layer_size_action_side': 256,
            'learning_rate': 0.00025,
            'learning_rate_location_side': 2.5e-05,
            'target_network_learning_rate': 0.005,
            'max_buffer_size': 50000,
            'gamma': env.β,
            'batch_size': 256,
            'num_points': 100,
            'temperature': 1,
            'policy_parameter': 1.5, # Determines how fast exploration decays
            'norm_smoothing': 1e-05,
            'updates_per_episode': 500,
            'burn_steps': 15,
            'dropout_rate': 0,
            'optimizer': 'Adam',
            'policy_type': 'e_greedy',
            'seed_number': 2056}

    for temp in np.linspace(0.25, 2.5, 10):
        params['temperature'] = temp
        train_model(params, device=device, save=False)

    params['temperature'] = 1

    for n in np.arange(start=50, stop=401, step=50):
        params['num_points'] = n
        train_model(params, device=device, save=False)

    env = RBC(δ=1)
    params = {'env_name': 'RBC',
            'env': env,
            'max_episode': 300,
            'num_layers': 2,
            'layer_size': 256,
            'num_layers_action_side': 1,
            'layer_size_action_side': 256,
            'learning_rate': 0.00025,
            'learning_rate_location_side': 2.5e-05,
            'target_network_learning_rate': 0.005,
            'max_buffer_size': 50000,
            'gamma': env.β,
            'batch_size': 256,
            'num_points': 100,
            'temperature': 1,
            'policy_parameter': 1.5, # Determines how fast exploration decays
            'norm_smoothing': 1e-05,
            'updates_per_episode': 500,
            'burn_steps': 15,
            'dropout_rate': 0,
            'optimizer': 'Adam',
            'policy_type': 'e_greedy',
            'seed_number': 2056}
    train_model(params, device=device, save=False)
