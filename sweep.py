import wandb
from score import train, train_raw
sweep_configuration = {
    'method': 'bayes',
    'metric': 
    {
        'goal': 'minimize', 
        'name': 'score'
        },
    'parameters': 
    {
        'score_lr': {'value': 0.0001},
        # 'score_bs': {'values': [1000, 2000, 2500]},
        'score_noise_init': {'max': 30., 'min': 5.},
        'score_noise_final': {'values':[0.03, 0.015, 0.1, 0.3]},
        'score_n_classes': {'values': [5, 8,10, 16, 18]},
        'hidden_layer_nums': {'values': [32, 64, 128, 256]},
     }
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='scones'
    )

def main():
    wandb.init(project='scones')
    score = train(wandb.config)
    wandb.log({'score': score})

wandb.agent(sweep_id, function=main, count=20)