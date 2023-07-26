import wandb
from score import train
sweep_configuration = {
    'method': 'bayes',
    'metric': 
    {
        'goal': 'minimize', 
        'name': 'score'
        },
    'parameters': 
    {
        'score_lr': {'max': 0.001, 'min': 0.0000001},
        # 'score_bs': {'values': [1000, 2000, 2500]},
        'score_noise_init': {'max': 30., 'min': 0.1},
        'score_noise_final': {'max': 0.1, 'min': 0.001},
        'score_n_classes': {'values': [2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40]},
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

wandb.agent(sweep_id, function=main, count=10)