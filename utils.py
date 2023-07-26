import numpy as np
import matplotlib.pyplot as plt

def draw_model_params(model, show=False):
    def get_minial_rectangle(x: int)->tuple:
        # Get the smallest rectangle that can hold x squares
        a = int(np.sqrt(x))
        b = int(np.ceil(x/a))
        return (a, b)

    model_params = list(model.named_parameters())

    # 创建一个画布，其中可以容纳len(score.score_net.named_parameters())个子图，尽可能为正方形
    a, b = get_minial_rectangle(len(list(model_params)))
    fig, axs = plt.subplots(a, b, figsize=(20, 15))


    for index, (name,param) in enumerate(model_params):
        
        axs[int(index/b)][int(index%b)].title.set_text(name)
        axs[int(index/b)][int(index%b)].hist(param.detach().cpu().numpy().flatten(), bins=100, color='#524199', alpha=0.7)
    
    if show:    
        fig.show()
        
    return fig

