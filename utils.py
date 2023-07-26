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

def plot_matchings(ax, t0_points, t1_points, projection=lambda x: x, **kwargs):
    kwargs["color"] = kwargs.get("color", "gray")
    kwargs["alpha"] = kwargs.get("alpha", .7)
    # kwargs["lw"] = kwargs.get("lw", .2)
    extended_coords = np.concatenate([projection(t0_points), projection(t1_points)], axis=1)
    ax.plot(extended_coords[:,::2].T, extended_coords[:,1::2].T, zorder=0, **kwargs);
    return ax


def draw_source2target(source, target):
    fig, ax = plt.subplots()
    ax.scatter(*source.T, color="#1d3557", alpha=0.5)
    ax.scatter(*target.T, color="#7B287D", alpha=0.5)
    ax = plot_matchings(ax, source, target, lw=.7)
    return fig    