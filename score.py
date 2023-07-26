import torch
from nets import FCNN, FCNN2
import os
import shutil
import numpy as np
import tqdm
from config import Config
import matplotlib.pyplot as plt
from utils import draw_model_params


class Score():
    def __init__(self, score_net, cnf, posterior_score=None):
        self.score_net = score_net
        self.cnf = cnf
        self.noise_scales = np.geomspace(start=cnf.score_noise_init, stop=cnf.score_noise_final, num=cnf.score_n_classes)
        self.num_classes = cnf.score_n_classes
        self.steps_per_class = cnf.score_steps_per_class
        self.sampling_lr = cnf.score_sampling_lr
        self.posterior_score = posterior_score
        self._noise_scales_th = torch.FloatTensor(self.noise_scales).to(cnf.device)

    def save(self, path, train_idx=None):
        if(os.path.exists(path)):
            shutil.rmtree(path)
        os.makedirs(path)
        state = self.score_net.state_dict()
        if(train_idx is None):
            torch.save(state, os.path.join(path, "score.pt"))
        else:
            torch.save(state, os.path.join(path, f"score_{train_idx}.pt"))

    def load(self, path):
        self.score_net.load_state_dict(torch.load(path))

    def score(self, x, noise_std=1):
        if(self.posterior_score is not None):
            return self.score_net(x) / noise_std + self.posterior_score(x)
        else:
            return self.score_net(x) / noise_std

    def dsm_loss(self, sample):
        noise = torch.randn([self.num_classes] + list(sample.size())).to(self.cnf.device)
        perturbed_samples = noise * self._noise_scales_th.reshape([-1, 1, 1]) + torch.stack([sample] * self.num_classes, dim = 0)
        d = self.cnf.target_dist.dim
        obj = (d/2) * torch.mean((self.score_net(perturbed_samples.view([-1, d])) + noise.view([-1, d]))**2)
        return obj

    def sample(self, size, Xs=None, ret_all = False):
        if ret_all:
            samples = []
        if Xs is None:
            Xs = torch.randn(size=[np.prod(size), 2]).to(self.cnf.device)
        with torch.no_grad():
            for s in self.noise_scales:
                for _ in range(self.steps_per_class):
                    a = self.sampling_lr * (s / self.noise_scales[-1])**2
                    noise = torch.randn(size=[np.prod(size), 2]).to(self.cnf.device)
                    Xs = Xs + a * self.score(Xs, s) + np.sqrt(2*a) * noise
                    samples.append(Xs.detach().cpu().numpy())
        # denoise via tweedie's identity
        Xs = Xs + self.noise_scales[-1]**2 * self.score(Xs, self.noise_scales[-1])
        
        return Xs.reshape(list(size) + [-1])

def init_score(cnf):
    d = cnf.target_dim
    T = FCNN(dims=[d, 2048, 2048, 2048, 2048, d], batchnorm=True).to(cnf.device)
    return Score(T, cnf)

def train_score(score, cnf, log_dir, run, verbose=True):
    bs = cnf.score_bs
    lr = cnf.score_lr
    iters = cnf.score_iters

    target_dist = cnf.target_dist

    opt = torch.optim.Adam(params=score.score_net.parameters(), lr=lr)

    if(verbose):
        t = tqdm.tqdm(total=iters, desc='', position=0)
    for i in range(iters):
        target_sample = torch.FloatTensor(target_dist.rvs(size=(bs,))).to(cnf.device)

        opt.zero_grad()
        obj = score.dsm_loss(target_sample)
        obj.backward()
        opt.step()

        if(verbose):
            t.set_description("Objective: {:.2E}".format(obj.item()))
            t.update(1)
        if run is not None:
            run.log({
                "mse_score": obj.item(),
            })
        if(i % 500 == 0):
            score.save(os.path.join(f"{log_dir}score", cnf.name), train_idx=i)
            score.save(os.path.join(f"{log_dir}score", cnf.name))

class GaussianScore():
    def __init__(self, gaussian, cnf):
        self.gaussian = gaussian
        self.cnf = cnf
        self.prec = torch.FloatTensor(gaussian.prec).to(cnf.device)
        self.dim = gaussian.dim

    def score(self, x, s=0):
        if(s > 0):
            prec = np.linalg.inv(self.gaussian.cov + s * np.eye(self.dim))
        else:
            prec = self.prec
        score = (-prec @ x.view((-1, self.dim, 1)))
        return score.view((-1, self.dim))


def train(config):
    cnf = Config("Swiss-Roll",
                 source="gaussian",
                 target="swiss-roll",
                 score_lr=config.score_lr,
                 score_iters=2000,
                 score_bs=2000,
                 score_noise_init=config.score_noise_init,
                 score_noise_final=config.score_noise_final,
                #  scones_iters=1000,
                #  scones_bs=1000,
                 device='cuda',
                 score_n_classes = config.score_n_classes,
                 score_steps_per_class = 20,
                 score_sampling_lr = 0.0001,
                 seed=2039)
    from  diagonal_matching import DiagonalMatching
    import wandb
    import time
    
    log_dir = 'logs/' + time.strftime("%Y-%m-%d/%H_%M_%S/", time.localtime())
    from pathlib import Path
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project="scones",
        config=cnf.__dict__,
        save_code=True, 
        name=cnf.name+time.strftime("%Y-%m-%d/%H:%M:%S", time.localtime()),
        dir=log_dir
    )
    run.define_metric("mse_score", summary="min")

    cnf.source_dist = DiagonalMatching(n_samples=cnf.scones_samples_per_source, mode='initial',easy=True)
    cnf.target_dist = DiagonalMatching(n_samples=cnf.scones_samples_per_source, mode='final', easy=True)
    ex_samples = cnf.target_dist.rvs(size=(1000,))
    Xs = cnf.source_dist.rvs(size=(1000,))
    score = init_score(cnf)
    train_score(score, cnf,log_dir, run, verbose=True)
    # score.load('/home/ljb/scones-synthetic/tools/logs/2023-07-24/21_09_53/score/Swiss-Roll/score.pt')
    # score.load(os.path.join("pretrained/score", cnf.name, "score.pt"))
    learned_samples = score.sample(size=(500,), ret_all=True).detach().cpu().numpy()
    
    def lerp_color(color1: str, color2: str, t: float) -> str:
            if color1.startswith('#'):
                color1 = color1[1:]
            if color2.startswith('#'):
                color2 = color2[1:]
            # 将16进制颜色值转换为RGB值
            rgb1 = tuple(int(color1[i:i+2], 16) for i in (0, 2, 4))
            rgb2 = tuple(int(color2[i:i+2], 16) for i in (0, 2, 4))

            # 对每个RGB通道进行线性插值
            rgb = tuple(int(rgb1[i] + t * (rgb2[i] - rgb1[i])) for i in range(3))

            # 将RGB值转换回16进制形式
            return '#{:02x}{:02x}{:02x}'.format(*rgb)

    def plot_matchings(fig, t0_points, t1_points, projection=lambda x: x, **kwargs):
        kwargs["color"] = kwargs.get("color", "gray")
        kwargs["alpha"] = kwargs.get("alpha", .7)
        # kwargs["lw"] = kwargs.get("lw", .2)
        extended_coords = np.concatenate([projection(t0_points), projection(t1_points)], axis=1)
        fig.plot(extended_coords[:,::2].T, extended_coords[:,1::2].T, zorder=0, **kwargs);
    
    fig, ax = plt.subplots()
    # ax.scatter(*ex_samples.T)
    ax.scatter(*ex_samples.T, color="#7B287D", alpha=0.5)
    ax.scatter(*Xs.T, color="#1d3557", alpha=0.5)
    plot_matchings(ax, Xs, ex_samples, lw=.2, alpha=.5)

    
    for i, sample in enumerate(learned_samples):
        ax.scatter(*sample.T, color=lerp_color("#50d67c","#0e5298", i/len(learned_samples)), alpha=0.5, s=0.5)
        if i == len(learned_samples) - 1:
            plot_matchings(ax, Xs, ex_samples, lw=.2, alpha=.5)
            
    # 打印出最后一个learned sample和target的MSE
    mse = np.mean((learned_samples[-1] - ex_samples)**2)
    print("MSE: ", mse)
    
    run.log(
        {
            'result':fig,
            'model': draw_model_params(score.score_net)
        }
    )
    # plt.show()
    fig.savefig(f'{log_dir}score.png')
    draw_model_params(score.score_net).savefig(f'{log_dir}score_model.png')
    return mse

if __name__ == "__main__":
    from easydict import EasyDict as edict
    train(edict({
        'score_lr': 0.0001,
        'score_bs': 100, 
        'score_noise_init': 1,
        'score_noise_final': 0.1,
        'score_n_classes': 10,
    }))