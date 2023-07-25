import numpy as np
import torch

class DiagonalMatching:

    def __init__(self, n_samples=None, mode='initial', easy=False):
        self.n_samples = n_samples
        self.mode = mode
        self.dim = 2
        self.easy = easy
        if n_samples is None and mode not in ['initial', 'final']:
            raise ValueError('mode must be either initial or final')
        self.data = self._gen_data()
        
    def _gen_data(self):
        # if not self.easy:
        #     n = size[0]//2+1
        # else:
        #     n = size[0]
        n = self.n_samples//2+1

        left_square = np.stack([np.random.uniform(-1.2, -1., size=(n,))*2-5, np.linspace(-.1, .5, n)*4+3], axis=1)
        right_square = np.stack([np.random.uniform(1., 1.2, size=(n,))*2+5, np.linspace(-.1, .5, n)*4+3], axis=1)

        top_square = np.stack([np.linspace(-.3, .3, n)*4, np.random.uniform(.8, 1., size=(n,))*2+3], axis=1)
        bottom_square = np.stack([np.linspace(-.3, .3, n)*4, np.random.uniform(-1.5, -1.3, size=(n,))*2-3], axis=1)
        return {
            "left": left_square,
            "right": right_square,
            "top": top_square,
            "bottom": bottom_square,
        }
        
        
    def sample(self):
        n = self.n_samples//2+1

        left_square = np.stack([np.random.uniform(-1.2, -1., size=(n,))*2-5, np.linspace(-.1, .5, n)*4+3], axis=1)
        right_square = np.stack([np.random.uniform(1., 1.2, size=(n,))*2+5, np.linspace(-.1, .5, n)*4+3], axis=1)

        top_square = np.stack([np.linspace(-.3, .3, n)*4, np.random.uniform(.8, 1., size=(n,))*2+3], axis=1)
        bottom_square = np.stack([np.linspace(-.3, .3, n)*4, np.random.uniform(-1.5, -1.3, size=(n,))*2-3], axis=1)

        rand_shuffling = np.random.permutation(self.n_samples)

        return {
            "initial": torch.from_numpy(np.concatenate([left_square, top_square], axis=0)[:self.n_samples][rand_shuffling]).float(),
            "final": torch.from_numpy(np.concatenate([right_square, bottom_square], axis=0)[:self.n_samples][rand_shuffling]).float()
        }
        
    def rvs(self, size=(1,)):
        if not self.easy:
            n = size[0]//2+1
        else:
            n = size[0]
        
        # rand_shuffling = np.random.permutation(size[0])
        rand_shuffling = np.arange(size[0])
            
        

        if self.mode == 'initial':
            left_square = np.stack([np.random.uniform(-1.1, -1., size=(n,))*2+5, np.linspace(-.1, .5, n)*4+3], axis=1)
            if not self.easy:
                top_square = np.stack([np.linspace(-.3, .3, n)*4, np.random.uniform(.8, 1., size=(n,))*2+3], axis=1)
                return np.concatenate([left_square, top_square], axis=0)[:size[0]][rand_shuffling]
            else:
                return left_square[:size[0]][rand_shuffling]
        elif self.mode == 'final':
            right_square = np.stack([np.random.uniform(1., 1.1, size=(n,))*2+15, np.linspace(-.1, .5, n)*4+3], axis=1)
            if not self.easy:
                bottom_square = np.stack([np.linspace(-.3, .3, n)*4, np.random.uniform(-1.5, -1.3, size=(n,))*2-3], axis=1)
                return np.concatenate([right_square, bottom_square], axis=0)[:size[0]][rand_shuffling]
            else:
                return right_square[:size[0]][rand_shuffling]
        else:
            raise ValueError('mode must be either initial or final')

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join('{}={!r}'.format(k, v) for k, v in self.__dict__.items())
        )
        
