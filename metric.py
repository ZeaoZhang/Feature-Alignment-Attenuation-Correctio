import torch
import numpy as np
from scipy.spatial import cKDTree
from skimage import transform
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

class runningScore(object):
    def __init__(self):
        self.mse, self.pnsr, self.ssim = [], [], []

    def update(self, ac, ac_pred):
        # for index in range(acs.shape[0]):
        #     ac, ac_pred = acs[index], ac_preds[index]
        self.mse.append(mean_squared_error(ac, ac_pred))
        self.pnsr.append(peak_signal_noise_ratio(ac, ac_pred,
                            data_range=max(ac.max(), ac_pred.max()) - min(ac.min(), ac_pred.min())))
        self.ssim.append(structural_similarity(ac, ac_pred, 
                            data_range=max(ac.max(), ac_pred.max()) - min(ac.min(), ac_pred.min())))

    def get_scores(self):
        return {'mse': np.average(self.mse),
              'pnsr': np.average(self.pnsr),
              'ssim': np.average(self.ssim),}

    def reset(self):
        self.mse, self.pnsr, self.ssim = [], [], []


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

