import torch


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super(XTanhLoss, self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * torch.tanh(ey_t))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super(XSigmoidLoss, self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        # return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)
        return torch.mean(2 * ey_t * torch.sigmoid(ey_t) - ey_t)


class AlgebraicLoss(torch.nn.Module):
    def __init__(self):
        super(AlgebraicLoss, self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * ey_t / torch.sqrt(1 + ey_t * ey_t))


class Loss:
    LogCoshLoss = LogCoshLoss
    XTanhLoss = XTanhLoss
    XSigmoidLoss = XSigmoidLoss
    AlgebraicLoss = AlgebraicLoss
    MSELoss = torch.nn.MSELoss
    L1Loss = torch.nn.L1Loss
    SmoothL1Loss = torch.nn.SmoothL1Loss

    def build_loss(self, loss_type='mse'):
        loss_type = loss_type.lower()
        if loss_type == 'mse':
            return self.MSELoss()
        elif loss_type == 'l1':
            return self.L1Loss()
        elif loss_type == 'smoothl1':
            return self.SmoothL1Loss()
        elif loss_type == 'logcosh':
            return self.LogCoshLoss()
        elif loss_type == 'xtanh':
            return self.XTanhLoss()
        elif loss_type == 'xsigmiod':
            return self.XSigmoidLoss()
        elif loss_type == 'algebraic':
            return self.AlgebraicLoss()
        else:
            raise NotImplementedError('loss function not supported, now support: '
                                      '[MSE, L1, SmoothL1, LogCosh, XTanh, XSigmoid, Algebraic]')
