import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, logger, scheduler, train_loader, val_loader, cuda=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.iters = 0
        self.best_result = 0
        self.cuda = cuda

    def _train_one_epoch(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        total_loss = 0
        for i, batch in enumerate(tbar):
            self.optimizer.zero_grad()
            imgs, targets = batch
            outputs = self.model(imgs.cuda() if self.cuda else imgs)
            try:
                targets = targets.cuda() if self.cuda else targets
            except AttributeError:
                for k, v in targets.items():
                    targets[k] = v.cuda() if self.cuda else v

            loss = self.criterion(outputs, targets)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            tbar.set_description('Train Loss: %.4f' % loss.item())
            self.logger.writter.add_scalar('Train/Loss', loss.item(), self.iters)
            self.iters += 1
        total_loss /= (i + 1)
        self.scheduler.step(epoch)
        lr = self.optimizer.param_groups[0]['lr']
        self.logger.log.info('Epoch: {}, Training Loss: {}, lr: {}'.format(epoch, total_loss, lr))
        self.logger.writter.add_scalar('Train/lr', lr, epoch)

    def _validation_one_epoch(self, epoch):
        with torch.no_grad():
            torch.cuda.empty_cache()
            self.model.eval()
            tbar = tqdm(self.val_loader)
            total_loss = 0
            for i, batch in enumerate(tbar):
                imgs, targets = batch
                outputs = self.model(imgs.cuda() if self.cuda else imgs)
                targets = targets.cuda() if self.cuda else targets
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                tbar.set_description('Validation Loss: %.4f' % loss.item())

            total_loss /= (i + 1)

            self.logger.log.info("Epoch:{}, Validation Loss: {}, ".format(epoch, total_loss))
        return total_loss

    def train(self, epochs):
        for epoch in range(epochs):
            self._train_one_epoch(epoch)
            metric = self._validation_one_epoch(epoch)
            states = {
                'arc': self.model,
                'state_dict': self.model.state_dict(),
                'epoch': epoch,
                'optimizer': self.optimizer.state_dict()
            }

            if metric > self.best_result:
                self.best_result = metric
                is_best = True
                self.logger.save_checkpoint(states, is_best, prefix=self.model._get_name() +
                                            '_%s_%.4f' % ('loss', metric))

            else:
                is_best = False
                self.logger.save_checkpoint(states, is_best, prefix=self.model._get_name() +
                                            '_%s_%.4f' % ('loss', metric))




