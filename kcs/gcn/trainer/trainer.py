import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.best_valid_metric = 10
        self.best_val_1 = 0
        self.best_val_acc = 0

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, mlp1, mlp2, mlp3) in enumerate(self.data_loader):
            X, A, target, mlp1, mlp2, mlp3 = data[0].to(self.device), \
                                             data[1].to(self.device), \
                                             target.to(self.device), \
                                             mlp1.to(self.device).float(), \
                                             mlp2.to(self.device).float(), \
                                             mlp3.to(self.device).float(), \

            self.optimizer.zero_grad()
            output = self.model(X, A, mlp1, mlp2, mlp3)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outputs = torch.zeros((1, 2)).to(self.device)
            targets = torch.zeros((1, )).to(self.device).long()
            for batch_idx, (data, target, mlp1, mlp2, mlp3) in enumerate(self.valid_data_loader):
                X, A, target, mlp1, mlp2, mlp3 = data[0].to(self.device), \
                                                 data[1].to(self.device), \
                                                 target.to(self.device), \
                                                 mlp1.to(self.device).float(), \
                                                 mlp2.to(self.device).float(), \
                                                 mlp3.to(self.device).float(), \

                output = self.model(X, A, mlp1, mlp2, mlp3)
                loss = self.criterion(output, target)
                
                outputs = torch.cat([outputs, output], dim=0)
                targets = torch.cat([targets, target.long()], dim=0)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            val_acc = self.metric_ftns[0](outputs[1:, :], targets[1:])
            val_f1  = self.metric_ftns[1](outputs[1:, :], targets[1:])
            val_loss = self.criterion(outputs[1:, :], targets[1:])
            
            
            
            self.valid_metrics.update('loss', val_loss)
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outputs, targets))
            
            if val_loss < self.best_valid_metric:
                self.best_valid_metric = val_loss.item()
                self.best_val_1 = val_f1
                self.best_val_acc = val_acc
            self.logger.info('------------------------------------')
            self.logger.info(f'Current Best val_f1: {self.best_val_1}, val_acc: {self.best_val_acc}, val_loss(F1_BCE): {self.best_valid_metric}')
            self.logger.info('------------------------------------')
            
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
