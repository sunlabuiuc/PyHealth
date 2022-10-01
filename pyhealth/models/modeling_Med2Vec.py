import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm


class Med2Vec:
    def __init__(self, med2vec_dataset):
        super(Med2Vec, self).__init__()

        self.Med2VecDataset = med2vec_dataset
        self.vocabulary_size = med2vec_dataset.num_codes
        self.model = self.Med2VecModel(self.vocabulary_size)

    class Med2VecModel(nn.Module):
        def __init__(
            self, voc_size, demographics_size=0, embedding_size=256, hidden_size=512
        ):
            super().__init__()
            self.voc_size = voc_size
            self.embedding_size = embedding_size
            self.demographics_size = demographics_size
            self.hidden_size = hidden_size
            self.embedding_demo_size = self.embedding_size + self.demographics_size
            self.embedding_w = torch.nn.Parameter(
                torch.Tensor(self.embedding_size, self.voc_size)
            )
            torch.nn.init.uniform_(self.embedding_w, a=-0.1, b=0.1)
            self.embedding_b = torch.nn.Parameter(torch.Tensor(1, self.embedding_size))
            self.embedding_b.data.fill_(0)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.linear = nn.Linear(self.embedding_demo_size, self.hidden_size)
            self.probits = nn.Linear(self.hidden_size, self.voc_size)

            self.bce_loss = nn.BCEWithLogitsLoss()

        def embedding(self, x):
            return F.linear(x, self.embedding_w, self.embedding_b)

        def forward(self, x, d=torch.Tensor([])):
            x = self.embedding(x)
            x = self.relu1(x)
            emb = F.relu(self.embedding_w)

            if self.demographics_size > 0:
                x = torch.cat((x, d), dim=1)
            x = self.linear(x)
            x = self.relu2(x)
            probits = self.probits(x)
            return probits, emb

    class Med2VecDataLoader(DataLoader):
        def __init__(
            self,
            dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            collate_fn=default_collate,
        ):

            self.validation_split = validation_split
            self.shuffle = shuffle

            self.batch_idx = 0
            self.n_samples = len(dataset)

            self.sampler, self.valid_sampler = self._split_sampler(
                self.validation_split
            )

            self.init_kwargs = {
                "dataset": dataset,
                "batch_size": batch_size,
                "shuffle": shuffle,
                "collate_fn": collate_fn,
                "num_workers": num_workers,
            }
            super().__init__(sampler=self.sampler, **self.init_kwargs)

        def _split_sampler(self, split):
            """
            通过不同的 sampler 来实现数据集分割
            :param split:
            :return:
            """
            if split == 0.0:
                return None, None
            idx_full = np.arange(self.n_samples)
            # shuffle indexes only if shuffle is true
            # added for med2vec dataset where order matters
            if self.shuffle:
                np.random.seed(0)
                np.random.shuffle(idx_full)

            len_valid = int(self.n_samples * split)

            valid_idx = idx_full[0:len_valid]
            train_idx = np.delete(idx_full, np.arange(0, len_valid))

            if self.shuffle is True:
                train_sampler = SubsetRandomSampler(train_idx)
                valid_sampler = SubsetRandomSampler(valid_idx)
            else:
                train_sampler = SequentialSampler(train_idx)
                valid_sampler = SequentialSampler(valid_idx)

            # turn off shuffle option which is mutually exclusive with sampler
            self.shuffle = False
            self.n_samples = len(train_idx)

            return train_sampler, valid_sampler

        def split_validation(self):
            if self.valid_sampler is None:
                return None
            else:
                return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def get_loader(
        self, batch_size=1000, shuffle=False, validation_split=0.05, num_workers=0
    ):
        """returns torch.utils.data.DataLoader for Med2Vec dataset"""
        med2vec = self.Med2VecDataset
        data_loader = self.Med2VecDataLoader(
            dataset=med2vec,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_split=validation_split,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        return data_loader

    class Med2VecTrainer:
        def __init__(
            self,
            model,
            loss,
            metrics,
            optimizer,
            resume,
            data_loader,
            n_gpu=1,
            valid_data_loader=None,
            lr_scheduler=None,
            train_logger=None,
        ):

            self.model = model
            self.loss = loss
            self.metrics = metrics
            self.optimizer = optimizer
            self.resume = resume
            self.logger = logging.getLogger(self.__class__.__name__)
            self.data_loader = data_loader
            self.n_gpu = n_gpu
            self.valid_data_loader = valid_data_loader
            self.do_validation = self.valid_data_loader is not None
            self.lr_scheduler = lr_scheduler
            self.log_step = 10
            self.train_logger = train_logger
            self.mnt_mode = "min"
            self.mnt_metric = "val_loss"
            self.mnt_best = math.inf

            torch.cuda.set_device("cuda:0")
            self.device, device_ids = self.prepare_gpu(n_gpu)
            if len(device_ids) > 1:
                self.model = torch.nn.DataParallel(model, device_ids=device_ids)
                self.model.cuda()

            self.epochs = 1000
            self.save_period = 10
            self.verbosity = 2000
            self.early_stop = 10
            self.start_epoch = 1

            start_time = datetime.now().strftime("%m%d_%H%M%S")
            self.checkpoint_dir = os.path.join("m2v_ckpt", start_time)
            os.makedirs(self.checkpoint_dir)

            if resume:
                self._resume_checkpoint(resume)

        def prepare_gpu(self, n_gpu_use):

            n_gpu = torch.cuda.device_count()
            print("Num of available GPUs: ", n_gpu)
            if n_gpu_use > 0 and n_gpu == 0:
                self.logger.warning(
                    "Warning: There's no GPU available on this machine, training will be performed on CPU."
                )
                n_gpu_use = 0
            if n_gpu_use > n_gpu:
                self.logger.warning(
                    "Warning: The number of GPU's configured to use is {}, but only {} are available on this machine.".format(
                        n_gpu_use, n_gpu
                    )
                )
                n_gpu_use = n_gpu
            device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
            list_ids = list(range(n_gpu_use))
            return device, list_ids

        def train(self):
            """
            Full training logic
            """
            for epoch in tqdm(range(self.start_epoch, self.epochs + 1)):

                result = self._train_epoch(epoch)

                # save logged informations into log dict
                log = {"epoch": epoch}
                for key, value in result.items():
                    if key == "metrics":
                        log.update(
                            {
                                mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)
                            }
                        )
                    elif key == "val_metrics":
                        log.update(
                            {
                                "val_" + mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)
                            }
                        )
                    else:
                        log[key] = value

                self.logger.info(f"train_log: {log}")
                # print logged informations to the screen
                if self.train_logger is not None:
                    self.train_logger.add_entry(log)
                    if self.verbosity >= 1:
                        for key, value in log.items():
                            self.logger.info("    {:15s}: {}".format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                best = True
                if self.mnt_mode != "off":
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (
                            self.mnt_mode == "min"
                            and log[self.mnt_metric] < self.mnt_best
                        ) or (
                            self.mnt_mode == "max"
                            and log[self.mnt_metric] > self.mnt_best
                        )
                    except KeyError:
                        self.logger.warning(
                            "Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(
                                self.mnt_metric
                            )
                        )
                        self.mnt_mode = "off"
                        improved = False
                        not_improved_count = 0

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                        self.logger.info(
                            f"update best epoch according to {self.mnt_metric} = {self.mnt_best}"
                        )
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        self.logger.info(
                            "Validation performance didn't improve for {} epochs. Training stops.".format(
                                self.early_stop
                            )
                        )
                        break

                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch, save_best=best)

        def _eval_metrics(self, output, target, **kwargs):
            acc_metrics = np.zeros(len(self.metrics))
            for i, metric in enumerate(self.metrics):
                acc_metrics[i] += metric(output, target, **kwargs)
                self.logger.info(f"{metric.__name__} {acc_metrics[i]}")
            return acc_metrics

        def _train_epoch(self, epoch):

            self.model.train()
            total_loss = 0
            total_metrics = np.zeros(len(self.metrics))
            """
            trainer相对于一般的训练过程的不同点在于这个特殊任务计算loss需要特别的输入和输出整合
            """
            for batch_idx, (x, ivec, jvec, mask, d) in enumerate(self.data_loader):
                data, ivec, jvec, mask, d = (
                    x.to(self.device),
                    ivec.to(self.device),
                    jvec.to(self.device),
                    mask.to(self.device),
                    d.to(self.device),
                )
                self.optimizer.zero_grad()
                probits, emb_w = self.model(data.float(), d)  # 每个visit的预测输出
                # 计算输出到周围visit的loss
                loss_dict = self.loss(
                    data,
                    mask.float(),
                    probits,
                    nn.BCEWithLogitsLoss(),
                    emb_w,
                    ivec,
                    jvec,
                    window=5,
                )
                loss = loss_dict["visit_loss"] + loss_dict["code_loss"]  # 不同级别的loss相加
                loss.backward()  # 前馈计算梯度
                self.optimizer.step()  # 更新参数

                # self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
                # 记录结果
                self.logger.info(
                    f'train: loss {loss.item()}, visit loss {loss_dict["visit_loss"]}, code loss {loss_dict["code_loss"]}'
                )
                total_metrics += self._eval_metrics(
                    probits.detach(),
                    data.detach(),
                    mask=mask,
                )

                self.logger.info(f"batch {batch_idx} in epoch {epoch}...")
                if self.verbosity >= 2 and (batch_idx % self.log_step == 0):
                    self.logger.info(
                        "Train Epoch: {} [{}/{} ({:.0f}%)] {}: {:.6f}, {}: {:.6f}".format(
                            epoch,
                            batch_idx * self.data_loader.batch_size,
                            self.data_loader.n_samples,
                            100.0 * batch_idx / len(self.data_loader),
                            "visit_loss",
                            loss_dict["visit_loss"],
                            "code_loss",
                            loss_dict["code_loss"],
                        )
                    )
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                total_loss += (
                    loss_dict["visit_loss"].detach() + loss_dict["code_loss"].detach()
                )

            log = {
                "loss": total_loss / len(self.data_loader),  # log 一个epoch的平均 loss
                "metrics": (total_metrics / len(self.data_loader)).tolist(),
            }

            if self.do_validation:
                val_log = self._valid_epoch(epoch)
                log = {**log, **val_log}

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            return log

        def _valid_epoch(self, epoch):

            self.model.eval()
            total_val_loss = 0
            total_val_metrics = np.zeros(len(self.metrics))
            with torch.no_grad():
                for batch_idx, (x, ivec, jvec, mask, d) in enumerate(
                    self.valid_data_loader
                ):
                    self.logger.info(f"batch {batch_idx} in validation")
                    data, ivec, jvec, mask, d = (
                        x.to(self.device),
                        ivec.to(self.device),
                        jvec.to(self.device),
                        mask.to(self.device),
                        d.to(self.device),
                    )
                    probits, emb_w = self.model(data.float(), d)
                    loss_dict = self.loss(
                        data,
                        mask.float(),
                        probits,
                        nn.BCEWithLogitsLoss(),
                        emb_w,
                        ivec,
                        jvec,
                        window=5,
                    )
                    loss = loss_dict["visit_loss"] + loss_dict["code_loss"]
                    self.logger.info(
                        f'valid: loss {loss}, visit loss {loss_dict["visit_loss"]}, code loss {loss_dict["code_loss"]}'
                    )
                    # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    # self.writer.add_scalar('loss', loss.item())
                    total_val_loss += loss.item()
                    total_val_metrics += self._eval_metrics(
                        probits.detach(),
                        data.detach(),
                        mask=mask,
                        **{"k": 10, "window": 3},
                    )
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            return {
                "val_loss": total_val_loss / len(self.valid_data_loader),
                "val_metrics": (
                    total_val_metrics / len(self.valid_data_loader)
                ).tolist(),
            }

        def _save_checkpoint(self, epoch, save_best=False):
            """
            Save checkpoints
            """
            state = {
                "arch": type(self.model).__name__,
                "epoch": epoch,
                "logger": self.train_logger,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "monitor_best": self.mnt_best,
            }
            filename = os.path.join(
                self.checkpoint_dir, "checkpoint-epoch{}.pth".format(epoch)
            )
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
            if save_best:
                best_path = os.path.join(self.checkpoint_dir, "model_best.pth")
                torch.save(state, best_path)
                self.logger.info("Saving current best: {} ...".format("model_best.pth"))

        def _resume_checkpoint(self, resume_path):
            """
            Resume from saved checkpoints
            """
            self.logger.info("Loading checkpoint: {} ...".format(resume_path))
            checkpoint = torch.load(resume_path)
            self.start_epoch = checkpoint["epoch"] + 1
            self.mnt_best = checkpoint["monitor_best"]

            self.model.load_state_dict(checkpoint["state_dict"])

            # load optimizer state from checkpoint only when optimizer type is not changed.

            self.optimizer.load_state_dict(checkpoint["optimizer"])

            self.train_logger = checkpoint["logger"]
            self.logger.info(
                "Checkpoint '{}' (epoch {}) loaded".format(
                    resume_path, self.start_epoch
                )
            )

    def train(self, n_gpu=3):

        data_loader = self.get_loader()
        valid_data_loader = data_loader.split_validation()

        model = self.model

        # get function handles of loss and metrics
        loss = med2vec_loss
        metrics = [recall_k]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(
            trainable_params, lr=0.001, weight_decay=0, amsgrad=True
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=50, gamma=0.1
        )

        trainer = self.Med2VecTrainer(
            model,
            loss,
            metrics,
            optimizer,
            resume=False,
            data_loader=data_loader,
            valid_data_loader=valid_data_loader,
            lr_scheduler=lr_scheduler,
            n_gpu=n_gpu,
        )

        logging.info("training ...")
        trainer.train()

    def test(self, resume=None, n_gpu=1):
        data_loader = self.get_loader(
            batch_size=512, shuffle=False, validation_split=0.0, num_workers=2
        )

        loss_fn = med2vec_loss
        metrics = [recall_k]
        if resume is not None:
            checkpoint = torch.load(resume)
            state_dict = checkpoint["state_dict"]

            model = self.Med2VecModel(self.vocabulary_size)
            #         model.summary()
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
            model.load_state_dict(state_dict)
            # prepare model for testing
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
        else:
            model = self.model

        total_loss = 0.0
        total_metrics = torch.zeros(len(metrics))

        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(data_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data)

                loss = loss_fn(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metrics):
                    total_metrics[i] += metric(output, target) * batch_size

        n_samples = len(data_loader.sampler)
        log = {"loss": total_loss / n_samples}
        log.update(
            {
                met.__name__: total_metrics[i].item() / n_samples
                for i, met in enumerate(metrics)
            }
        )
        print(log)


def med2vec_loss(inputs, mask, probits, bce_loss, emb_w, ivec, jvec, window=1):
    def visit_loss(x, mask, probits, window=1):

        loss = 0
        for i in range(1, window + 1):
            if loss != loss:
                import pdb

                pdb.set_trace()
            l = mask.shape[0]
            _shape = list(mask.shape)
            _shape[0] = l - i
            maski = torch.ones(_shape, device=mask.device)
            for j in range(0, i + 1):
                maski = maski * mask[i - j : l - j]
            backward_preds = probits[i:] * maski
            forward_preds = probits[:-i] * maski
            #
            loss += bce_loss(forward_preds, x[i:].float()) + bce_loss(
                backward_preds, x[:-i].float()
            )
        return loss

    def code_loss(emb_w, ivec, jvec, eps=1.0e-6):
        norm = torch.sum(
            torch.exp(torch.mm(emb_w.t(), emb_w)), dim=1
        )  # normalize embedding
        cost = -torch.log(
            (
                torch.exp(torch.sum(emb_w[:, ivec].t() * emb_w[:, jvec].t(), dim=1))
                / norm[ivec]
            )
            + eps
        )
        cost = torch.mean(cost)
        return cost

    vl = visit_loss(inputs, mask, probits, window=window)
    cl = code_loss(emb_w, ivec, jvec, eps=1.0e-6)
    return {"visit_loss": vl, "code_loss": cl}


def recall_k(output, target, mask, k=10, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    rates = []
    mask = mask.squeeze()
    for i in range(1, window + 1):
        mask_len = mask.shape[0]
        _shape = list(mask.shape)
        _shape[0] = mask_len - i
        maski = torch.ones(_shape).to(mask.device)
        for j in range(0, i + 1):
            maski = maski * mask[i - j : mask_len - j]
        maski = torch.nn.functional.pad(maski, (i, i))

        tm = maski[:-i] == 1
        im = maski[i:] == 1

        target_mask = torch.masked_select(idx, tm == 1)
        input_mask = torch.masked_select(idx, im == 1)

        masked_output = output[input_mask, :]
        masked_output = masked_output.float()
        masked_target = target[target_mask, :]
        masked_target = masked_target.float()

        _, tk = torch.topk(masked_output, k)
        tt = torch.gather(masked_target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / torch.sum(masked_target, dim=1))
        rates.append(r)
    return torch.tensor(rates, device=output.device).mean()


def collate_fn(data):
    x, ivec, jvec, d = zip(*data)  # x: one-hot, 1 x num_vocab -stack-> n x num_vocab
    x = torch.stack(x, dim=0)  # list of tensor to tensor with additional dimension
    mask = torch.sum(x, dim=1) > 0
    mask = mask[:, None]  # additional dimension
    ivec = torch.cat(ivec, dim=0)
    jvec = torch.cat(jvec, dim=0)
    d = torch.stack([torch.tensor(_) for _ in d], dim=0)

    return x, ivec, jvec, mask, d
