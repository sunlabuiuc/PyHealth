import os
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings


class CNN:
    def __init__(self, dataset, model='resnet'):
        super(CNN, self).__init__()
        self.dataset = dataset
        self.model = model
        self.trained_model = None

        split_ratio = 0.9
        idx = int(len(dataset) * split_ratio)
        self.train_dataset = []
        self.test_dataset = []
        self.cnn_train = None
        self.cnn_test = None
        self.test_dataloader = None
        self.train_dataloader = None
        for i in range(len(dataset)):
            item = dataset.__getitem__(i)
            if i <= idx:
                self.train_dataset.append(item)
            else:
                self.test_dataset.append(item)
        self.voc_size = dataset.voc_size
        self.max_visits = 0
        for patient in range(len(dataset)):
            length = len(dataset[patient]['conditions'])
            if length > self.max_visits:
                self.max_visits = length

    class CNNModel(nn.Module):
        def __init__(self, model, n_classes):
            super().__init__()
            if model == 'resnet':
                resnet = torchvision.models.resnext50_32x4d(pretrained=False)
                resnet.fc = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
                )
                self.model = resnet
                self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return self.sigmoid(self.model(x))

    class CNNDrugRecDataSet(Dataset):
        def __init__(self, dataset, voc_size, max_visits):

            condition_voc, procedure_voc, drug_voc = voc_size[0], voc_size[1], voc_size[2]
            features = condition_voc + procedure_voc
            print('Features are in shapes of ', max_visits, '*', features)

            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.Normalize(0.00030047886384355915,
                                     0.017331721677190905)
            ])

            input_data_list = []
            label_list = []
            for i in tqdm(range(len(dataset))):
                condition_procedure = np.zeros((max_visits, condition_voc + procedure_voc))
                drug_multi_hot = np.zeros(drug_voc)
                conditions_ = dataset[i]['conditions']
                procedures_ = dataset[i]['procedures']
                drugs_ = dataset[i]['drugs']

                # inputs
                for j in range(len(conditions_)):
                    condition = conditions_[j]
                    for k in range(len(condition)):
                        condition_procedure[j][condition[k]] = 1
                for m in range(len(procedures_)):
                    procedure = procedures_[m]
                    for n in range(len(procedure)):
                        condition_procedure[m][procedure[n] + condition_voc] = 1
                input_data_list.append(condition_procedure)

                # labels
                for p in range(len(drugs_)):
                    drug = drugs_[p]
                    for q in range(len(drug)):
                        drug_multi_hot[drug[q]] = 1
                label_list.append(drug_multi_hot)

            self.inputs = np.array(input_data_list, dtype=int)
            self.labels = np.array(label_list, dtype=float)

        def __getitem__(self, patient):
            x = self.inputs[patient]
            y = self.labels[patient]
            l = []
            for i in range(3):
                l.append(x)
            l = np.array(l)
            c = torch.from_numpy(l).float()
            x = self.transform(c)
            return x, y

        def __len__(self):
            return len(self.inputs)

    def train(self, lr=1e-4, batch_size=32, save_freq=5, test_freq=200, max_epoch=35,
              save_path="cnn_ckpt/", logdir='logs/cnn_logs/', epoch=0, iteration=0, 
              task='drug_rec', n_gpus=5, save=False):
        if task == 'drug_rec':
            print('--------prepare training data---------')
            self.cnn_train = self.CNNDrugRecDataSet(self.train_dataset, self.voc_size, self.max_visits)
            print('--------prepare testing data----------')
            self.cnn_test = self.CNNDrugRecDataSet(self.test_dataset, self.voc_size, self.max_visits)
        self.train_dataloader = DataLoader(self.cnn_train, batch_size=batch_size, shuffle=False, drop_last=True)
        self.test_dataloader = DataLoader(self.cnn_test, batch_size=batch_size, drop_last=True)
        num_train_batches = int(np.ceil(len(self.train_dataset) / batch_size))
        n_classes = len(self.cnn_train[0][1])

        model = self.CNNModel(self.model, n_classes)
        model.train()
        device, device_ids = prepare_gpu(n_gpu_use=n_gpus)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            model.cuda()
        model.to(f'cuda:{model.device_ids[0]}')

        os.makedirs(save_path, exist_ok=True)
        logger = SummaryWriter(logdir)

        warnings.filterwarnings('ignore')

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        criterion = nn.BCELoss()
        
        print('-------- Start Training ----------')
        while True:
            batch_losses = []
            for inputs, targets in tqdm(self.train_dataloader):
                inputs, targets = inputs.to(f'cuda:{model.device_ids[0]}'), targets.to(f'cuda:{model.device_ids[0]}')

                optimizer.zero_grad()

                model_result = model(inputs)
                loss = criterion(model_result, targets.type(torch.float))

                batch_loss_value = loss.item()
                loss.backward()
                optimizer.step()

                logger.add_scalar('train_loss', batch_loss_value, iteration)
                batch_losses.append(batch_loss_value)
                with torch.no_grad():
                    result = calculate_metrics(model_result.cpu().numpy(), targets.cpu().numpy())
                    for metric in result:
                        logger.add_scalar('train/' + metric, result[metric], iteration)

                if iteration % test_freq == 0:
                    model.eval()
                    with torch.no_grad():
                        model_result = []
                        targets = []
                        for inputs, batch_targets in self.test_dataloader:
                            inputs = inputs.to(device)
                            model_batch_result = model(inputs)
                            model_result.extend(model_batch_result.cpu().numpy())
                            targets.extend(batch_targets.cpu().numpy())

                    result = calculate_metrics(np.array(model_result), np.array(targets))
                    for metric in result:
                        logger.add_scalar('test/' + metric, result[metric], iteration)
                    print("epoch:{:2d} iter:{:3d} test: "
                          "micro f1: {:.3f} "
                          "macro f1: {:.3f} "
                          "samples f1: {:.3f}".format(epoch, iteration,
                                                      result['micro/f1'],
                                                      result['macro/f1'],
                                                      result['samples/f1']))

                    model.train()
                iteration += 1

            loss_value = np.mean(batch_losses)
            print("epoch:{:2d} iter:{:3d} train: loss:{:.3f}".format(epoch, iteration, loss_value))
            if (epoch % save_freq  == 0) and (save == True):
                checkpoint_save(model, save_path, epoch)
            epoch += 1
            if max_epoch < epoch:
                self.trained_model = model
                break

    def eval(self, num_sample):
        # Run inference on the test data
        self.trained_model.eval()
        sample = 0
        for inputs, targets in self.train_dataloader:
            sample += 1
            with torch.no_grad():
                raw_pred = self.trained_model(inputs).cpu().numpy()[0]
                raw_pred = np.array(raw_pred > 0.5, dtype=float)
                print(raw_pred)
            if sample > num_sample:
                break


def prepare_gpu(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    print('Num of available GPUs: ', n_gpu)
    if n_gpu_use > 0 and n_gpu == 0:
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }


def checkpoint_save(model, save_path, epoch):
    f = os.path.join(save_path, 'checkpoint-{:06d}.pth'.format(epoch))
    if 'module' in dir(model):
        torch.save(model.module.state_dict(), f)
    else:
        torch.save(model.state_dict(), f)
    print('saved checkpoint:', f)


