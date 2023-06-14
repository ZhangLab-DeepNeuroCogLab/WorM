import os
import torch

from src.model import WM_Model

from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, config, model, 
                 train_dataloader, val_dataloader, test_dataloader, gen_test_dataloader, 
                 wandb, device, logger):

        self.config = config
        self.model = model

        self.wandb = wandb
        self.device = device
        self.logger = logger

        self.task_list = list(train_dataloader.keys())
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.gen_test_dataloader = gen_test_dataloader

        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info('Total Trainable Parameters: {}'.format(total_trainable_params))

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info('Total Parameters: {}'.format(total_params))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                    mode="min", 
                                                                    factor=0.8,
                                                                    patience=3, 
                                                                    threshold=0.005,
                                                                    verbose=True)

        self.CE_criteria = torch.nn.CrossEntropyLoss()
        self.BCE_criteria = torch.nn.BCEWithLogitsLoss()

        if config.resume:
            checkpoint = torch.load(
                os.path.join(config.model_path, 
                            'model_'+str(config.resume_epoch).zfill(3)+'.pt'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_epoch = checkpoint['best_epoch']
            self.min_val_multitask_loss = checkpoint['min_val_multitask_loss']
            self.max_multitask_val_acc = checkpoint['max_multitask_val_acc']

    def train(self):
        if self.config.resume:
            min_val_multitask_loss = self.min_val_multitask_loss
            max_multitask_val_acc = self.max_multitask_val_acc
            best_epoch = self.best_epoch
            start_epoch = int(self.config.resume_epoch) + 1
        else:
            min_val_multitask_loss = float('inf')
            max_multitask_val_acc = float('-inf')
            best_epoch = 0        
            start_epoch = 1

        for epoch in range(start_epoch, self.config.num_epochs+1):
            self.logger.info('Epoch: {}'.format(epoch))

            epoch_train_multitask_loss, epoch_train_task_loss_dict = self.train_one_epoch()
            epoch_val_multitask_loss, epoch_val_task_loss_dict, epoch_val_task_acc_dict = self.val_one_epoch()

            self.scheduler.step(epoch_val_multitask_loss)
            
            avg_multitask_val_acc = sum(epoch_val_task_acc_dict.values())/len(epoch_val_task_acc_dict)
            if avg_multitask_val_acc > max_multitask_val_acc:
                min_val_multitask_loss = epoch_val_multitask_loss
                min_val_multitask_loss_dict = epoch_val_task_loss_dict
                max_multitask_val_acc_dict = epoch_val_task_acc_dict
                max_multitask_val_acc = avg_multitask_val_acc
                best_epoch = epoch

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_multitask_loss': epoch_train_multitask_loss,
                    'train_task_loss_dict': epoch_train_task_loss_dict,
                    'val_multitask_loss': epoch_val_multitask_loss,
                    'val_task_loss_dict': epoch_val_task_loss_dict,
                    'val_multitask_acc': avg_multitask_val_acc,
                    'val_task_acc_dict': epoch_val_task_acc_dict,
                    'min_val_multitask_loss': min_val_multitask_loss,
                    'min_val_multitask_loss_dict': min_val_multitask_loss_dict,
                    'max_multitask_val_acc': max_multitask_val_acc,
                    'max_multitask_val_acc_dict': max_multitask_val_acc_dict,
                    'save_condition': 'max_multitask_val_acc',
                    'best_epoch': best_epoch,
                    'config': vars(self.config)
                }, os.path.join(self.config.model_path, 'model.pt'))

            if epoch % self.config.test_interval == 0:
                self.logger.info('Testing at Epoch: {}'.format(epoch))
                self.test(epoch)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(), 
                    'optimizer_state_dict': self.optimizer.state_dict(), 
                    'train_multitask_loss': epoch_train_multitask_loss,
                    'train_task_loss_dict': epoch_train_task_loss_dict,
                    'val_multitask_loss': epoch_val_multitask_loss,
                    'val_task_loss_dict': epoch_val_task_loss_dict,
                    'val_multitask_acc': avg_multitask_val_acc,
                    'val_task_acc_dict': epoch_val_task_acc_dict, 
                    'min_val_multitask_loss': min_val_multitask_loss,
                    'min_val_multitask_loss_dict': min_val_multitask_loss_dict,
                    'max_multitask_val_acc': max_multitask_val_acc,
                    'max_multitask_val_acc_dict': max_multitask_val_acc_dict, 
                    'save_condition': 'every_10', 
                    'best_epoch': best_epoch,
                    'config': vars(self.config)
                }, os.path.join(self.config.model_path, 'model_'+str(epoch).zfill(3)+'.pt'))


            torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(), 
                    'optimizer_state_dict': self.optimizer.state_dict(), 
                    'train_multitask_loss': epoch_train_multitask_loss,
                    'train_task_loss_dict': epoch_train_task_loss_dict,
                    'val_multitask_loss': epoch_val_multitask_loss,
                    'val_task_loss_dict': epoch_val_task_loss_dict,
                    'val_multitask_acc': avg_multitask_val_acc,
                    'val_task_acc_dict': epoch_val_task_acc_dict, 
                    'min_val_multitask_loss': min_val_multitask_loss,
                    'max_multitask_val_acc': max_multitask_val_acc,
                    'save_condition': 'curr_epoch', 
                    'best_epoch': best_epoch,
                    'config': vars(self.config)
                }, os.path.join(self.config.model_path, 'model_curr_epoch.pt'))

            if self.config.resume:
                self.logger.info('##Resumed## Epoch: {}, Train Multitask Loss: {}, Train Task Loss Dict: {},\
                                Val Multitask Loss: {}, Val Task Loss Dict: {}, Val Multitask Acc: {},\
                                Val Task Acc Dict: {}'.format(epoch, epoch_train_multitask_loss, epoch_train_task_loss_dict, 
                                epoch_val_multitask_loss, epoch_val_task_loss_dict, avg_multitask_val_acc, epoch_val_task_acc_dict))
            else:
                self.logger.info('Epoch: {}, Train Multitask Loss: {}, Train Task Loss Dict: {},\
                                Val Multitask Loss: {}, Val Task Loss Dict: {}, Val Multitask Acc: {},\
                                Val Task Acc Dict: {}'.format(epoch, epoch_train_multitask_loss, epoch_train_task_loss_dict, 
                                epoch_val_multitask_loss, epoch_val_task_loss_dict, avg_multitask_val_acc, epoch_val_task_acc_dict))
            
            self.wandb.log({'Train Multitask Loss': epoch_train_multitask_loss, 'epoch': epoch})
            self.wandb.log({'Val Multitask Loss': epoch_val_multitask_loss, 'epoch': epoch})
            self.wandb.log({'Val Multitask Acc': avg_multitask_val_acc, 'epoch': epoch})
            for task in epoch_train_task_loss_dict:
                self.wandb.log({'Train '+task+' Loss': epoch_train_task_loss_dict[task], 'epoch': epoch})
                self.wandb.log({'Val '+task+' Loss': epoch_val_task_loss_dict[task], 'epoch': epoch})
                self.wandb.log({'Val '+task+' Acc': epoch_val_task_acc_dict[task], 'epoch': epoch})


        self.logger.info('Best Epoch: {}, Min Val Multitask Loss: {}, Min Val Multitask Loss Dict: {},\
                         Max Multitask Val Acc: {}, Max Multitask Val Acc Dict: {}'.format(best_epoch, min_val_multitask_loss, 
                         min_val_multitask_loss_dict, max_multitask_val_acc, max_multitask_val_acc_dict))
        
    def test(self, epoch='Complete'):
        best_model = torch.load(os.path.join(self.config.model_path, 'model.pt'))
        
        test_model = WM_Model(self.config, self.device)
        test_model.to(self.device)

        test_model.load_state_dict(best_model['model_state_dict'])

        test_epoch_acc_dict = self.test_one_epoch(test_model)
        self.logger.info('Test Acc Dict (Epoch - {}): {}'.format(
            str(epoch), test_epoch_acc_dict))

        self.viz_test(epoch, test_epoch_acc_dict)

    def train_one_epoch(self):
        train_dataloader = zip(*self.train_dataloader.values())

        self.model.train()

        epoch_train_multitask_loss = 0
        epoch_train_task_loss_dict = {task: 0 for task in self.task_list}

        for batch_index, multi_task_batch in tqdm(enumerate(train_dataloader)):

            self.optimizer.zero_grad()

            # SC Task
            stim_batch, resp_batch, seq_len, _ = multi_task_batch[0]

            stim_batch = stim_batch.to(self.device)
            resp_batch = resp_batch.to(self.device)

            out, _, _, _, _ = self.model(stim_batch, 'SC_Task', seq_len)

            out = out[torch.arange(out.size(0)), seq_len-1, :]
            resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
            sc_task_loss = self.BCE_criteria(out, resp_batch.float())
            
            # SFR Task
            stim_batch, resp_batch, seq_len, _ = multi_task_batch[1]

            stim_batch = stim_batch.to(self.device)
            resp_batch = resp_batch.to(self.device)

            out, _, _, _, _ = self.model(stim_batch, 'SFR_Task', seq_len)

            out = out[torch.arange(out.size(0)), seq_len-1, :]
            sfr_task_loss = self.BCE_criteria(out, resp_batch.float())

            # SI Task
            stim_batch, resp_batch, seq_len, _ = multi_task_batch[2]

            stim_batch = stim_batch.to(self.device)
            resp_batch = resp_batch.to(self.device)

            out, _, _, _, _ = self.model(stim_batch, 'SI_Task', seq_len)

            out = out[torch.arange(out.size(0)), seq_len-1, :]
            resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
            si_task_loss = self.BCE_criteria(out, resp_batch.float())

            # SMU Task
            stim_batch, resp_batch, seq_len, set_size = multi_task_batch[3]

            stim_batch = stim_batch.to(self.device)
            resp_batch = resp_batch.to(self.device)

            out, _, _, _, _ = self.model(stim_batch, 'SMU_Task', seq_len)

            out = torch.cat([out[i, seq_len[i]-set_size[i]:seq_len[i]] for i in range(len(seq_len))], dim=0)
            resp_batch = torch.cat([resp_batch[i, seq_len[i]-set_size[i]:seq_len[i]] for i in range(len(seq_len))], dim=0)
            smu_task_loss = self.CE_criteria(out, resp_batch)

            # STS Task
            stim_batch, resp_batch, seq_len = multi_task_batch[4]

            stim_batch = stim_batch.to(self.device)
            resp_batch = resp_batch.to(self.device)

            out, _, _, _, _ = self.model(stim_batch, 'STS_Task', seq_len)

            out = torch.cat([out[i, 0:seq_len[i]] for i in range(len(seq_len))], dim=0)
            resp_batch = torch.cat([resp_batch[i, 0:seq_len[i]] for i in range(len(seq_len))], dim=0)
            sts_task_loss = self.CE_criteria(out, resp_batch)

            # VIR Task
            stim_batch, resp_batch, seq_len, _, _ = multi_task_batch[5]

            stim_batch = stim_batch.to(self.device)
            resp_batch = resp_batch.to(self.device)

            out, _, _, _, _ = self.model(stim_batch, 'VIR_Task', seq_len)

            out = out[torch.arange(out.size(0)), seq_len-1, :]
            resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
            vir_task_loss = self.BCE_criteria(out, resp_batch.float())

            # VSR Task
            stim_batch, resp_batch, seq_len, list_length = multi_task_batch[6]

            stim_batch = stim_batch.to(self.device)
            resp_batch = resp_batch.to(self.device)

            out, _, _, _, _ = self.model(stim_batch, 'VSR_Task', seq_len)

            out = torch.cat([out[i, seq_len[i]-list_length[i]:seq_len[i]] for i in range(len(seq_len))], dim=0)
            resp_batch = torch.cat([resp_batch[i, seq_len[i]-list_length[i]:seq_len[i]] for i in range(len(seq_len))], dim=0)
            vsr_task_loss = self.CE_criteria(out, resp_batch)

            # VSRec Task
            stim_batch, resp_batch, seq_len, list_length, _ = multi_task_batch[7]

            stim_batch = stim_batch.to(self.device)
            resp_batch = resp_batch.to(self.device)

            out, _, _, _, _ = self.model(stim_batch, 'VSRec_Task', seq_len)

            out = torch.cat([out[i, seq_len[i]-list_length[i]:seq_len[i]] for i in range(len(seq_len))], dim=0)
            resp_batch = torch.cat([resp_batch[i, seq_len[i]-list_length[i]:seq_len[i]] for i in range(len(seq_len))] , dim=0).unsqueeze(1)
            vsrec_task_loss = self.BCE_criteria(out, resp_batch.float())

            # CD Color Task
            stim_batch, resp_batch, seq_len, _, _ = multi_task_batch[8]

            stim_batch = stim_batch.to(self.device)
            resp_batch = resp_batch.to(self.device)

            out, _, _, _, _ = self.model(stim_batch, 'CD_Color_Task', seq_len)

            out = out[torch.arange(out.size(0)), seq_len-1, :]
            resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
            cd_color_task_loss = self.BCE_criteria(out, resp_batch.float())

            # CD Orientation Task
            stim_batch, resp_batch, seq_len, _, _ = multi_task_batch[9]

            stim_batch = stim_batch.to(self.device)
            resp_batch = resp_batch.to(self.device)

            out, _, _, _, _ = self.model(stim_batch, 'CD_Orientation_Task', seq_len)

            out = out[torch.arange(out.size(0)), seq_len-1, :]
            resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
            cd_orientation_task_loss = self.BCE_criteria(out, resp_batch.float())

            # CD Size Task
            stim_batch, resp_batch, seq_len, _, _ = multi_task_batch[10]

            stim_batch = stim_batch.to(self.device)
            resp_batch = resp_batch.to(self.device)

            out, _, _, _, _ = self.model(stim_batch, 'CD_Size_Task', seq_len)

            out = out[torch.arange(out.size(0)), seq_len-1, :]
            resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
            cd_size_task_loss = self.BCE_criteria(out, resp_batch.float())

            # CD Gap Task
            stim_batch, resp_batch, seq_len, _, _ = multi_task_batch[11]

            stim_batch = stim_batch.to(self.device)
            resp_batch = resp_batch.to(self.device)

            out, _, _, _, _ = self.model(stim_batch, 'CD_Gap_Task', seq_len)

            out = out[torch.arange(out.size(0)), seq_len-1, :]
            resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
            cd_gap_task_loss = self.BCE_criteria(out, resp_batch.float())

            # CD Conj Task
            stim_batch, resp_batch, seq_len, _, _, _ = multi_task_batch[12]

            stim_batch = stim_batch.to(self.device)
            resp_batch = resp_batch.to(self.device)

            out, _, _, _, _ = self.model(stim_batch, 'CD_Conj_Task', seq_len)

            out = out[torch.arange(out.size(0)), seq_len-1, :]
            resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
            cd_conj_task_loss = self.BCE_criteria(out, resp_batch.float())

            # CS Task
            stim_batch, resp_batch, seq_len, _, _ = multi_task_batch[13]

            stim_batch = stim_batch.to(self.device)
            resp_batch = resp_batch.to(self.device)

            out, _, _, _, _ = self.model(stim_batch, 'CS_Task', seq_len)

            out = torch.cat([out[i, 0:seq_len[i]] for i in range(len(seq_len))], dim=0)
            resp_batch = torch.cat([resp_batch[i, 0:seq_len[i]] for i in range(len(seq_len))], dim=0)
            cs_task_loss = self.CE_criteria(out, resp_batch)
            
            # Multitask Loss
            multitask_loss = (sc_task_loss + sfr_task_loss + si_task_loss + smu_task_loss + 
                              sts_task_loss + vir_task_loss + 
                              vsr_task_loss + vsrec_task_loss + 
                              cd_color_task_loss + cd_orientation_task_loss + 
                              cd_size_task_loss + cd_gap_task_loss + cd_conj_task_loss + 
                              cs_task_loss)
                        
            multitask_loss.backward()

            self.optimizer.step()

            epoch_train_multitask_loss += multitask_loss.item() / len(self.task_list)

            epoch_train_task_loss_dict['SC_Task'] += sc_task_loss.item()
            epoch_train_task_loss_dict['SFR_Task'] += sfr_task_loss.item()
            epoch_train_task_loss_dict['SI_Task'] += si_task_loss.item()
            epoch_train_task_loss_dict['SMU_Task'] += smu_task_loss.item()
            epoch_train_task_loss_dict['STS_Task'] += sts_task_loss.item()
            epoch_train_task_loss_dict['VIR_Task'] += vir_task_loss.item()
            epoch_train_task_loss_dict['VSR_Task'] += vsr_task_loss.item()
            epoch_train_task_loss_dict['VSRec_Task'] += vsrec_task_loss.item()
            epoch_train_task_loss_dict['CD_Color_Task'] += cd_color_task_loss.item()
            epoch_train_task_loss_dict['CD_Orientation_Task'] += cd_orientation_task_loss.item()
            epoch_train_task_loss_dict['CD_Size_Task'] += cd_size_task_loss.item()
            epoch_train_task_loss_dict['CD_Gap_Task'] += cd_gap_task_loss.item()
            epoch_train_task_loss_dict['CD_Conj_Task'] += cd_conj_task_loss.item()
            epoch_train_task_loss_dict['CS_Task'] += cs_task_loss.item()

        epoch_train_multitask_loss /= batch_index + 1
        epoch_train_task_loss_dict = {k: v / (batch_index + 1) 
                                      for k, v in epoch_train_task_loss_dict.items()}
        
        return epoch_train_multitask_loss, epoch_train_task_loss_dict


    def val_one_epoch(self):
        val_dataloader = zip(*self.val_dataloader.values())

        self.model.eval()

        epoch_val_multitask_loss = 0
        epoch_val_task_loss_dict = {task: 0 for task in self.task_list}
        epoch_val_task_acc_dict = {task: 0 for task in self.task_list}

        with torch.no_grad():
            for batch_index, multi_task_batch in tqdm(enumerate(val_dataloader)):
                # SC Task
                stim_batch, resp_batch, seq_len, _ = multi_task_batch[0]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = self.model(stim_batch, 'SC_Task', seq_len)

                out = out[torch.arange(out.size(0)), seq_len-1, :]
                resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
                sc_task_loss = self.BCE_criteria(out, resp_batch.float())

                pred = torch.round(torch.sigmoid(out))
                sc_task_acc = torch.sum(pred == resp_batch).item() / resp_batch.size(0)
                
                # SFR Task
                stim_batch, resp_batch, seq_len, _ = multi_task_batch[1]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = self.model(stim_batch, 'SFR_Task', seq_len)

                out = out[torch.arange(out.size(0)), seq_len-1, :]
                sfr_task_loss = self.BCE_criteria(out, resp_batch.float())

                pred = torch.round(torch.sigmoid(out))
                sfr_task_acc = torch.sum(pred == resp_batch).item() / (resp_batch.size(0) * resp_batch.size(1))

                # SI Task
                stim_batch, resp_batch, seq_len, _ = multi_task_batch[2]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = self.model(stim_batch, 'SI_Task', seq_len)

                out = out[torch.arange(out.size(0)), seq_len-1, :]
                resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
                si_task_loss = self.BCE_criteria(out, resp_batch.float())

                pred = torch.round(torch.sigmoid(out))
                si_task_acc = torch.sum(pred == resp_batch).item() / resp_batch.size(0)

                # SMU Task
                stim_batch, resp_batch, seq_len, set_size = multi_task_batch[3]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = self.model(stim_batch, 'SMU_Task', seq_len)

                out = torch.cat([out[i, seq_len[i]-set_size[i]:seq_len[i]] for i in range(len(seq_len))], dim=0)
                resp_batch = torch.cat([resp_batch[i, seq_len[i]-set_size[i]:seq_len[i]] for i in range(len(seq_len))], dim=0)
                smu_task_loss = self.CE_criteria(out, resp_batch)

                pred = torch.argmax(out, dim=-1)
                smu_task_acc = torch.sum(pred == resp_batch).item() / resp_batch.size(0)

                # STS Task
                stim_batch, resp_batch, seq_len = multi_task_batch[4]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = self.model(stim_batch, 'STS_Task', seq_len)

                out = torch.cat([out[i, 0:seq_len[i]] for i in range(len(seq_len))], dim=0)
                resp_batch = torch.cat([resp_batch[i, 0:seq_len[i]] for i in range(len(seq_len))], dim=0)
                sts_task_loss = self.CE_criteria(out, resp_batch)

                pred = torch.argmax(out, dim=-1)
                sts_task_acc = torch.sum(pred == resp_batch).item() / resp_batch.size(0)

                # VIR Task
                stim_batch, resp_batch, seq_len, _, _ = multi_task_batch[5]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = self.model(stim_batch, 'VIR_Task', seq_len)

                out = out[torch.arange(out.size(0)), seq_len-1, :]
                resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
                vir_task_loss = self.BCE_criteria(out, resp_batch.float())

                pred = torch.round(torch.sigmoid(out))
                vir_task_acc = torch.sum(pred == resp_batch).item() / resp_batch.size(0)

                # VSR Task
                stim_batch, resp_batch, seq_len, list_length = multi_task_batch[6]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = self.model(stim_batch, 'VSR_Task', seq_len)

                out = torch.cat([out[i, seq_len[i]-list_length[i]:seq_len[i]] for i in range(len(seq_len))], dim=0)
                resp_batch = torch.cat([resp_batch[i, seq_len[i]-list_length[i]:seq_len[i]] for i in range(len(seq_len))], dim=0)
                vsr_task_loss = self.CE_criteria(out, resp_batch)

                pred = torch.argmax(out, dim=-1)
                vsr_task_acc = torch.sum(pred == resp_batch).item() / resp_batch.size(0)

                # VSRec Task
                stim_batch, resp_batch, seq_len, list_length, _ = multi_task_batch[7]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = self.model(stim_batch, 'VSRec_Task', seq_len)

                out = torch.cat([out[i, seq_len[i]-list_length[i]:seq_len[i]] for i in range(len(seq_len))], dim=0)
                resp_batch = torch.cat([resp_batch[i, seq_len[i]-list_length[i]:seq_len[i]] for i in range(len(seq_len))] , dim=0).unsqueeze(1)
                vsrec_task_loss = self.BCE_criteria(out, resp_batch.float())

                pred = torch.round(torch.sigmoid(out))
                vsrec_task_acc = torch.sum(pred == resp_batch).item() / resp_batch.size(0)

                # CD Color Task
                stim_batch, resp_batch, seq_len, _, _ = multi_task_batch[8]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = self.model(stim_batch, 'CD_Color_Task', seq_len)

                out = out[torch.arange(out.size(0)), seq_len-1, :]
                resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
                cd_color_task_loss = self.BCE_criteria(out, resp_batch.float())

                pred = torch.round(torch.sigmoid(out))
                cd_color_task_acc = torch.sum(pred == resp_batch).item() / resp_batch.size(0)

                # CD Orientation Task
                stim_batch, resp_batch, seq_len, _, _ = multi_task_batch[9]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = self.model(stim_batch, 'CD_Orientation_Task', seq_len)

                out = out[torch.arange(out.size(0)), seq_len-1, :]
                resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
                cd_orientation_task_loss = self.BCE_criteria(out, resp_batch.float())

                pred = torch.round(torch.sigmoid(out))
                cd_orientation_task_acc = torch.sum(pred == resp_batch).item() / resp_batch.size(0)

                # CD Size Task
                stim_batch, resp_batch, seq_len, _, _ = multi_task_batch[10]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = self.model(stim_batch, 'CD_Size_Task', seq_len)

                out = out[torch.arange(out.size(0)), seq_len-1, :]
                resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
                cd_size_task_loss = self.BCE_criteria(out, resp_batch.float())

                pred = torch.round(torch.sigmoid(out))
                cd_size_task_acc = torch.sum(pred == resp_batch).item() / resp_batch.size(0)

                # CD Gap Task
                stim_batch, resp_batch, seq_len, _, _ = multi_task_batch[11]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = self.model(stim_batch, 'CD_Gap_Task', seq_len)

                out = out[torch.arange(out.size(0)), seq_len-1, :]
                resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
                cd_gap_task_loss = self.BCE_criteria(out, resp_batch.float())

                pred = torch.round(torch.sigmoid(out))
                cd_gap_task_acc = torch.sum(pred == resp_batch).item() / resp_batch.size(0)

                # CD Conj Task
                stim_batch, resp_batch, seq_len, _, _, _ = multi_task_batch[12]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = self.model(stim_batch, 'CD_Conj_Task', seq_len)

                out = out[torch.arange(out.size(0)), seq_len-1, :]
                resp_batch = resp_batch[torch.arange(resp_batch.size(0)), seq_len-1].unsqueeze(1)
                cd_conj_task_loss = self.BCE_criteria(out, resp_batch.float())

                pred = torch.round(torch.sigmoid(out))
                cd_conj_task_acc = torch.sum(pred == resp_batch).item() / resp_batch.size(0)

                # CS Task
                stim_batch, resp_batch, seq_len, _, _ = multi_task_batch[13]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = self.model(stim_batch, 'CS_Task', seq_len)

                out = torch.cat([out[i, 0:seq_len[i]] for i in range(len(seq_len))], dim=0)
                resp_batch = torch.cat([resp_batch[i, 0:seq_len[i]] for i in range(len(seq_len))], dim=0)
                cs_task_loss = self.CE_criteria(out, resp_batch)

                pred = torch.argmax(out, dim=-1)
                cs_task_acc = torch.sum(pred == resp_batch).item() / resp_batch.size(0)


                # Multitask Loss
                multitask_loss = (sc_task_loss + sfr_task_loss + si_task_loss + smu_task_loss + 
                                sts_task_loss + vir_task_loss + 
                                vsr_task_loss + vsrec_task_loss + 
                                cd_color_task_loss + cd_orientation_task_loss + 
                                cd_size_task_loss + cd_gap_task_loss + cd_conj_task_loss + 
                                cs_task_loss)
                
                epoch_val_multitask_loss += multitask_loss.item() / len(self.task_list)

                epoch_val_task_loss_dict['SC_Task'] += sc_task_loss.item()
                epoch_val_task_loss_dict['SFR_Task'] += sfr_task_loss.item()
                epoch_val_task_loss_dict['SI_Task'] += si_task_loss.item()
                epoch_val_task_loss_dict['SMU_Task'] += smu_task_loss.item()
                epoch_val_task_loss_dict['STS_Task'] += sts_task_loss.item()
                epoch_val_task_loss_dict['VIR_Task'] += vir_task_loss.item()
                epoch_val_task_loss_dict['VSR_Task'] += vsr_task_loss.item()
                epoch_val_task_loss_dict['VSRec_Task'] += vsrec_task_loss.item()
                epoch_val_task_loss_dict['CD_Color_Task'] += cd_color_task_loss.item()
                epoch_val_task_loss_dict['CD_Orientation_Task'] += cd_orientation_task_loss.item()
                epoch_val_task_loss_dict['CD_Size_Task'] += cd_size_task_loss.item()
                epoch_val_task_loss_dict['CD_Gap_Task'] += cd_gap_task_loss.item()
                epoch_val_task_loss_dict['CD_Conj_Task'] += cd_conj_task_loss.item()
                epoch_val_task_loss_dict['CS_Task'] += cs_task_loss.item()

                # Accuracies
                epoch_val_task_acc_dict['SC_Task'] += sc_task_acc
                epoch_val_task_acc_dict['SFR_Task'] += sfr_task_acc
                epoch_val_task_acc_dict['SI_Task'] += si_task_acc
                epoch_val_task_acc_dict['SMU_Task'] += smu_task_acc
                epoch_val_task_acc_dict['STS_Task'] += sts_task_acc
                epoch_val_task_acc_dict['VIR_Task'] += vir_task_acc
                epoch_val_task_acc_dict['VSR_Task'] += vsr_task_acc
                epoch_val_task_acc_dict['VSRec_Task'] += vsrec_task_acc
                epoch_val_task_acc_dict['CD_Color_Task'] += cd_color_task_acc
                epoch_val_task_acc_dict['CD_Orientation_Task'] += cd_orientation_task_acc
                epoch_val_task_acc_dict['CD_Size_Task'] += cd_size_task_acc
                epoch_val_task_acc_dict['CD_Gap_Task'] += cd_gap_task_acc
                epoch_val_task_acc_dict['CD_Conj_Task'] += cd_conj_task_acc
                epoch_val_task_acc_dict['CS_Task'] += cs_task_acc

        epoch_val_multitask_loss /= batch_index + 1
        epoch_val_task_loss_dict = {k: v / (batch_index + 1) 
                                    for k, v in epoch_val_task_loss_dict.items()}
        epoch_val_task_acc_dict = {k: v / (batch_index + 1)
                                    for k, v in epoch_val_task_acc_dict.items()}
            
        return epoch_val_multitask_loss, epoch_val_task_loss_dict, epoch_val_task_acc_dict
    

    def test_one_epoch(self, test_model):
        test_dataloader = zip(*self.test_dataloader.values())

        test_model.eval()

        epoch_acc = {}
                    
        with torch.no_grad():
            for batch_index, multi_task_batch in tqdm(enumerate(test_dataloader)):
                # SC Task
                task = 'SC_Task'
                stim_batch, resp_batch, seq_len, symmetry_offset = multi_task_batch[0]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = test_model(stim_batch, 'SC_Task', seq_len)

                pred = torch.round(torch.sigmoid(out))

                for index, len in enumerate(seq_len):
                    if task not in epoch_acc:
                        epoch_acc[task] = [0, 0]
                    if task+'_Set_Size_'+str((len-1).item()) not in epoch_acc:
                        epoch_acc[task+'_Set_Size_'+str((len-1).item())] = [0, 0]
                    if task+'_Symmetry_Offset_'+str(symmetry_offset[index].item()) not in epoch_acc:
                        epoch_acc[task+'_Symmetry_Offset_'+str(symmetry_offset[index].item())] = [0, 0]

                    epoch_acc[task][1] += 1
                    epoch_acc[task+'_Set_Size_'+str((len-1).item())][1] += 1
                    epoch_acc[task+'_Symmetry_Offset_'+str(symmetry_offset[index].item())][1] += 1
                    
                    if (pred[index, len-1] == resp_batch[index, len-1]):
                        epoch_acc[task][0] += 1
                        epoch_acc[task+'_Set_Size_'+str((len-1).item())][0] += 1
                        epoch_acc[task+'_Symmetry_Offset_'+str(symmetry_offset[index].item())][0] += 1

                # SFR Task
                task = 'SFR_Task'
                stim_batch, resp_batch, seq_len, recall_gt = multi_task_batch[1]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = test_model(stim_batch, 'SFR_Task', seq_len)

                pred = torch.round(torch.sigmoid(out))

                for index, len in enumerate(seq_len):
                    if task+'_Order' not in epoch_acc:
                        epoch_acc[task+'_Order'] = [0, 0]
                    if task+'_No_Order' not in epoch_acc:
                        epoch_acc[task+'_No_Order'] = [0, 0]

                    curr_out = out[index, len-1]
                    _, idxs = curr_out.topk(len-1, dim=-1, sorted=True)
                    curr_recall_gt = recall_gt[index, :(len-1)]

                    idxs = idxs.cpu()

                    if (idxs == curr_recall_gt).all():
                        epoch_acc[task+'_Order'][0] += 1
                    epoch_acc[task+'_Order'][1] += 1

                    if torch.all(torch.eq(torch.sort(idxs).values, torch.sort(curr_recall_gt).values)):
                        epoch_acc[task+'_No_Order'][0] += 1
                    epoch_acc[task+'_No_Order'][1] += 1

                    for count, idx in enumerate(curr_recall_gt):
                        if task+'_List_Length_'+str((len-1).item())+'_Serial_Position_'+str(count+1) not in epoch_acc:
                            epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Serial_Position_'+str(count+1)] = [0, 0]
                        if task+'_List_Length_'+str((len-1).item())+'_First' not in epoch_acc:
                            epoch_acc[task+'_List_Length_'+str((len-1).item())+'_First'] = [0, 0]
                        if task+'_List_Length_'+str((len-1).item())+'_Last_4' not in epoch_acc:
                            epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Last_4'] = [0, 0]
                        if task+'_List_Length_'+str((len-1).item())+'_Other' not in epoch_acc:
                            epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Other'] = [0, 0]
                        if task+'_List_Length_'+str((len-1).item())+'_Error' not in epoch_acc:
                            epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Error'] = [0, 0]

                        if idx in idxs:
                            epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Serial_Position_'+str(count+1)][0] += 1
                        epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Serial_Position_'+str(count+1)][1] += 1


                    if idxs[0] == curr_recall_gt[0]:
                        epoch_acc[task+'_List_Length_'+str((len-1).item())+'_First'][0] += 1
                    epoch_acc[task+'_List_Length_'+str((len-1).item())+'_First'][1] += 1

                    if len-1 > 4:
                        if idxs[0] in curr_recall_gt[-4:]:
                            epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Last_4'][0] += 1
                        epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Last_4'][1] += 1
                    elif len-1 == 4:
                        if idxs[0] in curr_recall_gt[-3:]:
                            epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Last_4'][0] += 1
                        epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Last_4'][1] += 1
                    elif len-1 == 3:
                        if idxs[0] in curr_recall_gt[-2:]:
                            epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Last_4'][0] += 1
                        epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Last_4'][1] += 1
                    elif len-1 == 2:
                        if idxs[0] in curr_recall_gt[-1:]:
                            epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Last_4'][0] += 1
                        epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Last_4'][1] += 1

                    if idxs[0] not in curr_recall_gt:
                        epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Error'][0] += 1
                    epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Error'][1] += 1

                    if idxs[0] in curr_recall_gt[1:-4]:
                        epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Other'][0] += 1
                    epoch_acc[task+'_List_Length_'+str((len-1).item())+'_Other'][1] += 1


                # SI Task
                task = 'SI_Task'
                stim_batch, resp_batch, seq_len, part_size = multi_task_batch[2]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = test_model(stim_batch, 'SI_Task', seq_len)

                pred = torch.round(torch.sigmoid(out))

                for index, len in enumerate(seq_len):
                    if task not in epoch_acc:
                        epoch_acc[task] = [0, 0]
                    if task+'_Part_Size_'+str(part_size[index].item()) not in epoch_acc:
                        epoch_acc[task+'_Part_Size_'+str(part_size[index].item())] = [0, 0]

                    if (pred[index, len-1] == resp_batch[index, len-1]):
                        epoch_acc[task][0] += 1
                        epoch_acc[task+'_Part_Size_'+str(part_size[index].item())][0] += 1
                    epoch_acc[task][1] += 1
                    epoch_acc[task+'_Part_Size_'+str(part_size[index].item())][1] += 1

                # SMU Task
                task = 'SMU_Task'
                stim_batch, resp_batch, seq_len, set_size = multi_task_batch[3]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = test_model(stim_batch, 'SMU_Task', seq_len)

                pred = torch.argmax(out, dim=-1)

                for index, len in enumerate(seq_len):
                    if task not in epoch_acc:
                        epoch_acc[task] = [0, 0]

                    for count, val in enumerate(pred[index][len-set_size[index]:len]):
                        if task+'_Set_Size_'+str(set_size[index].item())+'_Serial_Position_'+str(count+1) not in epoch_acc:
                            epoch_acc[task+'_Set_Size_'+str(set_size[index].item())+'_Serial_Position_'+str(count+1)] = [0, 0]

                        if (val == resp_batch[index][len-set_size[index]+count]):
                            epoch_acc[task][0] += 1
                            epoch_acc[task+'_Set_Size_'+str(set_size[index].item())+'_Serial_Position_'+str(count+1)][0] += 1
                        epoch_acc[task][1] += 1
                        epoch_acc[task+'_Set_Size_'+str(set_size[index].item())+'_Serial_Position_'+str(count+1)][1] += 1


                # STS Task
                task = 'STS_Task'
                stim_batch, resp_batch, seq_len = multi_task_batch[4]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = self.model(stim_batch, 'STS_Task', seq_len)

                pred = torch.argmax(out, dim=-1)
                
                for index, len in enumerate(seq_len):
                    if task not in epoch_acc:
                        epoch_acc[task] = [0, 0]

                    for count, item in enumerate(resp_batch[index]):
                        if item != 2:
                            if (item == pred[index, count]):
                                epoch_acc[task][0] += 1
                            epoch_acc[task][1] += 1

                # VIR Task
                task = 'VIR_Task'
                stim_batch, resp_batch, seq_len, ri, gt_index = multi_task_batch[5]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = test_model(stim_batch, 'VIR_Task', seq_len)

                pred = torch.round(torch.sigmoid(out))

                for index, len in enumerate(seq_len):
                    if task not in epoch_acc:
                        epoch_acc[task] = [0, 0]
                    if task+'_RI_'+str(ri[index].item())+'_Serial_Position_'+str(gt_index[index].item()) not in epoch_acc:
                        epoch_acc[task+'_RI_'+str(ri[index].item())+'_Serial_Position_'+str(gt_index[index].item())] = [0, 0]

                    if (pred[index, len-1] == resp_batch[index, len-1]):
                        epoch_acc[task][0] += 1
                        epoch_acc[task+'_RI_'+str(ri[index].item())+
                                    '_Serial_Position_'+
                                    str(gt_index[index].item())][0] += 1
                    epoch_acc[task][1] += 1
                    epoch_acc[task+'_RI_'+str(ri[index].item())+
                                '_Serial_Position_'+
                                str(gt_index[index].item())][1] += 1

                # VSR Task
                task = 'VSR_Task'
                stim_batch, resp_batch, seq_len, list_length = multi_task_batch[6]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = test_model(stim_batch, 'VSR_Task', seq_len)

                pred = torch.argmax(out, dim=-1)

                for index, len in enumerate(seq_len):
                    if task not in epoch_acc:
                        epoch_acc[task] = [0, 0]
                    if task+'_All' not in epoch_acc:
                        epoch_acc[task+'_All'] = [0, 0]
                    if task+'_Each' not in epoch_acc:
                        epoch_acc[task+'_Each'] = [0, 0]
                        

                    curr_gt = resp_batch[index, list_length[index]:list_length[index]*2]
                    curr_pred = pred[index, list_length[index]:list_length[index]*2]

                    if (curr_pred == curr_gt).all():
                        epoch_acc[task+'_All'][0] += 1
                    epoch_acc[task+'_All'][1] += 1

                    for count, item in enumerate(curr_gt):
                        if task+'_List_Length_'+str(list_length[index].item())+'_Serial_Position_'+str(count+1) not in epoch_acc:
                            epoch_acc[task+'_List_Length_'+str(list_length[index].item())+'_Serial_Position_'+str(count+1)] = [0, 0]

                        if (item == curr_pred[count]):
                            epoch_acc[task+'_Each'][0] += 1
                            epoch_acc[task+'_List_Length_'+str(list_length[index].item())+
                                        '_Serial_Position_'+str(count+1)][0] += 1
                        epoch_acc[task+'_Each'][1] += 1
                        epoch_acc[task+'_List_Length_'+str(list_length[index].item())+
                                    '_Serial_Position_'+str(count+1)][1] += 1

                # VSRec Task
                task = 'VSRec_Task'
                stim_batch, resp_batch, seq_len, list_length, distractor_diff = multi_task_batch[7]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = test_model(stim_batch, 'VSRec_Task', seq_len)

                pred = torch.round(torch.sigmoid(out))

                for index, len in enumerate(seq_len):
                    if task not in epoch_acc:
                        epoch_acc[task] = [0, 0]
                    if task+'_Each' not in epoch_acc:
                        epoch_acc[task+'_Each'] = [0, 0]
                    if task+'_All' not in epoch_acc:
                        epoch_acc[task+'_All'] = [0, 0]
                    if task+'_All_Distractor_'+str(distractor_diff[index].item()) not in epoch_acc:
                        epoch_acc[task+'_All_Distractor_'+str(distractor_diff[index].item())] = [0, 0]
                    if task+'_Each_Distractor_'+str(distractor_diff[index].item()) not in epoch_acc:
                        epoch_acc[task+'_Each_Distractor_'+str(distractor_diff[index].item())] = [0, 0]

                    curr_gt = resp_batch[index, list_length[index]:list_length[index]*2]
                    curr_pred = pred[index, list_length[index]:list_length[index]*2]

                    if (curr_pred == curr_gt).all():
                        epoch_acc[task+'_All'][0] += 1
                        epoch_acc[task+'_All_Distractor_'+str(distractor_diff[index].item())][0] += 1
                    epoch_acc[task+'_All'][1] += 1
                    epoch_acc[task+'_All_Distractor_'+str(distractor_diff[index].item())][1] += 1

                    for count, item in enumerate(curr_gt):
                        if task+'_List_Length_'+str(list_length[index].item())+'_Serial_Position_'+str(count+1) not in epoch_acc:
                            epoch_acc[task+'_List_Length_'+str(list_length[index].item())+'_Serial_Position_'+str(count+1)] = [0, 0]
                        if task+'_List_Length_'+str(list_length[index].item())+'_Serial_Position_'+str(count+1)+'_Distractor_'+str(distractor_diff[index].item()) not in epoch_acc:
                            epoch_acc[task+'_List_Length_'+str(list_length[index].item())+'_Serial_Position_'+str(count+1)+'_Distractor_'+str(distractor_diff[index].item())] = [0, 0]

                        if (item == curr_pred[count]):
                            epoch_acc[task+'_Each'][0] += 1
                            epoch_acc[task+'_Each_Distractor_'+str(distractor_diff[index].item())][0] += 1
                            epoch_acc[task+'_List_Length_'+str(list_length[index].item())+
                                        '_Serial_Position_'+str(count+1)][0] += 1
                            epoch_acc[task+'_List_Length_'+str(list_length[index].item())+
                                        '_Serial_Position_'+str(count+1)+
                                        '_Distractor_'+str(distractor_diff[index].item())][0] += 1
                        epoch_acc[task+'_Each'][1] += 1
                        epoch_acc[task+'_Each_Distractor_'+str(distractor_diff[index].item())][1] += 1
                        epoch_acc[task+'_List_Length_'+str(list_length[index].item())+
                                    '_Serial_Position_'+str(count+1)][1] += 1
                        epoch_acc[task+'_List_Length_'+str(list_length[index].item())+
                                    '_Serial_Position_'+str(count+1)+
                                    '_Distractor_'+str(distractor_diff[index].item())][1] += 1
                        
                # CD Color Task
                task = 'CD_Color_Task'
                stim_batch, resp_batch, seq_len, ri, set_size = multi_task_batch[8]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = test_model(stim_batch, 'CD_Color_Task', seq_len)

                pred = torch.round(torch.sigmoid(out))

                for index, len in enumerate(seq_len):
                    if task not in epoch_acc:
                        epoch_acc[task] = [0, 0]
                    if task+'_RI_'+str(ri[index].item()) not in epoch_acc:
                        epoch_acc[task+'_RI_'+str(ri[index].item())] = [0, 0]
                    if task+'_Set_Size_'+str(set_size[index].item()) not in epoch_acc:
                        epoch_acc[task+'_Set_Size_'+str(set_size[index].item())] = [0, 0]
                    if task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item()) not in epoch_acc:
                        epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())] = [0, 0]

                    if (pred[index, len-1] == resp_batch[index, len-1]):
                        epoch_acc[task][0] += 1
                        epoch_acc[task+'_RI_'+str(ri[index].item())][0] += 1
                        epoch_acc[task+'_Set_Size_'+str(set_size[index].item())][0] += 1
                        epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())][0] += 1
                    epoch_acc[task][1] += 1
                    epoch_acc[task+'_RI_'+str(ri[index].item())][1] += 1
                    epoch_acc[task+'_Set_Size_'+str(set_size[index].item())][1] += 1
                    epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())][1] += 1


                # CD Orientation Task
                task = 'CD_Orientation_Task'
                stim_batch, resp_batch, seq_len, ri, set_size = multi_task_batch[9]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = test_model(stim_batch, 'CD_Orientation_Task', seq_len)

                pred = torch.round(torch.sigmoid(out))

                for index, len in enumerate(seq_len):
                    if task not in epoch_acc:
                        epoch_acc[task] = [0, 0]
                    if task+'_RI_'+str(ri[index].item()) not in epoch_acc:
                        epoch_acc[task+'_RI_'+str(ri[index].item())] = [0, 0]
                    if task+'_Set_Size_'+str(set_size[index].item()) not in epoch_acc:
                        epoch_acc[task+'_Set_Size_'+str(set_size[index].item())] = [0, 0]
                    if task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item()) not in epoch_acc:
                        epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())] = [0, 0]

                    if (pred[index, len-1] == resp_batch[index, len-1]):
                        epoch_acc[task][0] += 1
                        epoch_acc[task+'_RI_'+str(ri[index].item())][0] += 1
                        epoch_acc[task+'_Set_Size_'+str(set_size[index].item())][0] += 1
                        epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())][0] += 1
                    epoch_acc[task][1] += 1
                    epoch_acc[task+'_RI_'+str(ri[index].item())][1] += 1
                    epoch_acc[task+'_Set_Size_'+str(set_size[index].item())][1] += 1
                    epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())][1] += 1


                # CD Size Task
                task = 'CD_Size_Task'
                stim_batch, resp_batch, seq_len, ri, set_size = multi_task_batch[10]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = test_model(stim_batch, 'CD_Size_Task', seq_len)

                pred = torch.round(torch.sigmoid(out))

                for index, len in enumerate(seq_len):
                    if task not in epoch_acc:
                        epoch_acc[task] = [0, 0]
                    if task+'_RI_'+str(ri[index].item()) not in epoch_acc:
                        epoch_acc[task+'_RI_'+str(ri[index].item())] = [0, 0]
                    if task+'_Set_Size_'+str(set_size[index].item()) not in epoch_acc:
                        epoch_acc[task+'_Set_Size_'+str(set_size[index].item())] = [0, 0]
                    if task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item()) not in epoch_acc:
                        epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())] = [0, 0]

                    if (pred[index, len-1] == resp_batch[index, len-1]):
                        epoch_acc[task][0] += 1
                        epoch_acc[task+'_RI_'+str(ri[index].item())][0] += 1
                        epoch_acc[task+'_Set_Size_'+str(set_size[index].item())][0] += 1
                        epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())][0] += 1
                    epoch_acc[task][1] += 1
                    epoch_acc[task+'_RI_'+str(ri[index].item())][1] += 1
                    epoch_acc[task+'_Set_Size_'+str(set_size[index].item())][1] += 1
                    epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())][1] += 1

                
                # CD Gap Task
                task = 'CD_Gap_Task'
                stim_batch, resp_batch, seq_len, ri, set_size = multi_task_batch[11]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = test_model(stim_batch, 'CD_Gap_Task', seq_len)

                pred = torch.round(torch.sigmoid(out))

                for index, len in enumerate(seq_len):
                    if task not in epoch_acc:
                        epoch_acc[task] = [0, 0]
                    if task+'_RI_'+str(ri[index].item()) not in epoch_acc:
                        epoch_acc[task+'_RI_'+str(ri[index].item())] = [0, 0]
                    if task+'_Set_Size_'+str(set_size[index].item()) not in epoch_acc:
                        epoch_acc[task+'_Set_Size_'+str(set_size[index].item())] = [0, 0]
                    if task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item()) not in epoch_acc:
                        epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())] = [0, 0]

                    if (pred[index, len-1] == resp_batch[index, len-1]):
                        epoch_acc[task][0] += 1
                        epoch_acc[task+'_RI_'+str(ri[index].item())][0] += 1
                        epoch_acc[task+'_Set_Size_'+str(set_size[index].item())][0] += 1
                        epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())][0] += 1
                    epoch_acc[task][1] += 1
                    epoch_acc[task+'_RI_'+str(ri[index].item())][1] += 1
                    epoch_acc[task+'_Set_Size_'+str(set_size[index].item())][1] += 1
                    epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())][1] += 1

                
                # CD Conj Task
                task = 'CD_Conj_Task'
                stim_batch, resp_batch, seq_len, ri, set_size, conj_gt = multi_task_batch[12]

                stim_batch = stim_batch.to(self.device)
                resp_batch = resp_batch.to(self.device)

                out, _, _, _, _ = test_model(stim_batch, 'CD_Conj_Task', seq_len)

                pred = torch.round(torch.sigmoid(out))

                for index, len in enumerate(seq_len):
                    if task not in epoch_acc:
                        epoch_acc[task] = [0, 0]
                    if task+'_RI_'+str(ri[index].item()) not in epoch_acc:
                        epoch_acc[task+'_RI_'+str(ri[index].item())] = [0, 0]
                    if task+'_Set_Size_'+str(set_size[index].item()) not in epoch_acc:
                        epoch_acc[task+'_Set_Size_'+str(set_size[index].item())] = [0, 0]
                    if task+'_Conj_GT_'+str(conj_gt[index].item()) not in epoch_acc:
                        epoch_acc[task+'_Conj_GT_'+str(conj_gt[index].item())] = [0, 0]
                    if task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item()) not in epoch_acc:
                        epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())] = [0, 0]
                    if task+'_RI_'+str(ri[index].item())+'_Conj_GT_'+str(conj_gt[index].item()) not in epoch_acc:
                        epoch_acc[task+'_RI_'+str(ri[index].item())+'_Conj_GT_'+str(conj_gt[index].item())] = [0, 0]
                    if task+'_Set_Size_'+str(set_size[index].item())+'_Conj_GT_'+str(conj_gt[index].item()) not in epoch_acc:
                        epoch_acc[task+'_Set_Size_'+str(set_size[index].item())+'_Conj_GT_'+str(conj_gt[index].item())] = [0, 0]
                    if task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())+'_Conj_GT_'+str(conj_gt[index].item()) not in epoch_acc:
                        epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())+'_Conj_GT_'+str(conj_gt[index].item())] = [0, 0]

                    if (pred[index, len-1] == resp_batch[index, len-1]):
                        epoch_acc[task][0] += 1
                        epoch_acc[task+'_RI_'+str(ri[index].item())][0] += 1
                        epoch_acc[task+'_Set_Size_'+str(set_size[index].item())][0] += 1
                        epoch_acc[task+'_Conj_GT_'+str(conj_gt[index].item())][0] += 1
                        epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())][0] += 1
                        epoch_acc[task+'_RI_'+str(ri[index].item())+'_Conj_GT_'+str(conj_gt[index].item())][0] += 1
                        epoch_acc[task+'_Set_Size_'+str(set_size[index].item())+'_Conj_GT_'+str(conj_gt[index].item())][0] += 1
                        epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())+'_Conj_GT_'+str(conj_gt[index].item())][0] += 1
                    epoch_acc[task][1] += 1
                    epoch_acc[task+'_RI_'+str(ri[index].item())][1] += 1
                    epoch_acc[task+'_Set_Size_'+str(set_size[index].item())][1] += 1
                    epoch_acc[task+'_Conj_GT_'+str(conj_gt[index].item())][1] += 1
                    epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())][1] += 1
                    epoch_acc[task+'_RI_'+str(ri[index].item())+'_Conj_GT_'+str(conj_gt[index].item())][1] += 1
                    epoch_acc[task+'_Set_Size_'+str(set_size[index].item())+'_Conj_GT_'+str(conj_gt[index].item())][1] += 1
                    epoch_acc[task+'_RI_'+str(ri[index].item())+'_Set_Size_'+str(set_size[index].item())+'_Conj_GT_'+str(conj_gt[index].item())][1] += 1

        return epoch_acc
    

    def viz_test(self, epoch, epoch_acc):

        save_path = os.path.join(self.config.output_path, 'Epoch_'+str(epoch))

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # SI Task
        si_part_size_acc = [[], []]

        for key in list(epoch_acc.keys()):
            if 'SI_Task' in key and 'Part_Size' in key:
                part_size = int(key.split('_')[4])
                si_part_size_acc[0].append(part_size)
                si_part_size_acc[1].append(epoch_acc[key][0]/epoch_acc[key][1])

        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_subplot(111)

        ax.plot(si_part_size_acc[0], si_part_size_acc[1], marker='o')

        ax.set_xlabel('Part Size')
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        ax.set_ylabel('Accuracy')

        ax.set_title('Spatial Integration Part Size Accuracy')

        self.logger.info('SI Part Size Accuracy: {}'.format(si_part_size_acc))

        plt.savefig(os.path.join(save_path, 
                                 'SI_Part_Size_Accuracy.png'))


        si_num_integration_acc = [[], []]

        part_sizes = si_part_size_acc[0]
        accs = si_part_size_acc[1]

        part_sizes.reverse()
        accs.reverse()

        for idx, part_size in enumerate(part_sizes):
            num_integ = (12 // part_size) - 1
            si_num_integration_acc[0].append(num_integ)
            si_num_integration_acc[1].append(accs[idx])
        
        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_subplot(111)

        ax.plot(si_num_integration_acc[0], si_num_integration_acc[1], marker='o')

        ax.set_xlabel('Number of Integrations')
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        ax.set_ylabel('Accuracy')

        ax.set_title('Spatial Integration Number of Integrations Accuracy')

        self.logger.info('SI Number of Integrations Accuracy: {}'.format(si_num_integration_acc))
        plt.savefig(os.path.join(save_path, 
                                 'SI_Num_Integration_Accuracy.png'))


        # SC Task
        sc_task_set_size_length_acc = [[], []]

        for key in list(epoch_acc.keys()):
            if 'SC_Task' in key and 'Set_Size' in key:
                set_size = int(key.split('_')[4])
                sc_task_set_size_length_acc[0].append(set_size)
                sc_task_set_size_length_acc[1].append(epoch_acc[key][0]/epoch_acc[key][1])

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        ax.plot(sc_task_set_size_length_acc[0], sc_task_set_size_length_acc[1], marker='o')

        ax.set_xlabel('Set Size')
        ax.set_xticks(sc_task_set_size_length_acc[0])
        ax.set_ylabel('Accuracy')

        ax.set_title('Spatial Coordination Set Size Length Accuracy')

        self.logger.info('SC Task Set Size Length Accuracy: {}'.format(sc_task_set_size_length_acc))
        plt.savefig(os.path.join(save_path, 
                                 'Spatial_Coordination_Set_Size_Length_Accuracy.png'))


        sc_task_symmetry_offset_acc = [[], []]

        for key in list(epoch_acc.keys()):
            if 'SC_Task' in key and 'Offset' in key and '0' not in key:
                symmetry_offset = int(key.split('_')[4])
                sc_task_symmetry_offset_acc[0].append(symmetry_offset)
                sc_task_symmetry_offset_acc[1].append(epoch_acc[key][0]/epoch_acc[key][1])

        data_y = [val for (_, val) in sorted(zip(sc_task_symmetry_offset_acc[0], 
                                                 sc_task_symmetry_offset_acc[1]), key=lambda x: x[0])]
        data_x = sorted(sc_task_symmetry_offset_acc[0])

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        ax.plot(data_x, data_y, marker='o')

        ax.set_xlabel('Symmetry Offset')
        ax.set_xticks(data_x)
        ax.set_ylabel('Accuracy')

        ax.set_title('Spatial Coordination Symmetry Offset Accuracy')

        self.logger.info('SC Task Symmetry Offset Accuracy: {}'.format(sc_task_symmetry_offset_acc))
        plt.savefig(os.path.join(save_path, 
                                 'Spatial_Coordination_Symmetry_Offset_Accuracy.png'))


        # SMU Task
        smu_list_set_size_serial_acc = {}

        for key in list(epoch_acc.keys()):
            if 'SMU_Task' in key and 'Serial_Position' in key:
                set_size = int(key.split('_')[4])
                if set_size not in list(smu_list_set_size_serial_acc.keys()):
                    smu_list_set_size_serial_acc[set_size] = [[], [i for i in range(1, set_size+1)]]

        for key in list(smu_list_set_size_serial_acc.keys()):
            for pos in smu_list_set_size_serial_acc[key][1]:
                corr, tot = epoch_acc['SMU_Task_Set_Size_{}_Serial_Position_{}'.format(key, pos)]
                smu_list_set_size_serial_acc[key][0].append(corr/tot)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        for key in list(smu_list_set_size_serial_acc.keys()):
            ax.plot(smu_list_set_size_serial_acc[key][1], smu_list_set_size_serial_acc[key][0], 
                    label='Set Size: {}'.format(key), marker='o')

        ax.set_xlabel("Probe Position")
        ax.set_xticks(list(smu_list_set_size_serial_acc.values())[-1][1])
        ax.set_ylabel("Accuracy")
        ax.set_title("Spatial Memory Updating Set Size Probe Position Accuracy")
        ax.legend()

        self.logger.info('Spatial Memory Updating Set Size Probe Position Accuracy: {}'.format(smu_list_set_size_serial_acc))
        plt.savefig(os.path.join(save_path, 
                                 'SMU_Set_Size_Probe_Pos.png'))


        smu_list_set_size_acc = [[], []]

        for set_size in list(smu_list_set_size_serial_acc.keys()):
            accs = smu_list_set_size_serial_acc[set_size][0]
            smu_list_set_size_acc[0].append(set_size)
            smu_list_set_size_acc[1].append(sum(accs)/len(accs))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        ax.plot(smu_list_set_size_acc[0], smu_list_set_size_acc[1], marker='o')

        ax.set_xlabel('Set Size')
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
        ax.set_ylabel('Accuracy')

        ax.set_title('Spatial Memory Updating Set Size Accuracy')

        self.logger.info('SMU Task Set Size Accuracy: {}'.format(smu_list_set_size_acc))
        plt.savefig(os.path.join(save_path, 
                                 'SMU_Set_Size_Accuracy.png'))

        # VSRec Task
        vsrec_list_serial_acc = {}

        for key in list(epoch_acc.keys()):
            if 'VSRec_Task_List' in key and 'Distractor' not in key:
                list_length = int(key.split('_')[4])
                if list_length not in list(vsrec_list_serial_acc.keys()):
                    vsrec_list_serial_acc[list_length] = [[], [i for i in range(1, list_length+1)]]

        for key in list(vsrec_list_serial_acc.keys()):
            for pos in vsrec_list_serial_acc[key][1]:
                corr, tot = epoch_acc['VSRec_Task_List_Length_{}_Serial_Position_{}'.format(key, pos)]
                vsrec_list_serial_acc[key][0].append(corr/tot)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        for key in list(vsrec_list_serial_acc.keys()):
            ax.plot(vsrec_list_serial_acc[key][1], vsrec_list_serial_acc[key][0], 
                    label='List Length: {}'.format(key), marker='o')

        ax.set_xlabel("Serial Position")
        ax.set_ylabel("Accuracy")
        ax.set_title("Visual Serial Recognition Accuracy")
        ax.legend()

        self.logger.info('VSRec Task List Length Serial Position Accuracy: {}'.format(vsrec_list_serial_acc))
        plt.savefig(os.path.join(save_path, 
                                 'VSRec_Task_Accuracy.png'))


        vsrec_list_serial_distractor_acc = {}
        vsrec_list_lengths = []
        vsrec_distractors = []

        for key in list(epoch_acc.keys()):
            if 'VSRec_Task_List' in key and 'Distractor' in key:
                list_length = str(int(key.split('_')[4]))
                distractor = str(int(key.split('_')[9]))
                if list_length not in vsrec_list_lengths:
                    vsrec_list_lengths.append(list_length)
                if distractor not in vsrec_distractors:
                    vsrec_distractors.append(distractor)

        vsrec_list_lengths.sort(key=int)
        vsrec_distractors.sort(key=int)

        for list_length in vsrec_list_lengths:
            for distractor in vsrec_distractors:
                vsrec_list_serial_distractor_acc['{}_{}'.format(list_length, distractor)] = [[], []]
                for pos in range(1, int(list_length)+1):
                    corr, tot = epoch_acc['VSRec_Task_List_Length_{}_Serial_Position_{}_Distractor_{}'.format(
                        list_length, pos, distractor)]
                    vsrec_list_serial_distractor_acc['{}_{}'.format(list_length, distractor)][0].append(corr/tot)
                    vsrec_list_serial_distractor_acc['{}_{}'.format(list_length, distractor)][1].append(pos)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        for key in list(vsrec_list_serial_distractor_acc.keys()):
            ax.plot(vsrec_list_serial_distractor_acc[key][1], vsrec_list_serial_distractor_acc[key][0], 
                    label='List Length: {}, Distractor: {}'.format(key.split('_')[0], key.split('_')[1]), 
                    marker='o')

        ax.set_xlabel("Serial Position")
        ax.set_ylabel("Accuracy")
        ax.set_title("Visual Serial Recognition Distractor Accuracy")
        ax.legend()

        self.logger.info('VSRec Task List Length Serial Position Distractor Accuracy: {}'.format(vsrec_list_serial_distractor_acc))
        plt.savefig(os.path.join(save_path, 
                                 'VSRec_Task_Distractor_Serial_Accuracy.png'))


        vsrec_distractor_acc = [[], []]
        distractor_count = {}

        for key in list(epoch_acc.keys()):
            if 'VSRec_Task_List' in key and 'Distractor' in key:
                distractor = int(key.split('_')[9])
                if distractor not in list(distractor_count.keys()):
                    distractor_count[distractor] = [0, 0]

                if distractor in list(distractor_count.keys()):
                    corr, tot = epoch_acc[key]
                    distractor_count[distractor][0] += corr
                    distractor_count[distractor][1] += tot

        for distractor in sorted(list(distractor_count.keys())):
            vsrec_distractor_acc[0].append(distractor_count[distractor][0]/distractor_count[distractor][1])
            vsrec_distractor_acc[1].append(distractor)

        data_y = [val for (_, val) in sorted(zip(vsrec_distractor_acc[1], 
                                                 vsrec_distractor_acc[0]), key=lambda x: x[0])]
        data_x = sorted(vsrec_distractor_acc[1])

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        ax.plot(data_x, data_y, marker='o')

        ax.set_xlabel("Distractor Difference")
        ax.set_ylabel("Accuracy")
        ax.set_title("Visual Serial Recognition Distractor Accuracy")

        self.logger.info('VSRec Task Distractor Accuracy: {}'.format(vsrec_distractor_acc))
        plt.savefig(os.path.join(save_path, 
                                 'VSRec_Task_Distractor_Accuracy_Overall.png'))


        # VIR Task
        vir_ri_serial_acc = {}
        vir_ri = []
        vir_pos = []

        for key in list(epoch_acc.keys()):
            if 'VIR' in key and 'Serial' in key:
                ri = int(key.split('_')[4])
                pos = int(key.split('_')[7])
                if pos not in vir_pos:
                    vir_pos.append(pos)
                if ri not in vir_ri:
                    vir_ri.append(ri)

        vir_ri.sort()
        vir_pos.sort()

        for ri in vir_ri:
            vir_ri_serial_acc[ri] = [[], []]
            for pos in vir_pos:
                corr, tot = epoch_acc['VIR_Task_RI_{}_Serial_Position_{}'.format(ri, pos)]
                vir_ri_serial_acc[ri][0].append(corr/tot)
                vir_ri_serial_acc[ri][1].append(pos+1)        

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        for ri in list(vir_ri_serial_acc.keys()):
            ax.plot(vir_ri_serial_acc[ri][1], vir_ri_serial_acc[ri][0], 
                    label='RI: {}'.format(ri), marker='o')

        ax.set_xlabel("Probe Position")
        ax.set_xticks(list(vir_ri_serial_acc.values())[-1][1])
        ax.set_ylabel("Accuracy")

        ax.set_title("VIR RI Serial Position Accuracy")
        ax.legend()

        self.logger.info('VIR Task RI Serial Position Accuracy: {}'.format(vir_ri_serial_acc))
        plt.savefig(os.path.join(save_path, 
                                 'VIR_RI_Serial_Task_Accuracy.png'))


        vir_ri_acc = [[], []]

        for ri in list(vir_ri_serial_acc.keys()):
            vir_ri_acc[0].append(ri)
            vir_ri_acc[1].append(sum(vir_ri_serial_acc[ri][0])/len(vir_ri_serial_acc[ri][0]))

        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_subplot(111)

        ax.plot(vir_ri_acc[0], vir_ri_acc[1], marker='o')

        ax.set_xlabel('RI')
        ax.set_xticks(vir_ri_acc[0])
        ax.set_ylabel('Accuracy')

        ax.set_title('VIR RI Accuracy')

        self.logger.info('VIR Task RI Accuracy: {}'.format(vir_ri_acc))
        plt.savefig(os.path.join(save_path, 
                                 'VIR_RI_Accuracy.png'))

        # VSR Task
        vsr_list_serial_acc = {}

        for key in list(epoch_acc.keys()):
            if 'VSR_Task_List' in key and 'Serial' in key:
                list_length = int(key.split('_')[4])
                if list_length not in list(vsr_list_serial_acc.keys()):
                    vsr_list_serial_acc[list_length] = [[], [i for i in range(1, list_length+1)]]

        for key in list(vsr_list_serial_acc.keys()):
            for pos in vsr_list_serial_acc[key][1]:
                corr, tot = epoch_acc['VSR_Task_List_Length_{}_Serial_Position_{}'.format(key, pos)]
                vsr_list_serial_acc[key][0].append(corr/tot)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        for key in list(vsr_list_serial_acc.keys()):
            ax.plot(vsr_list_serial_acc[key][1], vsr_list_serial_acc[key][0], 
                    label='List Length: {}'.format(key), marker='o')

        ax.set_xlabel("Serial Position")
        ax.set_ylabel("Accuracy")
        ax.set_title("Visual Serial Recall Accuracy")
        ax.legend()

        self.logger.info('VSR Task List Length Serial Position Accuracy: {}'.format(vsr_list_serial_acc))
        plt.savefig(os.path.join(save_path, 
                                 'VSR_Task_List_Length_Serial_Position_Accuracy.png'))


        # SFR Task
        sfr_order_acc = [[epoch_acc['SFR_Task_Order'][0]/epoch_acc['SFR_Task_Order'][1], 
                  epoch_acc['SFR_Task_No_Order'][0]/epoch_acc['SFR_Task_No_Order'][1]], 
                  ['Forward Order', 'No Order']]

        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_subplot(111)

        ax.bar(sfr_order_acc[1], sfr_order_acc[0], color=['tab:orange', 'tab:green'])

        ax.set_xlabel('Order Of Recall')
        ax.set_ylabel('Accuracy')

        ax.set_title('Spatial Free Recall Order Of Recall Accuracy')

        self.logger.info('SFR Task Order Of Recall Accuracy: {}'.format(sfr_order_acc))
        plt.savefig(os.path.join(save_path, 
                                 'SFR_Task_Order_Of_Recall_Accuracy.png'))


        sfr_list_serial_acc = {}

        for key in list(epoch_acc.keys()):
            if 'SFR_Task_List' in key and 'Serial' in key:
                list_length = int(key.split('_')[4])
                if list_length not in list(sfr_list_serial_acc.keys()):
                    sfr_list_serial_acc[list_length] = [[], [i for i in range(1, list_length+1)]]

        for key in list(sfr_list_serial_acc.keys()):
            for pos in sfr_list_serial_acc[key][1]:
                corr, tot = epoch_acc['SFR_Task_List_Length_{}_Serial_Position_{}'.format(key, pos)]
                sfr_list_serial_acc[key][0].append(corr/tot)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        for key in list(sfr_list_serial_acc.keys()):
            ax.plot(sfr_list_serial_acc[key][1], sfr_list_serial_acc[key][0], 
                    label='List Length: {}'.format(key), marker='o')

        ax.set_xlabel("Serial Position")
        ax.set_xticks(sfr_list_serial_acc[key][1])
        ax.set_ylabel("Accuracy")
        ax.set_title("Spatial Free Recall Accuracy")
        ax.legend()

        self.logger.info('SFR Task List Length Serial Position Accuracy: {}'.format(sfr_list_serial_acc))
        plt.savefig(os.path.join(save_path, 
                                 'SFR_Task_List_Length_Serial_Position_Accuracy.png'))
        

        # CD Color Task
        cd_color_ri_setsize_acc = {}
        cd_color_ri = []
        cd_color_setsize = []

        for key in list(epoch_acc.keys()):
            if 'CD_Color' in key and 'RI' in key and 'Set_Size' not in key:
                ri = int(key.split('_')[4])
                if ri not in cd_color_ri:
                    cd_color_ri.append(ri)
            if 'CD_Color' in key and 'Set_Size' in key and 'RI' not in key:
                set_size = int(key.split('_')[5])
                if set_size not in cd_color_setsize:
                    cd_color_setsize.append(set_size)

        cd_color_ri.sort()
        cd_color_setsize.sort()

        for ri in cd_color_ri:
            cd_color_ri_setsize_acc[ri] = [[], []]
            for setsize in cd_color_setsize:
                corr, tot = epoch_acc['CD_Color_Task_RI_{}_Set_Size_{}'.format(ri, setsize)]
                cd_color_ri_setsize_acc[ri][0].append(corr/tot)
                cd_color_ri_setsize_acc[ri][1].append(setsize)      

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        for ri in list(cd_color_ri_setsize_acc.keys()):
            ax.plot(cd_color_ri_setsize_acc[ri][1], cd_color_ri_setsize_acc[ri][0], 
                    label='RI: {}'.format(ri), marker='o')

        ax.set_xlabel("Set Size")
        ax.set_xticks(list(cd_color_ri_setsize_acc.values())[-1][1])
        ax.set_ylabel("Accuracy")

        ax.set_title("CD Color RI Set Size Accuracy")
        ax.legend()

        self.logger.info('CD Color Task RI Set Size Accuracy: {}'.format(cd_color_ri_setsize_acc))
        plt.savefig(os.path.join(save_path, 
                                 'CD_Color_RI_Set_Size_Task_Accuracy.png'))


        # CD Orientation Task
        cd_orientation_ri_setsize_acc = {}
        cd_orientation_ri = []
        cd_orientation_setsize = []

        for key in list(epoch_acc.keys()):
            if 'CD_Orientation' in key and 'RI' in key and 'Set_Size' not in key:
                ri = int(key.split('_')[4])
                if ri not in cd_orientation_ri:
                    cd_orientation_ri.append(ri)
            if 'CD_Orientation' in key and 'Set_Size' in key and 'RI' not in key:
                set_size = int(key.split('_')[5])
                if set_size not in cd_orientation_setsize:
                    cd_orientation_setsize.append(set_size)

        cd_orientation_ri.sort()
        cd_orientation_setsize.sort()

        for ri in cd_orientation_ri:
            cd_orientation_ri_setsize_acc[ri] = [[], []]
            for setsize in cd_orientation_setsize:
                corr, tot = epoch_acc['CD_Orientation_Task_RI_{}_Set_Size_{}'.format(ri, setsize)]
                cd_orientation_ri_setsize_acc[ri][0].append(corr/tot)
                cd_orientation_ri_setsize_acc[ri][1].append(setsize)    

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        for ri in list(cd_orientation_ri_setsize_acc.keys()):
            ax.plot(cd_orientation_ri_setsize_acc[ri][1], cd_orientation_ri_setsize_acc[ri][0], 
                    label='RI: {}'.format(ri), marker='o')

        ax.set_xlabel("Set Size")
        ax.set_xticks(list(cd_orientation_ri_setsize_acc.values())[-1][1])
        ax.set_ylabel("Accuracy")

        ax.set_title("CD Orientation RI Set Size Accuracy")
        ax.legend()

        self.logger.info('CD Orientation Task RI Set Size Accuracy: {}'.format(cd_orientation_ri_setsize_acc))
        plt.savefig(os.path.join(save_path, 
                                 'CD_Orientation_RI_Set_Size_Task_Accuracy.png'))
        

        # CD Size Task
        cd_size_ri_setsize_acc = {}
        cd_size_ri = []
        cd_size_setsize = []

        for key in list(epoch_acc.keys()):
            if 'CD_Size' in key and 'RI' in key and 'Set_Size' not in key:
                ri = int(key.split('_')[4])
                if ri not in cd_size_ri:
                    cd_size_ri.append(ri)
            if 'CD_Size' in key and 'Set_Size' in key and 'RI' not in key:
                set_size = int(key.split('_')[5])
                if set_size not in cd_size_setsize:
                    cd_size_setsize.append(set_size)

        cd_size_ri.sort()
        cd_size_setsize.sort()

        for ri in cd_size_ri:
            cd_size_ri_setsize_acc[ri] = [[], []]
            for setsize in cd_size_setsize:
                corr, tot = epoch_acc['CD_Size_Task_RI_{}_Set_Size_{}'.format(ri, setsize)]
                cd_size_ri_setsize_acc[ri][0].append(corr/tot)
                cd_size_ri_setsize_acc[ri][1].append(setsize)     

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        for ri in list(cd_size_ri_setsize_acc.keys()):
            ax.plot(cd_size_ri_setsize_acc[ri][1], cd_size_ri_setsize_acc[ri][0], 
                    label='RI: {}'.format(ri), marker='o')

        ax.set_xlabel("Set Size")
        ax.set_xticks(list(cd_size_ri_setsize_acc.values())[-1][1])
        ax.set_ylabel("Accuracy")

        ax.set_title("CD Size RI Set Size Accuracy")
        ax.legend()

        self.logger.info('CD Size Task RI Set Size Accuracy: {}'.format(cd_size_ri_setsize_acc))
        plt.savefig(os.path.join(save_path, 
                                 'CD_Size_RI_Set_Size_Task_Accuracy.png'))
        

        # CD Gap Task
        cd_gap_ri_setsize_acc = {}
        cd_gap_ri = []
        cd_gap_setsize = []

        for key in list(epoch_acc.keys()):
            if 'CD_Gap' in key and 'RI' in key and 'Set_Size' not in key:
                ri = int(key.split('_')[4])
                if ri not in cd_gap_ri:
                    cd_gap_ri.append(ri)
            if 'CD_Gap' in key and 'Set_Size' in key and 'RI' not in key:
                set_size = int(key.split('_')[5])
                if set_size not in cd_gap_setsize:
                    cd_gap_setsize.append(set_size)

        cd_gap_ri.sort()
        cd_gap_setsize.sort()

        for ri in cd_gap_ri:
            cd_gap_ri_setsize_acc[ri] = [[], []]
            for setsize in cd_gap_setsize:
                corr, tot = epoch_acc['CD_Gap_Task_RI_{}_Set_Size_{}'.format(ri, setsize)]
                cd_gap_ri_setsize_acc[ri][0].append(corr/tot)
                cd_gap_ri_setsize_acc[ri][1].append(setsize)    

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        for ri in list(cd_gap_ri_setsize_acc.keys()):
            ax.plot(cd_gap_ri_setsize_acc[ri][1], cd_gap_ri_setsize_acc[ri][0], 
                    label='RI: {}'.format(ri), marker='o')

        ax.set_xlabel("Set Size")
        ax.set_xticks(list(cd_gap_ri_setsize_acc.values())[-1][1])
        ax.set_ylabel("Accuracy")

        ax.set_title("CD Gap RI Set Size Accuracy")
        ax.legend()

        self.logger.info('CD Gap Task RI Set Size Accuracy: {}'.format(cd_gap_ri_setsize_acc))
        plt.savefig(os.path.join(save_path, 
                                 'CD_Gap_RI_Set_Size_Task_Accuracy.png'))
        

        # CD Conjunction Task
        cd_conj_ri_setsize_acc = {}
        cd_conj_ri = []
        cd_conj_setsize = []

        for key in list(epoch_acc.keys()):
            if 'CD_Conj' in key and 'RI' in key and 'Set_Size' not in key:
                ri = int(key.split('_')[4])
                if ri not in cd_conj_ri:
                    cd_conj_ri.append(ri)
            if 'CD_Conj' in key and 'Set_Size' in key and 'RI' not in key:
                set_size = int(key.split('_')[5])
                if set_size not in cd_conj_setsize:
                    cd_conj_setsize.append(set_size)

        cd_conj_ri.sort()
        cd_conj_setsize.sort()

        for ri in cd_conj_ri:
            cd_conj_ri_setsize_acc[ri] = [[], []]
            for setsize in cd_conj_setsize:
                corr, tot = epoch_acc['CD_Conj_Task_RI_{}_Set_Size_{}'.format(ri, setsize)]
                cd_conj_ri_setsize_acc[ri][0].append(corr/tot)
                cd_conj_ri_setsize_acc[ri][1].append(setsize)      

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        for ri in list(cd_conj_ri_setsize_acc.keys()):
            ax.plot(cd_conj_ri_setsize_acc[ri][1], cd_conj_ri_setsize_acc[ri][0], 
                    label='RI: {}'.format(ri), marker='o')

        ax.set_xlabel("Set Size")
        ax.set_xticks(list(cd_conj_ri_setsize_acc.values())[-1][1])
        ax.set_ylabel("Accuracy")

        ax.set_title("CD Conj RI Set Size Accuracy")
        ax.legend()

        self.logger.info('CD Conj Task RI Set Size Accuracy: {}'.format(cd_conj_ri_setsize_acc))
        plt.savefig(os.path.join(save_path, 
                                 'CD_Conj_RI_Set_Size_Task_Accuracy.png'))