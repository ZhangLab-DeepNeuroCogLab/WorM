import torch
import torch.nn as nn

class WM_Model(nn.Module):
    def __init__(self, config, device):
        super(WM_Model, self).__init__()

        self.config = config
        self.device = device

        self.use_cnn = config.use_cnn
        self.img_size = config.rs_img_size
        self.num_input_channels = config.num_input_channels
        self.max_seq_len = config.max_seq_len

        self.mem_architecture = config.mem_architecture
        
        if config.task_embedding_given == 'All_TS':
            self.mem_input_size = config.mem_input_size
        
        self.mem_hidden_size = config.mem_hidden_size
        self.mem_num_layers = config.mem_num_layers
        self.trf_dim_ff = config.trf_dim_ff

        if self.use_cnn:
            self.CNN_Encoder = nn.Sequential(
                nn.Conv2d(self.num_input_channels, 64, kernel_size=3, stride=1, padding=1), 
                nn.ReLU(), 
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size=2, stride=2), 
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size=2, stride=2), 
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), 
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )

            self.representation_size = 512
            self.projection_size = config.mem_input_size - config.num_tasks

            if config.projection == "nonlinear":
                self.ProjectionHead = nn.Sequential(
                    nn.Linear(self.representation_size, self.representation_size//2),
                    nn.ReLU(),
                    nn.Linear(self.representation_size//2, self.projection_size)
                )
            elif config.projection == "linear":
                self.ProjectionHead = nn.Linear(self.representation_size, 
                                                self.projection_size)

        # RNN
        if self.mem_architecture == "RNN":
            self.MEM = nn.RNN(
                input_size=self.mem_input_size,
                hidden_size=self.mem_hidden_size,
                num_layers=self.mem_num_layers,
                batch_first=True
            )
        elif self.mem_architecture == "LSTM":
            self.MEM = nn.LSTM(
                input_size=self.mem_input_size,
                hidden_size=self.mem_hidden_size,
                num_layers=self.mem_num_layers,
                batch_first=True
            )
        elif self.mem_architecture == "GRU":
            self.MEM = nn.GRU(
                input_size=self.mem_input_size,
                hidden_size=self.mem_hidden_size,
                num_layers=self.mem_num_layers,
                batch_first=True
            )
        elif self.mem_architecture == "TRF":
            self.positional_encoding = nn.Embedding(self.max_seq_len, self.mem_input_size)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.mem_input_size, nhead=8, dim_feedforward=self.trf_dim_ff, 
                                                       batch_first=True)
            self.MEM = nn.TransformerEncoder(encoder_layer, num_layers=self.mem_num_layers)

        # Classifier
        if config.classifier == "linear":
            self.sc_task_head = nn.Linear(self.mem_hidden_size, 1)
            self.sfr_task_head = nn.Linear(self.mem_hidden_size, 100)
            self.si_task_head = nn.Linear(self.mem_hidden_size, 1)
            self.smu_task_head = nn.Linear(self.mem_hidden_size, 9)
            self.sts_task_head = nn.Linear(self.mem_hidden_size, 3)
            self.vir_task_head = nn.Linear(self.mem_hidden_size, 1)
            self.vsr_task_head = nn.Linear(self.mem_hidden_size, 9)
            self.vsrec_task_head = nn.Linear(self.mem_hidden_size, 1)
            self.cd_task_head = nn.Linear(self.mem_hidden_size, 1)
            self.cs_task_head = nn.Linear(self.mem_hidden_size, 21)

        
        if config.task_embedding_given == 'First_TS':
            if config.task_embedding == 'Learned':
                self.task_embedding = nn.Embedding(config.num_tasks, self.mem_input_size)
        elif config.task_embedding_given == 'All_TS':
            if config.task_embedding == 'Learned':
                self.task_embedding = nn.Embedding(config.num_tasks, config.num_tasks)
            elif config.task_embedding == 'Static':
                self.task_embedding = nn.Embedding(config.num_tasks, config.num_tasks)
                self.task_embedding.weight.data = torch.eye(config.num_tasks)
                self.task_embedding.weight.requires_grad = False

        self.task_id_dict = {
            'SC_Task': 0,
            'SFR_Task': 1, 
            'SI_Task': 2, 
            'SMU_Task': 3, 
            'STS_Task': 4, 
            'VIR_Task': 5, 
            'VSR_Task': 6, 
            'VSRec_Task': 7, 
            'CD_Color_Task': 8, 
            'CD_Orientation_Task': 9, 
            'CD_Size_Task': 10, 
            'CD_Gap_Task': 11, 
            'CD_Conj_Task': 12, 
            'CS_Task': 13
        }
        
        if self.mem_architecture in ['LSTM', 'GRU', 'RNN']:
            self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
    def forward(self, seq, task, actual_seq_len=None):

        batch_size = seq.shape[0]
        seq_length = seq.shape[1]

        if self.use_cnn:
            seq = seq.view(batch_size*seq_length, self.num_input_channels, 
                        self.img_size, self.img_size)
            cnn_out = self.CNN_Encoder(seq)

            cnn_out = cnn_out.view(batch_size*seq_length, self.representation_size)
            
            proj_out = self.ProjectionHead(cnn_out)
            proj_out = proj_out.view(batch_size, seq_length, self.projection_size)
        else:
            proj_out = seq.view(batch_size, seq_length, self.projection_size)

        if self.config.task_embedding_given == 'First_TS':
            task_embedding = self.task_embedding(torch.tensor([self.task_id_dict[task]]).to(self.device))
            task_embedding = task_embedding.repeat(batch_size, 1)
            task_embedding = task_embedding.unsqueeze(1)
            proj_out = torch.cat((task_embedding, proj_out), dim=1)
        elif self.config.task_embedding_given == 'All_TS':
            task_embedding = self.task_embedding(torch.tensor([self.task_id_dict[task]]).to(self.device))
            task_embedding = task_embedding.repeat(batch_size, seq_length, 1)
            proj_out = torch.cat((task_embedding, proj_out), dim=2)


        if self.mem_architecture in ['RNN', 'GRU']:
            self.MEM.flatten_parameters()
            mem_out, hn = self.MEM(proj_out)
            mem_out_reshaped = mem_out.contiguous().view(-1, self.mem_hidden_size)
        elif self.mem_architecture == 'LSTM':
            self.MEM.flatten_parameters()
            mem_out, (hn, cn) = self.MEM(proj_out)
            mem_out_reshaped = mem_out.contiguous().view(-1, self.mem_hidden_size)
        elif self.mem_architecture == 'TRF':
            hn = None
            mask = torch.triu(torch.ones(20, 20), diagonal=1).bool().to(self.device)

            if actual_seq_len is not None:
                src_key_padding_mask = torch.zeros(batch_size, seq_length).bool().to(self.device)
                for i in range(batch_size):
                    src_key_padding_mask[i, actual_seq_len[i]:] = True
           
            position_ids = torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            positional_encodings = self.positional_encoding(position_ids)

            proj_out = proj_out + positional_encodings

            mem_out = self.MEM(proj_out, mask=mask, src_key_padding_mask=src_key_padding_mask)
            mem_out_reshaped = mem_out.view(-1, self.mem_hidden_size)

        if task == 'SC_Task':
            out = self.sc_task_head(mem_out_reshaped)
        elif task == 'SFR_Task':
            out = self.sfr_task_head(mem_out_reshaped)
        elif task == 'SI_Task':
            out = self.si_task_head(mem_out_reshaped)
        elif task == 'SMU_Task':
            out = self.smu_task_head(mem_out_reshaped)
        elif task == 'STS_Task':
            out = self.sts_task_head(mem_out_reshaped)
        elif task == 'VIR_Task':
            out = self.vir_task_head(mem_out_reshaped)
        elif task == 'VSR_Task':
            out = self.vsr_task_head(mem_out_reshaped)
        elif task == 'VSRec_Task':
            out = self.vsrec_task_head(mem_out_reshaped)
        elif task in ['CD_Color_Task', 'CD_Orientation_Task', 
                      'CD_Size_Task', 'CD_Gap_Task', 'CD_Conj_Task']:
            out = self.cd_task_head(mem_out_reshaped)
        elif task == 'CS_Task':
            out = self.cs_task_head(mem_out_reshaped)

        out = out.view(batch_size, seq_length, -1)

        if self.use_cnn:
            return out, mem_out, hn, proj_out, cnn_out
        else:
            return out, mem_out, hn, proj_out, None