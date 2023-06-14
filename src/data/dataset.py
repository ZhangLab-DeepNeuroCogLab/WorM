import os
import json
import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

from src.data.spatial_free_recall import Spatial_Free_Recall_DataGen
from src.data.spatial_integration import Spatial_Integration_DataGen
from src.data.spatial_coordination import Spatial_Coordination_DataGen
from src.data.spatial_task_switching import Spatial_Task_Switching_DataGen
from src.data.spatial_memory_updating import Spatial_Memory_Updating_DataGen
from src.data.visual_item_recognition import Visual_Item_Recognition_DataGen
from src.data.visual_serial_recall_recognition import Visual_Serial_Recall_Recognition_DataGen
from src.data.color_orientation_size_gap_change_detection import Color_Orientation_Size_Gap_Conjunction_Change_Detection_DataGen
from src.data.complex_span import Complex_Span_DataGen


class Spatial_Memory_Updating_Dataset(Dataset):

    def __init__(self, data_path, max_seq_len=20, grid_size=3, set_size_options=[1, 2, 3, 4], 
                 presentation_time_options=[1], num_updates_options=[8], 
                 held_out_set_sizes=[], held_out_num_updates=[], gen_random_trials=True,
                 num_samples=48000, img_size=224, rs_img_size=224, write=True, split='train'):
        
        self.max_seq_len = max_seq_len
        self.task_data_path = os.path.join(data_path, 'Spatial_Memory_Updating')

        if not os.path.exists(self.task_data_path):
            Spatial_Memory_Updating_DataGen(data_path, grid_size, set_size_options, 
                                            presentation_time_options, num_updates_options, 
                                            gen_random_trials, held_out_set_sizes, 
                                            held_out_num_updates, num_samples, img_size, write)
        else:
            print('Getting data for Spatial_Memory_Updating Task')
            print('Data already exists. Skipping data generation.')

        self.data_file = json.load(open(os.path.join(self.task_data_path, 'data_rand_trials.json'), 'r'))

        self.transform = transforms.Compose([
                transforms.Resize((rs_img_size, rs_img_size)),
                transforms.ToTensor()
                ])
        
        if split == 'train':
            self.data_dict = self.data_file['train']
            self.imgset_path = os.path.join(self.task_data_path, 'train')
        elif split == 'test':
            self.data_dict = self.data_file['test']
            self.imgset_path = os.path.join(self.task_data_path, 'test')
        elif split == 'gen_test':
            self.data_dict = self.data_file['gen_test']
            self.imgset_path = os.path.join(self.task_data_path, 'gen_test')

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        img_seq = []
        gt = []

        memory_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['memory_stim_fnames'][0])
        memory_img = Image.open(memory_stim_fname).convert('RGB')

        img_seq.append(self.transform(memory_img))
        gt.append(9)

        for i in range(self.data_dict[idx]['num_updates']):
            update_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['update_stim_fnames'][i])
            update_img = Image.open(update_stim_fname).convert('RGB')

            for j in range(self.data_dict[idx]['presentation_time']):
                img_seq.append(self.transform(update_img))
                gt.append(9)

        for i in range(self.data_dict[idx]['set_size']):
            probe_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['probe_stim_fnames'][i])
            probe_img = Image.open(probe_stim_fname).convert('RGB')

            img_seq.append(self.transform(probe_img))

        gt = gt + self.data_dict[idx]['probe_gt']

        seq_len = len(img_seq)
        while len(img_seq) < self.max_seq_len:
            img_seq.append(torch.zeros_like(img_seq[0]))
            gt.append(9)

        set_size = torch.tensor(self.data_dict[idx]['set_size'])

        img_seq = torch.stack(img_seq)
        gt = torch.tensor(gt)

        return img_seq, gt, seq_len, set_size


class Visual_Serial_Recall_Recognition_Dataset(Dataset):

    def __init__(self, data_path, max_seq_len=20, probe_variant='Recall', grid_size=6, 
                 list_length_options=[3, 4, 5, 6], distractor_difference_options=[2, 4, 6], 
                 gen_random_trials=True, held_out_list_lengths=[], held_out_distractor_diffs=[],
                 num_samples=48000, img_size=224, rs_img_size=224, write=True, split='train'):
        
        self.max_seq_len = max_seq_len
        self.probe_variant = probe_variant
        self.task_data_path = os.path.join(data_path, 'Visual_Serial_'+probe_variant)

        if not os.path.exists(self.task_data_path):
            Visual_Serial_Recall_Recognition_DataGen(data_path, grid_size, probe_variant, list_length_options, 
                                                     distractor_difference_options, gen_random_trials, 
                                                     held_out_list_lengths, held_out_distractor_diffs, 
                                                     num_samples, img_size, write)
        else:
            print('Getting data for Visual_Serial_'+probe_variant+' Task')
            print('Data already exists. Skipping data generation.')

        self.data_file = json.load(open(os.path.join(self.task_data_path, 'data_rand_trials.json'), 'r'))

        self.transform = transforms.Compose([
                transforms.Resize((rs_img_size, rs_img_size)), 
                transforms.ToTensor()
                ])
        
        if split == 'train':
            self.data_dict = self.data_file['train']
            self.imgset_path = os.path.join(self.task_data_path, 'train')
        elif split == 'test':
            self.data_dict = self.data_file['test']
            self.imgset_path = os.path.join(self.task_data_path, 'test')
        elif split == 'gen_test':
            self.data_dict = self.data_file['gen_test']
            self.imgset_path = os.path.join(self.task_data_path, 'gen_test')

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        img_seq = []
        gt = []

        for i in range(self.data_dict[idx]['list_length']):
            memory_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['memory_stim_fnames'][i])
            memory_img = Image.open(memory_stim_fname).convert('RGB')

            img_seq.append(self.transform(memory_img))
            if self.probe_variant == 'Recall':
                gt.append(6)
            elif self.probe_variant == 'Recognition':
                gt.append(2)
        
        if self.probe_variant == 'Recall':
            probe_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['probe_stim_fnames'][0])
            probe_img = Image.open(probe_stim_fname).convert('RGB')

            for i in range(self.data_dict[idx]['list_length']):
                img_seq.append(self.transform(probe_img))

            gt = gt + self.data_dict[idx]['recall_gt']

            seq_len = len(img_seq)
            while len(img_seq) < self.max_seq_len:
                img_seq.append(torch.zeros_like(img_seq[0]))
                gt.append(6)

            list_length = torch.tensor(self.data_dict[idx]['list_length'])

            img_seq = torch.stack(img_seq)
            gt = torch.tensor(gt)

            return img_seq, gt, seq_len, list_length
        elif self.probe_variant == 'Recognition':
            for i in range(self.data_dict[idx]['list_length']):
                probe_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['probe_stim_fnames'][i])
                probe_img = Image.open(probe_stim_fname).convert('RGB')

                img_seq.append(self.transform(probe_img))

            gt = gt + self.data_dict[idx]['recognition_gt']
            
            seq_len = len(img_seq)
            while len(img_seq) < self.max_seq_len:
                img_seq.append(torch.zeros_like(img_seq[0]))
                gt.append(2)

            list_length = torch.tensor(self.data_dict[idx]['list_length'])
            distractor_diff = torch.tensor(self.data_dict[idx]['distractor_diff'])

            img_seq = torch.stack(img_seq)
            gt = torch.tensor(gt)
            
            return img_seq, gt, seq_len, list_length, distractor_diff
        

class Spatial_Free_Recall_Dataset(Dataset):

    def __init__(self, data_path, max_seq_len=20, grid_size=6, set_size_options=[30], 
                 list_length_options=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18], 
                 gen_random_trials=True, held_out_set_sizes=[], held_out_list_lengths=[],
                 num_samples=48000, img_size=224, rs_img_size=224, write=True, split='train'):
        
        self.max_seq_len = max_seq_len
        self.task_data_path = os.path.join(data_path, 'Spatial_Free_Recall')

        if not os.path.exists(self.task_data_path):
            Spatial_Free_Recall_DataGen(data_path, grid_size, set_size_options, 
                                        list_length_options, gen_random_trials, 
                                        held_out_set_sizes, held_out_list_lengths, 
                                        num_samples, img_size, write)
        else:
            print('Getting data for Spatial_Free_Recall Task')
            print('Data already exists. Skipping data generation.')

        self.data_file = json.load(open(os.path.join(self.task_data_path, 'data_rand_trials.json'), 'r'))

        self.transform = transforms.Compose([
                transforms.Resize((rs_img_size, rs_img_size)), 
                transforms.ToTensor()
                ])
        
        if split == 'train':
            self.data_dict = self.data_file['train']
            self.imgset_path = os.path.join(self.task_data_path, 'train')
        elif split == 'test':
            self.data_dict = self.data_file['test']
            self.imgset_path = os.path.join(self.task_data_path, 'test')
        elif split == 'gen_test':
            self.data_dict = self.data_file['gen_test']
            self.imgset_path = os.path.join(self.task_data_path, 'gen_test')

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        img_seq = []

        for i in range(self.data_dict[idx]['list_length']):
            memory_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['memory_stim_fnames'][i])
            memory_img = Image.open(memory_stim_fname).convert('RGB')

            img_seq.append(self.transform(memory_img))

        probe_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['probe_stim_fnames'][0])
        probe_img = Image.open(probe_stim_fname).convert('RGB')

        img_seq.append(self.transform(probe_img))

        recall_gt = self.data_dict[idx]['recall_gt']
        recall_gt_orig = recall_gt.copy()

        while len(recall_gt_orig) < self.max_seq_len:
            recall_gt_orig.append(-1)

        seq_len = len(img_seq)
        while len(img_seq) < self.max_seq_len:
            img_seq.append(torch.zeros_like(img_seq[0]))

        img_seq = torch.stack(img_seq)
        recall_gt = torch.tensor(recall_gt)
        recall_gt_orig = torch.tensor(recall_gt_orig)
        
        gt_one_hot = torch.nn.functional.one_hot(recall_gt, 
                                                 num_classes=100)
        gt_one_hot = gt_one_hot.sum(dim=0)

        return img_seq, gt_one_hot, seq_len, recall_gt_orig
    

class Visual_Item_Recognition_Dataset(Dataset):

    def __init__(self, data_path, max_seq_len=20, grid_size=6,  
                 list_length_options=[4], distractor_difference_options=[4], 
                 ri_options=[0, 5], gen_random_trials=True, held_out_list_lengths=[], 
                 held_out_distractor_diffs=[], held_out_ri_options=[],
                 num_samples=48000, img_size=224, rs_img_size=224, write=True, split='train'):
        
        self.max_seq_len = max_seq_len
        self.task_data_path = os.path.join(data_path, 'Visual_Item_Recognition')

        if not os.path.exists(self.task_data_path):
            Visual_Item_Recognition_DataGen(data_path, grid_size, list_length_options, 
                                               distractor_difference_options, ri_options, gen_random_trials, 
                                               held_out_list_lengths, held_out_distractor_diffs, held_out_ri_options, 
                                               num_samples, img_size, write)
        else:
            print('Getting data for Visual_Item_Recognition Task')
            print('Data already exists. Skipping data generation.')

        self.data_file = json.load(open(os.path.join(self.task_data_path, 'data_rand_trials.json'), 'r'))

        self.transform = transforms.Compose([
                transforms.Resize((rs_img_size, rs_img_size)), 
                transforms.ToTensor()
                ])
        
        self.blank_img = Image.open(os.path.join(self.task_data_path, 'blank.png')).convert('RGB')
        self.blank_img = self.transform(self.blank_img)
        
        if split == 'train':
            self.data_dict = self.data_file['train']
            self.imgset_path = os.path.join(self.task_data_path, 'train')
        elif split == 'test':
            self.data_dict = self.data_file['test']
            self.imgset_path = os.path.join(self.task_data_path, 'test')
        elif split == 'gen_test':
            self.data_dict = self.data_file['gen_test']
            self.imgset_path = os.path.join(self.task_data_path, 'gen_test')

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        img_seq = []
        gt = []

        for i in range(self.data_dict[idx]['list_length']):
            memory_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['memory_stim_fnames'][i])
            memory_img = Image.open(memory_stim_fname).convert('RGB')

            img_seq.append(self.transform(memory_img))
            gt.append(2)

        for i in range(self.data_dict[idx]['retention_interval']):
            img_seq.append(self.blank_img)
            gt.append(2)

        probe_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['probe_stim_fnames'][0])
        probe_img = Image.open(probe_stim_fname).convert('RGB')
        
        img_seq.append(self.transform(probe_img))

        gt.append(self.data_dict[idx]['recognition_gt'])

        seq_len = len(img_seq)
        while len(img_seq) < self.max_seq_len:
            img_seq.append(torch.zeros_like(img_seq[0]))
            gt.append(2)

        ri = torch.tensor(self.data_dict[idx]['retention_interval'])
        gt_index = torch.tensor(self.data_dict[idx]['recognition_gt_index'])

        img_seq = torch.stack(img_seq)
        gt = torch.tensor(gt)

        return img_seq, gt, seq_len, ri, gt_index
    

class Spatial_Coordination_Dataset(Dataset):

    def __init__(self, data_path, max_seq_len=20, grid_size=8, set_size_options=[4], 
                 symmetry_offset_options=[4], gen_random_trials=True, 
                 held_out_set_sizes=[], held_out_symmetry_offsets=[],
                 num_samples=24000, img_size=224, rs_img_size=224, write=True, 
                 show_gt_pattern=True, split='train'):
        
        self.max_seq_len = max_seq_len
        self.show_gt_pattern = show_gt_pattern
        self.task_data_path = os.path.join(data_path, 'Spatial_Coordination')

        if not os.path.exists(self.task_data_path):
            Spatial_Coordination_DataGen(data_path, grid_size, set_size_options, 
                                         symmetry_offset_options, gen_random_trials, 
                                         held_out_set_sizes, held_out_symmetry_offsets, 
                                         num_samples, img_size, write)
        else:
            print('Getting data for Spatial_Coordination Task')
            print('Data already exists. Skipping data generation.')

        self.data_file = json.load(open(os.path.join(self.task_data_path, 'data_rand_trials.json'), 'r'))

        self.transform = transforms.Compose([
                transforms.Resize((rs_img_size, rs_img_size)), 
                transforms.ToTensor(),
                ])
        
        self.probe_img = Image.open(os.path.join(self.task_data_path, 'probe.png')).convert('RGB')
        self.probe_img = self.transform(self.probe_img)
        
        if split == 'train':
            self.data_dict = self.data_file['train']
            self.imgset_path = os.path.join(self.task_data_path, 'train')
        elif split == 'test':
            self.data_dict = self.data_file['test']
            self.imgset_path = os.path.join(self.task_data_path, 'test')
        elif split == 'gen_test':
            self.data_dict = self.data_file['gen_test']
            self.imgset_path = os.path.join(self.task_data_path, 'gen_test')

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        img_seq = []
        gt = []

        for i in range(self.data_dict[idx]['set_size']):
            memory_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['memory_stim_fnames'][i])
            memory_img = Image.open(memory_stim_fname).convert('RGB')

            img_seq.append(self.transform(memory_img))
            gt.append(2)

        img_seq.append(self.probe_img)

        gt.append(self.data_dict[idx]['gt'])

        seq_len = len(img_seq)
        while len(img_seq) < self.max_seq_len:
            img_seq.append(torch.zeros_like(img_seq[0]))
            gt.append(2)

        symmetry_offset = torch.tensor(self.data_dict[idx]['symmetry_offset'])

        img_seq = torch.stack(img_seq)
        gt = torch.tensor(gt)

        if self.show_gt_pattern:
            gt_pattern_fname = os.path.join(self.imgset_path, self.data_dict[idx]['gt_pattern_fname'])
            gt_pattern_img = Image.open(gt_pattern_fname).convert('RGB')
            gt_pattern_img = self.transform(gt_pattern_img)

            return img_seq, gt, gt_pattern_img, seq_len, symmetry_offset
        else:
            return img_seq, gt, seq_len, symmetry_offset
        

class Spatial_Integration_Dataset(Dataset):

    def __init__(self, data_path, max_seq_len=20, grid_size_options=[4], pattern_size_options=[12], 
                 part_size_options=[3, 4, 6], distractor_difference_options=[1], 
                 gen_random_trials=True, held_out_grid_sizes=[], 
                 held_out_pattern_sizes=[], held_out_part_sizes=[], 
                 held_out_distractor_diffs=[], num_samples=48000, img_size=224, rs_img_size=224, 
                 write=True, split='train'):
        
        self.max_seq_len = max_seq_len
        self.task_data_path = os.path.join(data_path, 'Spatial_Integration')

        if not os.path.exists(self.task_data_path):
            Spatial_Integration_DataGen(data_path, grid_size_options, pattern_size_options, 
                                        part_size_options, distractor_difference_options, 
                                        gen_random_trials, held_out_grid_sizes, 
                                        held_out_pattern_sizes, held_out_part_sizes, 
                                        held_out_distractor_diffs, num_samples, img_size, write)
        else:
            print('Getting data for Spatial_Integration Task')
            print('Data already exists. Skipping data generation.')

        self.data_file = json.load(open(os.path.join(self.task_data_path, 'data_rand_trials.json'), 'r'))

        self.transform = transforms.Compose([
                transforms.Resize((rs_img_size, rs_img_size)), 
                transforms.ToTensor(),
                ])
        
        if split == 'train':
            self.data_dict = self.data_file['train']
            self.imgset_path = os.path.join(self.task_data_path, 'train')
        elif split == 'test':
            self.data_dict = self.data_file['test']
            self.imgset_path = os.path.join(self.task_data_path, 'test')
        elif split == 'gen_test':
            self.data_dict = self.data_file['gen_test']
            self.imgset_path = os.path.join(self.task_data_path, 'gen_test')

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        img_seq = []
        gt = []

        for i in range(self.data_dict[idx]['memory_length']):
            memory_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['memory_stim_fnames'][i])
            memory_img = Image.open(memory_stim_fname).convert('RGB')

            img_seq.append(self.transform(memory_img))
            gt.append(2)

        probe_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['probe_stim_fnames'][0])
        probe_img = Image.open(probe_stim_fname).convert('RGB')
        
        img_seq.append(self.transform(probe_img))

        gt.append(self.data_dict[idx]['gt'])

        seq_len = len(img_seq)
        while len(img_seq) < self.max_seq_len:
            img_seq.append(torch.zeros_like(img_seq[0]))
            gt.append(2)

        part_size = torch.tensor(self.data_dict[idx]['part_size'])

        img_seq = torch.stack(img_seq)
        gt = torch.tensor(gt)

        return img_seq, gt, seq_len, part_size
    

class Spatial_Task_Switching_Dataset(Dataset):

    def __init__(self, data_path, max_seq_len=20, variant='Cued', trial_lengths_options=[10, 15, 20], 
                 gen_random_trials=True, held_out_trial_lengths=[],
                 num_samples=48000, img_size=224, rs_img_size=224, write=True, split='train'):
        
        self.max_seq_len = max_seq_len
        self.task_data_path = os.path.join(data_path, 'Spatial_Task_Switching_'+variant)

        if not os.path.exists(self.task_data_path):
            Spatial_Task_Switching_DataGen(data_path, variant, trial_lengths_options, 
                                           gen_random_trials, held_out_trial_lengths, 
                                           num_samples, img_size, write)
        else:
            print('Getting data for Spatial_Task_Switching Task')
            print('Data already exists. Skipping data generation.')

        self.data_file = json.load(open(os.path.join(self.task_data_path, 'data_rand_trials.json'), 'r'))

        self.transform = transforms.Compose([
                transforms.Resize((rs_img_size, rs_img_size)), 
                transforms.ToTensor()
                ])
        
        if split == 'train':
            self.data_dict = self.data_file['train']
            self.imgset_path = os.path.join(self.task_data_path, 'train')
        elif split == 'test':
            self.data_dict = self.data_file['test']
            self.imgset_path = os.path.join(self.task_data_path, 'test')
        elif split == 'gen_test':
            self.data_dict = self.data_file['gen_test']
            self.imgset_path = os.path.join(self.task_data_path, 'gen_test')

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        img_seq = []
        gt = []

        for i in range(self.data_dict[idx]['trial_length']):
            stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['stim_fnames'][i])
            img = Image.open(stim_fname).convert('RGB')

            img_seq.append(self.transform(img))

        gt = gt + self.data_dict[idx]['gt']

        seq_len = len(img_seq)
        while len(img_seq) < self.max_seq_len:
            img_seq.append(torch.zeros_like(img_seq[0]))
            gt.append(2)

        img_seq = torch.stack(img_seq)
        gt = torch.tensor(gt)

        return img_seq, gt, seq_len
    

class Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(Dataset):

    def __init__(self, data_path, max_seq_len=20, variant='Color', set_size_options=[2, 4, 6, 8, 12], 
                 ri_options=[2, 6, 10, 14, 18], gen_random_trials=True, held_out_set_size_options=[],
                 held_out_ri_options=[], num_samples=48000, img_size=224, rs_img_size=224, 
                 write=True, split='train'):
        
        self.max_seq_len = max_seq_len
        self.variant = variant

        self.task_data_path = os.path.join(data_path, 'Color_Orientation_Size_Gap_Conjunction_Change_Detection_'+variant)

        if not os.path.exists(self.task_data_path):
            Color_Orientation_Size_Gap_Conjunction_Change_Detection_DataGen(
                data_path, variant, set_size_options, ri_options,
                gen_random_trials, held_out_set_size_options, 
                held_out_ri_options, 
                num_samples, img_size, write)
        else:
            print('Getting data for Color_Orientation_Size_Gap_Conjunction_Change_Detection_'+variant+' Task')
            print('Data already exists. Skipping data generation.')

        self.data_file = json.load(open(os.path.join(self.task_data_path, 'data_rand_trials.json'), 'r'))

        self.conj_map = {'Color': 0, 'Orientation': 1, 'Size': 2, 'Gap': 3}

        self.transform = transforms.Compose([
                transforms.Resize((rs_img_size, rs_img_size)), 
                transforms.ToTensor()
                ])
        
        self.blank_img = Image.open(os.path.join(self.task_data_path, 'blank.png')).convert('RGB')
        self.blank_img = self.transform(self.blank_img)
        
        if split=='train':
            self.data_dict = self.data_file['train']
            self.imgset_path = os.path.join(self.task_data_path, 'train')
        elif split=='test':
            self.data_dict = self.data_file['test']
            self.imgset_path = os.path.join(self.task_data_path, 'test')
        elif split=='gen_test':
            self.data_dict = self.data_file['gen_test']
            self.imgset_path = os.path.join(self.task_data_path, 'gen_test')

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        img_seq = []
        gt = []

        stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['memory_stim_fnames'][0])
        img = Image.open(stim_fname).convert('RGB')
        img_seq.append(self.transform(img))
        gt.append(2)

        for i in range(self.data_dict[idx]['retention_interval']):
            img_seq.append(self.blank_img)
            gt.append(2)

        probe_fname = os.path.join(self.imgset_path, self.data_dict[idx]['probe_stim_fnames'][0])
        img = Image.open(probe_fname).convert('RGB')
        img_seq.append(self.transform(img))
        gt.append(self.data_dict[idx]['gt'])

        seq_len = len(img_seq)
        while len(img_seq) < self.max_seq_len:
            img_seq.append(torch.zeros_like(img_seq[0]))
            gt.append(2)

        ri = torch.tensor(self.data_dict[idx]['retention_interval'])
        set_size = torch.tensor(self.data_dict[idx]['set_size'])
        if self.variant == 'Conjunction':
            if self.data_dict[idx]['conjunction_gt'] is not None:
                conj_gt = torch.tensor(
                    self.conj_map[self.data_dict[idx]['conjunction_gt']]
                    )
            else:
                conj_gt = torch.tensor(4)

        img_seq = torch.stack(img_seq)
        gt = torch.tensor(gt)

        if self.variant == 'Conjunction':
            return img_seq, gt, seq_len, ri, set_size, conj_gt
        else:
            return img_seq, gt, seq_len, ri, set_size
        

class Complex_Span_Dataset(Dataset):

    def __init__(self, data_path, max_seq_len=20, num_storage_options=[2], num_distractor_options=[0, 1, 3, 5], 
                 visual_memory_grid_size=4, spatial_distractor_grid_size_options=[10], 
                 spatial_distractor_set_size_options=[20], spatial_distractor_symmetry_offset_options=[4], 
                 gen_random_trials=True, num_samples=48000, img_size=224, rs_img_size=224, 
                 write=True, split='train'):
        
        self.max_seq_len = max_seq_len
        self.task_data_path = os.path.join(data_path, 'Complex_Span')

        if not os.path.exists(self.task_data_path):
            Complex_Span_DataGen(data_path, num_storage_options, num_distractor_options, 
                               visual_memory_grid_size, spatial_distractor_grid_size_options, 
                               spatial_distractor_set_size_options, spatial_distractor_symmetry_offset_options, 
                               gen_random_trials,num_samples, img_size, write)
        else:
            print('Getting data for Complex_Span Task')
            print('Data already exists. Skipping data generation.')

        self.data_file = json.load(open(os.path.join(self.task_data_path, 'data_rand_trials.json'), 'r'))

        self.variation_map = {'spatial-spatial': 0, 
                              'spatial-visual': 1, 
                              'visual-spatial': 2, 
                              'visual-visual': 3}

        # Calculate mean and std of current dataset
        self.transform = transforms.Compose([
                transforms.Resize((rs_img_size, rs_img_size)), 
                transforms.ToTensor(),
                ])

        self.blank_img = Image.open(os.path.join(self.task_data_path, 'blank.png')).convert('RGB')
        self.blank_img = self.transform(self.blank_img)
        
        if split == 'train':
            self.data_dict = self.data_file['train']
            self.imgset_path = os.path.join(self.task_data_path, 'train')
        elif split == 'test':
            self.data_dict = self.data_file['test']
            self.imgset_path = os.path.join(self.task_data_path, 'test')
        elif split == 'gen_test':
            self.data_dict = self.data_file['gen_test']
            self.imgset_path = os.path.join(self.task_data_path, 'gen_test')

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        img_seq = []
        gt = []

        num_distractor = self.data_dict[idx]['num_distractor']
        variation = self.variation_map[self.data_dict[idx]['variation']]

        for i in range(3):
            memory_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['memory_stim_1_fnames'][i])
            memory_img = Image.open(memory_stim_fname).convert('RGB')

            img_seq.append(self.transform(memory_img))
            gt.append(20)

        if num_distractor == 1:
            img_seq.append(self.blank_img)
            img_seq.append(self.blank_img)

            gt.append(20)
            gt.append(20)
            
            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_1_fnames'][0])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][0])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][0])

            img_seq.append(self.blank_img)
            img_seq.append(self.blank_img)

            gt.append(20)
            gt.append(20)

        elif num_distractor == 3:
            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_1_fnames'][0])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][0])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][0])

            img_seq.append(self.blank_img)
            gt.append(20)

            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_1_fnames'][1])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][1])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][1])

            img_seq.append(self.blank_img)
            gt.append(20)

            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_1_fnames'][2])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][2])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][2])

        elif num_distractor == 5:
            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_1_fnames'][0])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_1_fnames'][1])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_1_fnames'][2])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_1_fnames'][3])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_1_fnames'][4])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][0])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][0])

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][1])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][1])

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][2])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][2])

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][3])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][3])

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][4])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][4])


        elif num_distractor == 0:
            img_seq.append(self.blank_img)
            img_seq.append(self.blank_img)
            img_seq.append(self.blank_img)
            img_seq.append(self.blank_img)
            img_seq.append(self.blank_img)

            gt.append(20)
            gt.append(20)
            gt.append(20)
            gt.append(20)
            gt.append(20)


        for i in range(3):
            memory_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['memory_stim_2_fnames'][i])
            memory_img = Image.open(memory_stim_fname).convert('RGB')

            img_seq.append(self.transform(memory_img))
            gt.append(20)


        if num_distractor == 1:
            img_seq.append(self.blank_img)
            img_seq.append(self.blank_img)

            gt.append(20)
            gt.append(20)
            
            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_2_fnames'][0])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][1])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][1])

            img_seq.append(self.blank_img)
            img_seq.append(self.blank_img)

            gt.append(20)
            gt.append(20)

        elif num_distractor == 3:
            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_2_fnames'][0])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][3])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][3])

            img_seq.append(self.blank_img)
            gt.append(20)

            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_2_fnames'][1])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][4])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][4])

            img_seq.append(self.blank_img)
            gt.append(20)

            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_2_fnames'][2])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][5])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][5])

        elif num_distractor == 5:
            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_2_fnames'][0])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_2_fnames'][1])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_2_fnames'][2])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_2_fnames'][3])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            distractor_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['distractor_stim_2_fnames'][4])
            distractor_img = Image.open(distractor_stim_fname).convert('RGB')
            img_seq.append(self.transform(distractor_img))

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][5])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][5])

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][6])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][6])

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][7])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][7])

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][8])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][8])

            if self.data_dict[idx]['variation'] in ['spatial-visual', 'visual-visual']:
                gt.append(self.data_dict[idx]['distractor_gts'][9])
            elif self.data_dict[idx]['variation'] in ['spatial-spatial', 'visual-spatial']:
                gt.append(2+self.data_dict[idx]['distractor_gts'][9])

        elif num_distractor == 0:
            img_seq.append(self.blank_img)
            img_seq.append(self.blank_img)
            img_seq.append(self.blank_img)
            img_seq.append(self.blank_img)
            img_seq.append(self.blank_img)

            gt.append(20)
            gt.append(20)
            gt.append(20)
            gt.append(20)
            gt.append(20)


        for i in range(2):
            probe_stim_fname = os.path.join(self.imgset_path, self.data_dict[idx]['probe_stim_fnames'][i])
            probe_img = Image.open(probe_stim_fname).convert('RGB')

            img_seq.append(self.transform(probe_img))
            if self.data_dict[idx]['variation'] in ['spatial-visual', 'spatial-spatial']:
                if i == 0:
                    gt.append(4+self.data_dict[idx]['recall_gt_1'])
                elif i == 1:
                    gt.append(4+self.data_dict[idx]['recall_gt_2'])
            elif self.data_dict[idx]['variation'] in ['visual-visual', 'visual-spatial']:
                if i == 0:
                    gt.append(12+self.data_dict[idx]['recall_gt_1'])
                elif i == 1:
                    gt.append(12+self.data_dict[idx]['recall_gt_2'])


        seq_len = 18
        while len(img_seq) < self.max_seq_len:
            img_seq.append(torch.zeros_like(img_seq[0]))
            gt.append(20)

        img_seq = torch.stack(img_seq)
        gt = torch.tensor(gt)

        return img_seq, gt, seq_len, num_distractor, variation