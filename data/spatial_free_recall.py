import os
import json
import random
from tqdm import tqdm
from PIL import Image, ImageDraw

class Spatial_Free_Recall_DataGen:
    '''
    Task: Spatial Free Recall
    '''

    def __init__(self, data_path, grid_size, set_size_options, 
                 list_length_options, gen_random_trials, held_out_set_sizes, 
                 held_out_list_lengths, num_samples, img_size, write):

        self.data_path = data_path
        self.grid_size = grid_size
        self.num_samples = num_samples
        self.img_size = (img_size, img_size)
        self.write = write

        self.bg_color = 'gray'
        self.active_cell_color = 'red'
        self.inactive_cell_color = 'green'
        self.cell_outline_color = 'gray'
        self.cell_outline_width = 2
        self.stim_img_padding = 3

        self.color_map = {'red': '#ee1d23', 'blue': '#015fae', 'violet': '#bc1b8d', 
                          'green': '#3ab54a', 'yellow': 'yellow', 'black': 'black', 
                          'white': 'white', 'brown': '#b78565', 'gray': '#d6d6d6'}

        self.task_data_path = os.path.join(self.data_path, 'Spatial_Free_Recall')

        if gen_random_trials:
            self.train_data_dir = os.path.join(self.task_data_path, 'train')
            self.test_data_dir = os.path.join(self.task_data_path, 'test')
            self.gen_test_data_dir = os.path.join(self.task_data_path, 'gen_test')

            self.train_num_samples = num_samples
            self.test_num_samples = int(num_samples * 0.1)
            self.gen_test_num_samples = int(num_samples * 0.1)

            assert self.train_num_samples % (len(set_size_options)*len(list_length_options)) == 0
            assert self.test_num_samples % (len(set_size_options)*len(list_length_options)) == 0
            if len(held_out_set_sizes) > 0 and len(held_out_list_lengths) > 0:
                assert self.gen_test_num_samples % (len(held_out_set_sizes)*len(held_out_list_lengths)) == 0

            assert min(set_size_options) >= max(list_length_options)
            if len(held_out_set_sizes) > 0 and len(held_out_list_lengths) > 0:
                assert min(held_out_set_sizes) >= max(held_out_list_lengths)

            if not os.path.exists(self.train_data_dir):
                os.makedirs(self.train_data_dir, exist_ok=True)
            if not os.path.exists(self.test_data_dir):
                os.makedirs(self.test_data_dir, exist_ok=True)
            if not os.path.exists(self.gen_test_data_dir):
                os.makedirs(self.gen_test_data_dir, exist_ok=True)

            trials = self.gen_random_trials(set_size_options, held_out_set_sizes, 
                                            list_length_options, held_out_list_lengths)

            json.dump(trials, open(os.path.join(self.task_data_path, 'data_rand_trials.json'), 'w'))

            _ = self.draw_trials_stim(trials)
        else:
            raise NotImplementedError
        
    def gen_random_trials(self, set_sizes, held_out_set_sizes, 
                          list_lengths, held_out_list_lengths):
        trials = {'train': [], 'test': [], 'gen_test': []}

        for split in ['train', 'test', 'gen_test']:
            print('Generating {} trials'.format(split))
            if split == 'train':
                num_samples = self.train_num_samples
                set_size_options = set_sizes
                list_length_options = list_lengths
            elif split == 'test':
                num_samples = self.test_num_samples
                set_size_options = set_sizes
                list_length_options = list_lengths
            elif split == 'gen_test':
                num_samples = self.gen_test_num_samples
                set_size_options = held_out_set_sizes
                list_length_options = held_out_list_lengths

            overall_sample_count = 0
            per_condition_sample_count = {}

            if len(set_size_options) > 0  and len(list_length_options) > 0:
                num_samples_per_condition = num_samples // (len(set_size_options)*len(list_length_options))

            for set_size in set_size_options:
                for list_length in tqdm(list_length_options):
                    print('Generating {} trials for set_size {}, list_length {}'.format(split, set_size, 
                                                                                        list_length))
                    per_condition_sample_count[(set_size, list_length)] = 0
                    
                    while per_condition_sample_count[(set_size, list_length)] < num_samples_per_condition:
                        visible_cells = random.sample(range(self.grid_size**2), set_size)
                        recall_gt = random.sample(visible_cells, list_length)

                        memory_stim_fnames = [split+'_'+str(overall_sample_count).zfill(6)+
                                              '_memory_'+str(i).zfill(2)+'.png' for i in range(list_length)]
                        probe_stim_fnames = [split+'_'+str(overall_sample_count).zfill(6)+'_probe.png']

                        trials[split].append({'grid_size': self.grid_size, 
                                              'set_size': set_size, 
                                              'list_length': list_length, 
                                              'visible_cells': visible_cells,
                                              'recall_gt': recall_gt, 
                                              'trial_type': split, 
                                              'trial_id': split+'_'+str(overall_sample_count).zfill(6), 
                                              'memory_stim_fnames': memory_stim_fnames, 
                                              'probe_stim_fnames': probe_stim_fnames})

                        overall_sample_count += 1
                        per_condition_sample_count[(set_size, list_length)] += 1

        return trials
    
    def draw_trials_stim(self, trials):
        trials_stim_list = {'train': [], 'test': [], 'gen_test': []}

        for split in ['train', 'test', 'gen_test']:
            print('Generating {} stimuli'.format(split))
            if split == 'train':
                write_dir = self.train_data_dir
            elif split == 'test':
                write_dir = self.test_data_dir
            elif split == 'gen_test':
                write_dir = self.gen_test_data_dir

            for trial in tqdm(trials[split]):
                grid_size = trial['grid_size']
                visible_cells = trial['visible_cells']
                recall_gt = trial['recall_gt']

                memory_stim_list = self.draw_memory_stim(grid_size, visible_cells, recall_gt)
                probe_stim_list = self.draw_probe_stim(grid_size, visible_cells)

                if self.write:
                    memory_stim_fnames = trial['memory_stim_fnames']
                    probe_stim_fnames = trial['probe_stim_fnames']

                    for memory_stim, memory_stim_fname in zip(memory_stim_list, memory_stim_fnames):
                        memory_stim.save(os.path.join(write_dir, memory_stim_fname))
                    for probe_stim, probe_stim_fname in zip(probe_stim_list, probe_stim_fnames):
                        probe_stim.save(os.path.join(write_dir, probe_stim_fname))

                else:
                    trial_stim = {}
                    trial_stim['memory_stim_list'] = memory_stim_list
                    trial_stim['probe_stim_list'] = probe_stim_list

                    trials_stim_list[split].append(trial_stim)

        if not self.write:
            return trials_stim_list
        
    def draw_memory_stim(self, grid_size, visible_cells, recall_gt):
        memory_stim_list = []

        for curr_cell in recall_gt:
            memory_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
            draw = ImageDraw.Draw(memory_stim)

            grid_size_px = self.img_size[0] - 2*self.stim_img_padding
            grid_cell_size_px = grid_size_px // grid_size

            for cell in visible_cells:
                cell_x = cell % grid_size
                cell_y = cell // grid_size

                cell_x_px = cell_x * grid_cell_size_px + self.stim_img_padding
                cell_y_px = cell_y * grid_cell_size_px + self.stim_img_padding
                
                if cell == curr_cell:
                    draw.rectangle([cell_x_px, cell_y_px, 
                                    cell_x_px+grid_cell_size_px, cell_y_px+grid_cell_size_px], 
                                    fill=self.color_map[self.active_cell_color], 
                                    outline=self.color_map[self.cell_outline_color], 
                                    width=self.cell_outline_width)
                else:
                    draw.rectangle([cell_x_px, cell_y_px, 
                                    cell_x_px+grid_cell_size_px, cell_y_px+grid_cell_size_px], 
                                    fill=self.color_map[self.inactive_cell_color], 
                                    outline=self.color_map[self.cell_outline_color], 
                                    width=self.cell_outline_width)
                    
            memory_stim_list.append(memory_stim)

        return memory_stim_list

    def draw_probe_stim(self, grid_size, visible_cells):
        probe_stim_list = []

        probe_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
        draw = ImageDraw.Draw(probe_stim)

        # grid_size_px = self.img_size[0] - 2*self.stim_img_padding
        # grid_cell_size_px = grid_size_px // grid_size

        # for cell in visible_cells:
        #     cell_x = cell % grid_size
        #     cell_y = cell // grid_size

        #     cell_x_px = cell_x * grid_cell_size_px + self.stim_img_padding
        #     cell_y_px = cell_y * grid_cell_size_px + self.stim_img_padding

        #     draw.rectangle([cell_x_px, cell_y_px, 
        #                     cell_x_px+grid_cell_size_px, cell_y_px+grid_cell_size_px], 
        #                     fill=self.inactive_cell_color, outline=self.cell_outline_color, 
        #                     width=self.cell_outline_width)

        probe_stim_list.append(probe_stim)

        return probe_stim_list