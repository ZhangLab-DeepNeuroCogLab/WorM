import os
import json
import random
from tqdm import tqdm
from PIL import Image, ImageDraw

class Spatial_Coordination_DataGen:
    '''
    Task: Spatial Coordination
    '''

    def __init__(self, data_path, grid_size, set_size_options, symmetry_offset_options,
                 gen_random_trials, held_out_set_sizes, held_out_symmetry_offsets, 
                 num_samples, img_size, write):

        self.data_path = data_path
        self.grid_size = grid_size
        self.num_samples = num_samples
        self.img_size = (img_size, img_size)
        self.write = write

        self.memory_bg_color = 'gray'
        self.probe_bg_color = 'gray'
        self.active_cell_color = 'green'
        self.inactive_cell_color = 'white'
        self.cell_outline_color = 'black'
        self.stim_img_padding = 3
        self.rect_radius = 2

        self.color_map = {'red': '#ee1d23', 'blue': '#015fae', 'violet': '#bc1b8d', 
                          'green': '#3ab54a', 'yellow': 'yellow', 'black': 'black', 
                          'white': 'white', 'brown': '#b78565', 'gray': '#d6d6d6'}

        self.task_data_path = os.path.join(self.data_path, 'Spatial_Coordination')

        if gen_random_trials:
            self.train_data_dir = os.path.join(self.task_data_path, 'train')
            self.test_data_dir = os.path.join(self.task_data_path, 'test')
            self.gen_test_data_dir = os.path.join(self.task_data_path, 'gen_test')

            self.train_num_samples = num_samples
            self.test_num_samples = int(num_samples * 0.1)
            self.gen_test_num_samples = int(num_samples * 0.1)

            assert self.train_num_samples % len(set_size_options) == 0
            assert self.test_num_samples % len(set_size_options) == 0
            if len(held_out_set_sizes) > 0:
                assert self.gen_test_num_samples % len(held_out_set_sizes) == 0

            assert self.grid_size % 2 == 0
            assert all([set_size % 2 == 0 for set_size in set_size_options])
            assert max(set_size_options) <= self.grid_size**2

            if len(held_out_set_sizes) > 0:
                assert all([set_size % 2 == 0 for set_size in held_out_set_sizes])
                assert max(held_out_set_sizes) <= self.grid_size**2

            assert all([symmetry_offset % 2 == 0 for symmetry_offset in symmetry_offset_options])

            trials = self.gen_random_trials(set_size_options, held_out_set_sizes, 
                                            symmetry_offset_options)

            if not os.path.exists(self.train_data_dir):
                os.makedirs(self.train_data_dir, exist_ok=True)
            if not os.path.exists(self.test_data_dir):
                os.makedirs(self.test_data_dir, exist_ok=True)
            if not os.path.exists(self.gen_test_data_dir):
                os.makedirs(self.gen_test_data_dir, exist_ok=True)

            json.dump(trials, open(os.path.join(self.task_data_path, 'data_rand_trials.json'), 'w'))

            _ = self.draw_trials_stim(trials)
        else:
            raise NotImplementedError
        
    def gen_random_trials(self, set_sizes, held_out_set_sizes, symmetry_offset_options):
        trials = {'train': [], 'test': [], 'gen_test': []}

        for split in ['train', 'test', 'gen_test']:
            print('Generating {} trials'.format(split))
            if split == 'train':
                num_samples = self.train_num_samples
                set_size_options = set_sizes
            elif split == 'test':
                num_samples = self.test_num_samples
                set_size_options = set_sizes
            elif split == 'gen_test':
                num_samples = self.gen_test_num_samples
                set_size_options = held_out_set_sizes

            overall_sample_count = 0
            per_set_size_sample_count = {}

            if len(set_size_options) > 0:
                num_samples_per_set_size = num_samples // len(set_size_options)

            for set_size in set_size_options:
                print('Generating {} trials for set size {}'.format(split, set_size))
                per_set_size_sample_count[set_size] = 0

                while per_set_size_sample_count[set_size] < num_samples_per_set_size:
                    left_grid_cell_idxs = [((random.choice(range(self.grid_size))*self.grid_size) + 
                                            random.choice(range(self.grid_size//2))) 
                                            for _ in range(set_size // 2)]
                                        
                    right_grid_cell_idxs = [(idx + (self.grid_size - 2*(idx%self.grid_size)) - 1) 
                                            for idx in left_grid_cell_idxs]

                    gt = random.choice([0, 1])

                    if gt == 1:
                        symmetry_offset = 0
                        gt_item = left_grid_cell_idxs + right_grid_cell_idxs
                    else:
                        symmetry_offset = random.choice(symmetry_offset_options)

                        while symmetry_offset > set_size:
                            symmetry_offset = random.choice(symmetry_offset_options)

                        right_inactive_grid_cell_idxs = [idx for idx in range(self.grid_size**2) 
                                                         if (idx not in right_grid_cell_idxs) and 
                                                         (idx%self.grid_size >= self.grid_size//2)]

                        right_grid_cell_idxs = random.sample(right_grid_cell_idxs, 
                                                             len(right_grid_cell_idxs)-symmetry_offset//2)

                        sample_inactive_right_grid_cell_idxs = random.sample(right_inactive_grid_cell_idxs, 
                                                                            symmetry_offset//2)
                        
                        gt_item = left_grid_cell_idxs + right_grid_cell_idxs + sample_inactive_right_grid_cell_idxs

                    random.shuffle(gt_item)

                    memory_stim_fnames = [split+'_'+str(overall_sample_count).zfill(6)+
                                          '_memory_'+str(i).zfill(3)+'.png' 
                                          for i in range(set_size)]
                    
                    gt_pattern_fname = split+'_'+str(overall_sample_count).zfill(6)+'_gt_pattern.png'
                    
                    trials[split].append({'grid_size': self.grid_size, 
                                          'set_size': set_size, 
                                          'max_set_size': max(set_size_options), 
                                          'symmetry_offset': symmetry_offset, 
                                          'gt_item': gt_item, 
                                          'gt': gt, 
                                          'trial_type': split, 
                                          'trial_id': split+'_'+str(overall_sample_count).zfill(6), 
                                          'gt_pattern_fname': gt_pattern_fname, 
                                          'memory_stim_fnames': memory_stim_fnames})
                    
                    overall_sample_count += 1
                    per_set_size_sample_count[set_size] += 1

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
                gt_item = trial['gt_item']

                memory_stim_list = self.draw_memory_stim(grid_size, gt_item)
                gt_pattern = self.draw_gt_pattern(grid_size, gt_item)

                if self.write:
                    memory_stim_fnames = trial['memory_stim_fnames']
                    gt_pattern_fname = trial['gt_pattern_fname']

                    for memory_stim, memory_stim_fname in zip(memory_stim_list, memory_stim_fnames):
                        memory_stim.save(os.path.join(write_dir, memory_stim_fname))
                    
                    gt_pattern.save(os.path.join(write_dir, gt_pattern_fname))

                else:
                    trials_stim = {}
                    trials_stim['memory_stim_list'] = memory_stim_list
                    trials_stim['gt_pattern'] = gt_pattern

                    trials_stim_list[split].append(trials_stim)

        if self.write:
            probe_stim = self.draw_probe_stim()
            probe_stim.save(os.path.join(self.task_data_path, 'probe.png'))
        else:
            return trials_stim_list
        
    def draw_gt_pattern(self, grid_size, gt_item):
        gt_pattern = Image.new('RGB', self.img_size, color=self.color_map[self.memory_bg_color])
        draw = ImageDraw.Draw(gt_pattern)

        grid_size_px = self.img_size[0] - 2*self.stim_img_padding
        grid_cell_size_px = grid_size_px // grid_size

        for cell in range(grid_size**2):
            cell_x = cell % grid_size
            cell_y = cell // grid_size

            cell_x_px = cell_x * grid_cell_size_px + self.stim_img_padding
            cell_y_px = cell_y * grid_cell_size_px + self.stim_img_padding

            if cell in gt_item:
                draw.rounded_rectangle([cell_x_px, cell_y_px, 
                                cell_x_px+grid_cell_size_px, cell_y_px+grid_cell_size_px], 
                                radius=self.rect_radius, 
                                fill=self.color_map[self.active_cell_color], 
                                outline=self.color_map[self.cell_outline_color])
            else:
                draw.rounded_rectangle([cell_x_px, cell_y_px, 
                                cell_x_px+grid_cell_size_px, cell_y_px+grid_cell_size_px], 
                                radius=self.rect_radius, 
                                fill=self.color_map[self.inactive_cell_color], 
                                outline=self.color_map[self.cell_outline_color])
                
        return gt_pattern

    def draw_memory_stim(self, grid_size, gt_item):
        memory_stim_list = []

        for gt_cell in gt_item:
            memory_stim = Image.new('RGB', self.img_size, color=self.color_map[self.memory_bg_color])
            draw = ImageDraw.Draw(memory_stim)

            grid_size_px = self.img_size[0] - 2*self.stim_img_padding
            grid_cell_size_px = grid_size_px // grid_size

            for cell in range(grid_size**2):
                cell_x = cell % grid_size
                cell_y = cell // grid_size

                cell_x_px = cell_x * grid_cell_size_px + self.stim_img_padding
                cell_y_px = cell_y * grid_cell_size_px + self.stim_img_padding

                if cell == gt_cell:
                    draw.rounded_rectangle([cell_x_px, cell_y_px, 
                                    cell_x_px+grid_cell_size_px, cell_y_px+grid_cell_size_px], 
                                    radius=self.rect_radius, 
                                    fill=self.color_map[self.active_cell_color], 
                                    outline=self.color_map[self.cell_outline_color])
                else:
                    draw.rounded_rectangle([cell_x_px, cell_y_px, 
                                    cell_x_px+grid_cell_size_px, cell_y_px+grid_cell_size_px], 
                                    radius=self.rect_radius, 
                                    fill=self.color_map[self.inactive_cell_color], 
                                    outline=self.color_map[self.cell_outline_color])
            
            memory_stim_list.append(memory_stim)

        return memory_stim_list

    def draw_probe_stim(self):
        probe_stim = Image.new('RGB', self.img_size, color=self.color_map[self.probe_bg_color])
        
        return probe_stim