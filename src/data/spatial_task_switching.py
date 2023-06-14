import os
import json
import random
from tqdm import tqdm
from PIL import Image, ImageDraw

class Spatial_Task_Switching_DataGen:
    '''
    Task: Spatial Task Switching
    '''

    def __init__(self, data_path, variant='Cued', trial_length_options=[30], 
                 gen_random_trials=True, held_out_trial_lengths=[], num_samples=48000, 
                 img_size=224, write=True):
        
        self.data_path = data_path
        self.variant = variant
        self.num_samples = num_samples
        self.img_size = (img_size, img_size)
        self.write = write

        self.bg_color = 'gray'
        self.stim_color = 'black'
        self.padding_color = 'black'
        self.grid_color = 'white'
        self.marker_color = 'red'
        self.cue_line_width = 3
        self.cue_line_padding = 4
        self.grid_line_width = 2
        self.stim_grid_padding = 20
        self.marker_padding = 8
        self.rect_radius = 2
        self.marker_rect_radius = 1

        self.color_map = {'red': '#ee1d23', 'blue': '#015fae', 'violet': '#bc1b8d', 
                          'green': '#3ab54a', 'yellow': 'yellow', 'black': 'black', 
                          'white': 'white', 'brown': '#b78565', 'gray': '#d6d6d6', 
                          'calm_red': '#e84855'}

        self.task_data_path = os.path.join(self.data_path, 'Spatial_Task_Switching_'+self.variant)

        if gen_random_trials:
            self.train_data_dir = os.path.join(self.task_data_path, 'train')
            self.test_data_dir = os.path.join(self.task_data_path, 'test')
            self.gen_test_data_dir = os.path.join(self.task_data_path, 'gen_test')

            self.train_num_samples = num_samples
            self.test_num_samples = int(num_samples * 0.1)
            self.gen_test_num_samples = int(num_samples * 0.1)

            assert self.train_num_samples % len(trial_length_options) == 0
            assert self.test_num_samples % len(trial_length_options) == 0
            if len(held_out_trial_lengths) > 0:
                assert self.gen_test_num_samples % len(held_out_trial_lengths) == 0

            assert all([trial_length > 1 for trial_length in trial_length_options])
            if len(held_out_trial_lengths) > 0:
                assert all([trial_length > 1 for trial_length in held_out_trial_lengths])

            trials = self.gen_random_trials(trial_length_options, held_out_trial_lengths)

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
        
    def gen_random_trials(self, trial_lengths, held_out_trial_lengths):
        trials = {'train': [], 'test': [], 'gen_test': []}

        for split in ['train', 'test', 'gen_test']:
            print('Generating {} trials'.format(split))
            if split == 'train':
                num_samples = self.train_num_samples
                trial_length_options = trial_lengths
            elif split == 'test':
                num_samples = self.test_num_samples
                trial_length_options = trial_lengths
            elif split == 'gen_test':
                num_samples = self.gen_test_num_samples
                trial_length_options = held_out_trial_lengths

            overall_sample_count = 0
            per_trial_length_count = {}

            if len(trial_length_options) > 0:
                num_samples_per_trial_length = num_samples // len(trial_length_options)

            for trial_length in trial_length_options:
                print('Generating {} trials for trial length {}'.format(split, trial_length))
                per_trial_length_count[trial_length] = 0

                while per_trial_length_count[trial_length] < num_samples_per_trial_length:

                    if self.variant == 'Alternate':
                        task_gt = [random.choice(['Cue_Up_Down', 'Cue_Left_Right'])]
                        task_gt.append(task_gt[0][4:])
                        
                        while len(task_gt) < trial_length:
                            sample_task_gt = random.choice(['Up_Down', 'Left_Right'])
                            if sample_task_gt != task_gt[-1]:
                                task_gt.append(sample_task_gt)
                        
                        gt = []
                        for k in range(trial_length):
                            if task_gt[k][:3] == 'Cue':
                                gt.append(2)
                            else:
                                gt.append(random.choice([0, 1]))

                        marker_location = {}
                        for gt_index, gt_value in enumerate(gt):
                            if task_gt[gt_index] == 'Up_Down':
                                if gt_value == 0:
                                    # Up
                                    marker_location[gt_index] = random.choice([0, 1])
                                elif gt_value == 1:
                                    # Down
                                    marker_location[gt_index] = random.choice([2, 3])
                            elif task_gt[gt_index] == 'Left_Right':
                                if gt_value == 0:
                                    # Left
                                    marker_location[gt_index] = random.choice([0, 2])
                                elif gt_value == 1:
                                    # Right
                                    marker_location[gt_index] = random.choice([1, 3])

                    elif self.variant == 'Cued':
                        task_gt = []
                        curr_task_gt = random.choice(['Up_Down', 'Left_Right'])
                        task_gt.append('Cue_{}'.format(curr_task_gt))
                        task_gt.append(curr_task_gt)
                        while len(task_gt) < trial_length:
                            if random.random() < 0.2 and len(task_gt) < trial_length - 1:
                                curr_task_gt = random.choice(['Up_Down', 'Left_Right'])

                                while curr_task_gt == task_gt[-1]:
                                    curr_task_gt = random.choice(['Up_Down', 'Left_Right'])

                                task_gt.append('Cue_{}'.format(curr_task_gt))
                            task_gt.append(curr_task_gt)

                        gt = []
                        for k in range(trial_length):
                            if task_gt[k][:3] == 'Cue':
                                gt.append(2)
                            else:
                                gt.append(random.choice([0, 1]))

                        marker_location = {}
                        for gt_index, gt_value in enumerate(gt):
                            if task_gt[gt_index] == 'Up_Down':
                                if gt_value == 0:
                                    # Up
                                    marker_location[gt_index] = random.choice([0, 1])
                                elif gt_value == 1:
                                    # Down
                                    marker_location[gt_index] = random.choice([2, 3])
                            elif task_gt[gt_index] == 'Left_Right':
                                if gt_value == 0:
                                    # Left
                                    marker_location[gt_index] = random.choice([0, 2])
                                elif gt_value == 1:
                                    # Right
                                    marker_location[gt_index] = random.choice([1, 3])

                    stim_fnames = [split+'_'+str(overall_sample_count).zfill(6)+
                                   '_trial_'+str(i).zfill(2)+'.png' 
                                   for i in range(trial_length)]
                    
                    trials[split].append({'trial_length': trial_length, 
                                          'variant': self.variant, 
                                          'marker_location': marker_location, 
                                          'task_gt': task_gt, 
                                          'gt': gt, 
                                          'trial_type': split, 
                                          'trial_id': split+'_'+str(overall_sample_count).zfill(6), 
                                          'stim_fnames': stim_fnames})
                    
                    overall_sample_count += 1
                    per_trial_length_count[trial_length] += 1

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
                task_gt = trial['task_gt']
                marker_location = trial['marker_location']

                stim_list = self.draw_stim(task_gt, marker_location)

                if self.write:
                    stim_fnames = trial['stim_fnames']

                    for stim, fname in zip(stim_list, stim_fnames):
                        stim.save(os.path.join(write_dir, fname))

                else:
                    trial_stim = {}
                    trial_stim['stim_list'] = stim_list

                    trials_stim_list[split].append(trial_stim)

        if not self.write:
            return trials_stim_list
        
    def draw_stim(self, task_gt, marker_location):
        stim_list = []

        for idx, task in enumerate(task_gt):
            stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
            draw = ImageDraw.Draw(stim)

            grid_size_px = self.img_size[0] - 2*self.stim_grid_padding
            grid_cell_size_px = grid_size_px // 2

            marker_cell_size_px = grid_cell_size_px - 2*self.marker_padding

            for cell_index in range(4):
                cell_x = cell_index % 2
                cell_y = cell_index // 2

                cell_x = cell_x * grid_cell_size_px + self.stim_grid_padding
                cell_y = cell_y * grid_cell_size_px + self.stim_grid_padding

                draw.rounded_rectangle([cell_x, cell_y, 
                                cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                radius=self.rect_radius, 
                                outline=self.color_map[self.padding_color], 
                                fill=self.color_map[self.grid_color], 
                                width=self.grid_line_width)


            if task[:3] == 'Cue':
                if task == 'Cue_Up_Down':
                    draw.line([self.img_size[0]//2, self.cue_line_padding,
                               self.img_size[0]//2, 
                               self.stim_grid_padding-self.cue_line_padding], 
                               fill=self.color_map[self.stim_color], 
                               width=self.cue_line_width)
                    
                    draw.line([self.img_size[0]//2, 
                               self.img_size[1]-self.stim_grid_padding+self.cue_line_padding, 
                               self.img_size[0]//2, self.img_size[1]-self.cue_line_padding], 
                               fill=self.color_map[self.stim_color], 
                               width=self.cue_line_width)
                elif task == 'Cue_Left_Right':
                    draw.line([self.cue_line_padding, self.img_size[1]//2,
                               self.stim_grid_padding-self.cue_line_padding, 
                               self.img_size[1]//2], 
                               fill=self.color_map[self.stim_color], 
                               width=self.cue_line_width)
                    
                    draw.line([self.img_size[0]-self.stim_grid_padding+self.cue_line_padding, 
                               self.img_size[1]//2, 
                               self.img_size[0]-self.cue_line_padding, self.img_size[1]//2], 
                               fill=self.color_map[self.stim_color], 
                               width=self.cue_line_width)

            else:
                marker_x = marker_location[idx] % 2
                marker_y = marker_location[idx] // 2

                marker_x = (marker_x * grid_cell_size_px + self.stim_grid_padding + 
                            self.marker_padding)
                marker_y = (marker_y * grid_cell_size_px + self.stim_grid_padding + 
                            self.marker_padding)

                draw.rounded_rectangle([marker_x, marker_y, 
                              marker_x+marker_cell_size_px, marker_y+marker_cell_size_px], 
                              radius=self.marker_rect_radius,
                              fill=self.color_map[self.marker_color])

            stim_list.append(stim)

        return stim_list


