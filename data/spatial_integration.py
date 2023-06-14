import os
import json
import random
from tqdm import tqdm
from PIL import Image, ImageDraw

class Spatial_Integration_DataGen:
    '''
    Task: Spatial Integration
    '''

    def __init__(self, data_path, grid_size_options, pattern_size_options, 
                 part_size_options, distractor_difference_options, gen_random_trials, 
                 held_out_grid_sizes, held_out_pattern_sizes, held_out_part_sizes, 
                 held_out_distractor_diffs, num_samples, img_size, write):
        
        self.data_path = data_path
        self.num_samples = num_samples
        self.img_size = (img_size, img_size)
        self.write = write

        self.num_fail_trials = 50
        self.bg_color = 'gray'
        self.line_color = 'green'
        self.probe_border_color = 'red'
        self.line_width = 3
        self.border_padding = 5
        self.probe_border_width = 3
        self.stim_img_padding = 10
        self.probe_img_padding = 15

        self.color_map = {'red': '#ee1d23', 'blue': '#015fae', 'violet': '#bc1b8d', 
                          'green': '#3ab54a', 'yellow': 'yellow', 'black': 'black', 
                          'white': 'white', 'brown': '#b78565', 'gray': '#d6d6d6'}

        self.task_data_path = os.path.join(self.data_path, 'Spatial_Integration')

        if gen_random_trials:
            self.train_data_dir = os.path.join(self.task_data_path, 'train')
            self.test_data_dir = os.path.join(self.task_data_path, 'test')
            self.gen_test_data_dir = os.path.join(self.task_data_path, 'gen_test')

            self.train_num_samples = num_samples
            self.test_num_samples = int(num_samples * 0.1)
            self.gen_test_num_samples = int(num_samples * 0.1)

            assert self.train_num_samples % len(part_size_options) == 0
            assert self.test_num_samples % len(part_size_options) == 0
            # assert self.gen_test_num_samples % len(part_size_options) == 0
            
            assert self.num_fail_trials > max(pattern_size_options)
            assert all([pattern_size % part_size == 0 for pattern_size in pattern_size_options 
                        for part_size in part_size_options])

            trials = self.gen_random_trials(grid_size_options, pattern_size_options, 
                                            part_size_options, held_out_part_sizes, 
                                            distractor_difference_options)
            
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

    def gen_random_trials(self, grid_size_options, pattern_size_options, 
                          part_sizes, held_out_part_sizes, distractor_difference_options):
        trials = {'train': [], 'test': [], 'gen_test': []}

        for split in ['train', 'test', 'gen_test']:
            print('Generating {} trials'.format(split))
            if split == 'train':
                num_samples = self.train_num_samples
                part_size_options = part_sizes
            elif split == 'test':
                num_samples = self.test_num_samples
                part_size_options = part_sizes
            elif split == 'gen_test':
                num_samples = self.gen_test_num_samples
                part_size_options = held_out_part_sizes

            overall_sample_count = 0
            per_part_size_sample_count = {}

            if len(part_size_options) > 0:
                num_samples_per_part_size = num_samples // len(part_size_options)

            for part_size in part_size_options:
                print('Generating {} trials for part size {}'.format(split, part_size))
                per_part_size_sample_count[part_size] = 0

                while per_part_size_sample_count[part_size] < num_samples_per_part_size:
                    grid_size = random.choice(grid_size_options)
                    pattern_size = random.choice(pattern_size_options)
                    
                    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                                 (0, 1), (1, -1), (1, 0), (1, 1)]
                    
                    
                    while True:
                        start_x = random.randint(0, grid_size)
                        start_y = random.randint(0, grid_size)

                        gt_line_segment = {0: [start_x, start_y]}

                        count = 0
                        while len(gt_line_segment) <= pattern_size: 
                            direction_x, direction_y = random.choice(neighbors)

                            old_start_x = gt_line_segment[len(gt_line_segment) - 1][0]
                            old_start_y = gt_line_segment[len(gt_line_segment) - 1][1]

                            start_x = old_start_x + direction_x
                            start_y = old_start_y + direction_y

                            if start_x >= 0 and start_x <= grid_size and start_y >= 0 and start_y <= grid_size:
                                if [start_x, start_y] not in list(gt_line_segment.values()):
                                    gt_line_segment[len(gt_line_segment)] = [start_x, start_y]

                            if count == self.num_fail_trials:
                                break
                            count += 1

                        if len(gt_line_segment) == pattern_size+1:
                            break

                    memory_part_line_segment = []
                    for i in range(pattern_size // part_size):
                        line_segment = {}
                        for j in range(part_size+1):
                            line_segment[j] = gt_line_segment[i*part_size + j]
                        memory_part_line_segment.append(line_segment)

                    gt = random.choice([0, 1])

                    if gt == 0:
                        distractor_diff = random.choice(distractor_difference_options)
                        
                        failed_count = 0

                        distractor_line_segment = {k: gt_line_segment[k] 
                                                for k in list(gt_line_segment.keys())[:-distractor_diff]}

                        while len(distractor_line_segment) <= pattern_size:
                            direction_x, direction_y = random.choice(neighbors)

                            old_start_x = distractor_line_segment[len(distractor_line_segment) - 1][0]
                            old_start_y = distractor_line_segment[len(distractor_line_segment) - 1][1]
                            
                            start_x = old_start_x + direction_x
                            start_y = old_start_y + direction_y

                            if start_x >= 0 and start_x <= grid_size and start_y >= 0 and start_y <= grid_size:
                                if ([start_x, start_y] not in list(distractor_line_segment.values()) and 
                                    [start_x, start_y] not in list(gt_line_segment.values())):
                                    distractor_line_segment[len(distractor_line_segment)] = [start_x, start_y]

                            if failed_count == self.num_fail_trials:
                                break
                            failed_count += 1

                        if len(distractor_line_segment) != pattern_size+1:
                            continue

                    elif gt == 1:
                        distractor_diff = None
                        distractor_line_segment = None

                    random.shuffle(memory_part_line_segment)

                    memory_stim_fnames = [split+'_'+str(overall_sample_count).zfill(6)+
                                          '_memory_'+str(idx).zfill(2)+'.png' 
                                          for idx in range(pattern_size // part_size)]
                    
                    probe_stim_fnames = [split+'_'+str(overall_sample_count).zfill(6)+'_probe.png']

                    trials[split].append({'grid_size': grid_size, 
                                          'pattern_size': pattern_size, 
                                          'part_size': part_size, 
                                          'memory_length': pattern_size // part_size, 
                                          'distractor_diff': distractor_diff,
                                          'memory_part_line_segment': memory_part_line_segment, 
                                          'gt': gt, 
                                          'gt_line_segment': gt_line_segment, 
                                          'distractor_line_segment': distractor_line_segment, 
                                          'trial_type': split, 
                                          'trial_id': split+'_'+str(overall_sample_count).zfill(6), 
                                          'memory_stim_fnames': memory_stim_fnames, 
                                          'probe_stim_fnames': probe_stim_fnames})
                    
                    overall_sample_count += 1
                    per_part_size_sample_count[part_size] += 1

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

                gt = trial['gt']
                gt_line_segment = trial['gt_line_segment']
                distractor_line_segment = trial['distractor_line_segment']
                memory_part_line_segment = trial['memory_part_line_segment']

                memory_stim_list = self.draw_memory_stim(grid_size, memory_part_line_segment)
                probe_stim_list = self.draw_probe_stim(grid_size, gt, 
                                                       gt_line_segment, distractor_line_segment)
                
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
        
    def draw_memory_stim(self, grid_size, memory_part_line_segment):
        memory_stim_list = []

        grid_size_px = self.img_size[0] - 2*self.stim_img_padding
        grid_cell_size_px = grid_size_px // grid_size 

        for part_line in memory_part_line_segment:
            memory_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
            draw = ImageDraw.Draw(memory_stim)

            for line_idx in list(part_line.keys())[:-1]:
                x_px = part_line[line_idx][1]*grid_cell_size_px + self.stim_img_padding
                y_px = part_line[line_idx][0]*grid_cell_size_px + self.stim_img_padding

                next_x_px = part_line[line_idx+1][1]*grid_cell_size_px + self.stim_img_padding
                next_y_px = part_line[line_idx+1][0]*grid_cell_size_px + self.stim_img_padding

                draw.line((x_px, y_px, next_x_px, next_y_px), fill=self.color_map[self.line_color], 
                          width=self.line_width)
                
            memory_stim_list.append(memory_stim)

        return memory_stim_list
    
    def draw_probe_stim(self, grid_size, gt, gt_line_segment, distractor_line_segment):
        probe_stim_list = []

        grid_size_px = self.img_size[0] - 2*self.probe_img_padding
        grid_cell_size_px = grid_size_px // grid_size 

        probe_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
        draw = ImageDraw.Draw(probe_stim)

        # draw.line((self.stim_img_padding, self.img_size[1]-20, 
        #            self.img_size[0]-self.stim_img_padding, self.img_size[1]-20), 
        #           fill=self.color_map[self.probe_border_color], 
        #           width=self.probe_border_width)

        draw.rectangle((self.border_padding, self.border_padding, 
                        self.img_size[0]-self.border_padding, self.img_size[1]-self.border_padding), 
                        outline=self.color_map[self.probe_border_color], width=self.probe_border_width)
                       
        if gt == 0:
            line_segment = distractor_line_segment
        elif gt == 1:
            line_segment = gt_line_segment

        for line_idx in list(line_segment.keys())[:-1]:
            x_px = line_segment[line_idx][1]*grid_cell_size_px + self.probe_img_padding
            y_px = line_segment[line_idx][0]*grid_cell_size_px + self.probe_img_padding

            next_x_px = line_segment[line_idx+1][1]*grid_cell_size_px + self.probe_img_padding
            next_y_px = line_segment[line_idx+1][0]*grid_cell_size_px + self.probe_img_padding

            draw.line((x_px, y_px, next_x_px, next_y_px), fill=self.color_map[self.line_color], 
                        width=self.line_width)

        probe_stim_list.append(probe_stim)

        return probe_stim_list