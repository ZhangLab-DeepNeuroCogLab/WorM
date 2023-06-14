import os
import json
import math
import random
from tqdm import tqdm
from PIL import Image, ImageDraw

class Visual_Serial_Recall_Recognition_DataGen:
    '''
    Task: Visual Serial Recall and Recognition
    '''

    def __init__(self, data_path, grid_size, probe_variant, list_length_options, 
                 distractor_difference_options, gen_random_trials, 
                 held_out_list_lengths, held_out_distractor_diffs,
                 num_samples, img_size, write):
        
        self.data_path = data_path
        self.grid_size = grid_size
        self.num_samples = num_samples
        self.img_size = (img_size, img_size)
        self.probe_variant = probe_variant
        self.write = write

        self.bg_color = 'gray'
        self.active_cell_color = 'green'
        self.inactive_cell_color = 'white'
        self.cell_outline_color = 'black'
        self.grid_outline_color = 'black'
        self.memory_stim_padding = 9
        self.probe_stim_padding = 2
        self.probe_recognition_stim_padding = 4
        self.mem_rect_radius = 3
        self.probe_rect_radius = 1
        self.probe_recall_rect_radius = 0

        self.color_map = {'red': '#ee1d23', 'blue': '#015fae', 'violet': '#bc1b8d', 
                          'green': '#3ab54a', 'yellow': 'yellow', 'black': 'black', 
                          'white': 'white', 'brown': '#b78565', 'gray': '#d6d6d6'}

        self.task_data_path = os.path.join(self.data_path, 'Visual_Serial_'+self.probe_variant)

        if gen_random_trials:
            self.train_data_dir = os.path.join(self.task_data_path, 'train')
            self.test_data_dir = os.path.join(self.task_data_path, 'test')
            self.gen_test_data_dir = os.path.join(self.task_data_path, 'gen_test')

            self.train_num_samples = num_samples
            self.test_num_samples = int(num_samples * 0.1)
            self.gen_test_num_samples = int(num_samples * 0.1)

            assert self.train_num_samples % len(list_length_options) == 0
            assert self.test_num_samples % len(list_length_options) == 0
            if len(held_out_list_lengths) > 0:
                assert self.gen_test_num_samples % len(held_out_list_lengths) == 0

            max_list_length = max(list_length_options)
            # max_list_length should be even or a perfect square for drawing probe grid
            assert ((max_list_length % 2 == 0 or max_list_length==int(math.sqrt(max_list_length))**2) 
                    and (max_list_length >= 4))

            if len(held_out_list_lengths) > 0:
                max_held_out_list_length = max(held_out_list_lengths)
                assert ((max_held_out_list_length % 2 == 0 or max_held_out_list_length==int(math.sqrt(max_held_out_list_length))**2) 
                        and (max_held_out_list_length >= 4))

            assert all([distractor_diff % 2 == 0 for distractor_diff in distractor_difference_options])

            trials = self.gen_random_trials(list_length_options, 
                                            held_out_list_lengths, 
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
        
    def gen_random_trials(self, list_lengths, 
                          held_out_list_lengths, distractor_diff_options):
        
        trials = {'train': [], 'test': [], 'gen_test': []}

        for split in ['train', 'test', 'gen_test']:
            print('Generating {} trials'.format(split))
            if split == 'train':
                num_samples = self.train_num_samples
                list_length_options = list_lengths
            elif split == 'test':
                num_samples = self.test_num_samples
                list_length_options = list_lengths
            elif split == 'gen_test':
                num_samples = self.gen_test_num_samples
                list_length_options = held_out_list_lengths

            if len(list_length_options) > 0:
                max_list_length = max(list_length_options)
                num_samples_per_list_length = num_samples // len(list_length_options)

            overall_sample_count = 0
            list_length_sample_count = {}

            for list_length in list_length_options:
                print('Generating {} trials of list length {}'.format(split, list_length))
                list_length_sample_count[list_length] = 0
                
                while list_length_sample_count[list_length] < num_samples_per_list_length:
                    memory_items = {}
                    distractor_items = {}

                    recall_gt = None
                    recognition_gt = None
                    distractor_diff = None

                    for item_index in range(list_length):
                        dark_cells = random.sample(range(self.grid_size**2), self.grid_size**2//2)
                        memory_items[item_index] = dark_cells

                    if len(set([tuple(sorted(item)) for item in memory_items.values()])) != list_length:
                        continue

                    if self.probe_variant == 'Recall':
                        recall_gt = random.sample(range(max_list_length), list_length)
                        probe_stim_fnames = [split+'_'+str(overall_sample_count).zfill(6)+'_probe.png']
                    elif self.probe_variant == 'Recognition':
                        recognition_gt = []
                        probe_stim_fnames = []
                        distractor_diff = random.choice(distractor_diff_options)
                        for item_index in range(list_length):
                            memory_dark_cells = memory_items[item_index]
                            memory_white_cells = [cell for cell in range(self.grid_size**2)
                                                    if cell not in memory_dark_cells]
                            
                            distractor_dark_cells = random.sample(memory_dark_cells, 
                                                                  len(memory_dark_cells)-distractor_diff//2)
                            sample_white_memory_cells = random.sample(memory_white_cells, 
                                                                      distractor_diff//2)

                            distractor_items[item_index] = distractor_dark_cells + sample_white_memory_cells

                            recognition_gt.append(random.choice([0, 1]))
                            probe_stim_fnames.append(split+'_'+str(overall_sample_count).zfill(6)+
                                                     '_probe_'+str(item_index).zfill(2)+'.png')
                                                 
                    
                    memory_stim_fnames = [split+'_'+str(overall_sample_count).zfill(6)+
                                          '_memory_'+str(item_index).zfill(2)+'.png' 
                                          for item_index in range(list_length)]
                    
                    trials[split].append({'grid_size': self.grid_size, 
                                          'list_length': list_length, 
                                          'max_list_length': max_list_length, 
                                          'probe_variant': self.probe_variant, 
                                          'memory_items': memory_items, 
                                          'distractor_diff': distractor_diff,
                                          'distractor_items': distractor_items,
                                          'recall_gt': recall_gt, 
                                          'recognition_gt': recognition_gt,
                                          'trial_type': split, 
                                          'trial_id': split+'_'+str(overall_sample_count).zfill(6), 
                                          'memory_stim_fnames': memory_stim_fnames, 
                                          'probe_stim_fnames': probe_stim_fnames})

                    overall_sample_count += 1
                    list_length_sample_count[list_length] += 1

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
                max_list_length = trial['max_list_length']
                memory_items = trial['memory_items']

                memory_stim_list = self.draw_memory_stim(grid_size, memory_items)

                if self.probe_variant == 'Recall':
                    recall_gt = trial['recall_gt']
                    probe_stim_list = self.draw_recall_probe_stim(grid_size, memory_items, 
                                                                  recall_gt, max_list_length)
                elif self.probe_variant == 'Recognition':
                    recognition_gt = trial['recognition_gt']
                    distractor_items = trial['distractor_items']
                    probe_stim_list = self.draw_recognition_probe_stim(grid_size, memory_items, 
                                                                       distractor_items, recognition_gt)
                
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
        
    def draw_memory_stim(self, grid_size, memory_items):
        memory_stim_list = []

        for item_index in range(len(memory_items)):
            memory_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
            draw = ImageDraw.Draw(memory_stim)

            grid_size_px = self.img_size[0] - 2*self.memory_stim_padding
            grid_cell_size_px = grid_size_px // grid_size

            for cell_index in range(grid_size**2):
                cell_x = cell_index % grid_size
                cell_y = cell_index // grid_size

                cell_x = cell_x * grid_cell_size_px + self.memory_stim_padding
                cell_y = cell_y * grid_cell_size_px + self.memory_stim_padding

                if cell_index in memory_items[item_index]:
                    draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                   radius=self.mem_rect_radius, fill=self.color_map[self.active_cell_color], outline=self.color_map[self.cell_outline_color])
                else:
                    draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                   radius=self.mem_rect_radius, fill=self.color_map[self.inactive_cell_color], outline=self.color_map[self.cell_outline_color])

            memory_stim_list.append(memory_stim)

        return memory_stim_list
    
    def draw_recall_probe_stim(self, grid_size, memory_items, recall_gt, max_list_length):
        probe_stim_list = []

        probe_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
        draw = ImageDraw.Draw(probe_stim)

        if max_list_length == int(math.sqrt(max_list_length))**2:
            num_rows = int(math.sqrt(max_list_length))
            num_cols = int(math.sqrt(max_list_length))

            for row in range(1, num_rows):
                draw.line([0, row*self.img_size[0]//num_rows, 
                           self.img_size[0], row*self.img_size[0]//num_rows], 
                           fill=self.color_map[self.grid_outline_color])

            for col in range(1, num_cols):
                draw.line([col*self.img_size[1]//num_cols, 0, 
                           col*self.img_size[1]//num_cols, self.img_size[1]], 
                           fill=self.color_map[self.grid_outline_color])
        else:
            num_rows = 2
            num_cols = max_list_length // 2
            
            draw.line([0, self.img_size[1]//2, 
                       self.img_size[0], self.img_size[1]//2], 
                       fill=self.color_map[self.grid_outline_color])
            
            for col in range(1, num_cols):
                draw.line([col*self.img_size[0]//num_cols, 0, 
                           col*self.img_size[0]//num_cols, self.img_size[1]], 
                           fill=self.color_map[self.grid_outline_color])

        big_grid_cell_size_x_px = self.img_size[0] // num_cols
        big_grid_cell_size_y_px = self.img_size[1] // num_rows

        probe_stim_x_padding = self.probe_stim_padding

        grid_size_px = (self.img_size[0]-(2*num_cols*probe_stim_x_padding)) // num_cols
        grid_cell_size_px = grid_size_px // grid_size

        probe_stim_y_padding = (big_grid_cell_size_y_px - grid_size_px) // 2

        for item_index, item_location in enumerate(recall_gt):
            row = item_location // num_cols
            col = item_location % num_cols

            for cell_index in range(grid_size**2):
                cell_x = cell_index % grid_size
                cell_y = cell_index // grid_size

                cell_x = (col*big_grid_cell_size_x_px + 
                          cell_x*grid_cell_size_px + probe_stim_x_padding)
                cell_y = (row*big_grid_cell_size_y_px + 
                          cell_y*grid_cell_size_px + probe_stim_y_padding)

                if cell_index in memory_items[item_index]:
                    draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                   radius=self.probe_recall_rect_radius, fill=self.color_map[self.active_cell_color], outline=self.color_map[self.cell_outline_color])
                else:
                    draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                   radius=self.probe_recall_rect_radius, fill=self.color_map[self.inactive_cell_color], outline=self.color_map[self.cell_outline_color])
                
        probe_stim_list.append(probe_stim)

        return probe_stim_list
    
    def draw_recognition_probe_stim(self, grid_size, memory_items, distractor_items, recognition_gt):
        probe_stim_list = []

        probe_stim_x_padding = self.probe_recognition_stim_padding

        for index, gt in enumerate(recognition_gt):
            memory_item = memory_items[index]
            distractor_item = distractor_items[index]

            probe_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
            draw = ImageDraw.Draw(probe_stim)

            draw.line([(self.img_size[0]//2)-2, 0+20, 
                          (self.img_size[0]//2)-2, self.img_size[1]-20], 
                            fill='black')

            big_grid_size_x_px = self.img_size[0] // 2

            grid_size_px = (self.img_size[0] - 4*probe_stim_x_padding) // 2
            grid_cell_size_px = grid_size_px // grid_size

            probe_stim_y_padding = (self.img_size[1] - grid_size_px) // 2

            for i in range(2):
                if i == gt:
                    item = memory_item
                else:
                    item = distractor_item

                for cell_index in range(grid_size**2):
                    cell_x = cell_index % grid_size
                    cell_y = cell_index // grid_size

                    cell_x = (i*big_grid_size_x_px + 
                              cell_x*grid_cell_size_px + probe_stim_x_padding)
                    cell_y = (cell_y*grid_cell_size_px + probe_stim_y_padding)

                    if cell_index in item:
                        draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                       radius=self.probe_rect_radius, fill=self.color_map[self.active_cell_color], outline=self.color_map[self.cell_outline_color])
                    else:
                        draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                       radius=self.probe_rect_radius, fill=self.color_map[self.inactive_cell_color], outline=self.color_map[self.cell_outline_color])
                        
            probe_stim_list.append(probe_stim)

        return probe_stim_list