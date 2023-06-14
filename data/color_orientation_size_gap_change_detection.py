import os
import json
import random
from tqdm import tqdm
from PIL import Image, ImageDraw

class Color_Orientation_Size_Gap_Conjunction_Change_Detection_DataGen:
    '''
    Task: Color_Orientation_Size_Gap_Conjunction_Change_Detection
    '''

    def __init__(self, data_path, variant, set_size_options, ri_options,
                 gen_random_trials, held_out_set_sizes, held_out_ri_options,
                 num_samples, img_size, write):

        self.data_path = data_path
        self.num_samples = num_samples
        self.img_size = (img_size, img_size)
        self.write = write

        self.variant = variant

        self.max_tries = 100
        self.img_edge_padding = 3
        self.bar_padding = 11
        self.bar_width = 3

        self.small_bar_size = 3 * 3
        self.large_bar_size = 5 * 3
        self.gap_size = self.large_bar_size / 5
        self.bar_sizes = ['small', 'large']
        self.bar_colors = ['red', 'green']
        self.bar_orientations = ['horizontal', 'vertical']
        self.bar_gaps = ['continuous', 'broken']

        self.bg_color = 'gray'
        self.color_map = {'red': '#ee1d23', 'blue': '#015fae', 'violet': '#bc1b8d', 
                          'green': '#3ab54a', 'yellow': 'yellow', 'black': 'black', 
                          'white': 'white', 'brown': '#b78565', 'gray': '#d6d6d6'}

        self.task_data_path = os.path.join(self.data_path, 
            'Color_Orientation_Size_Gap_Conjunction_Change_Detection_' + self.variant)

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

            trials = self.gen_random_trials(set_size_options, held_out_set_sizes, 
                                            ri_options, held_out_ri_options)

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
        
    def gen_random_trials(self, set_sizes, held_out_set_sizes, 
                          ris, held_out_ri_options):
        trials = {'train': [], 'test': [], 'gen_test': []}

        for split in ['train', 'test']:
            print('Generating {} trials'.format(split))
            if split == 'train':
                num_samples = self.train_num_samples
                set_size_options = set_sizes
                ri_options = ris
            elif split == 'test':
                num_samples = self.test_num_samples
                set_size_options = set_sizes
                ri_options = ris
            elif split == 'gen_test':
                num_samples = self.gen_test_num_samples
                set_size_options = held_out_set_sizes
                ri_options = held_out_ri_options

            overall_sample_count = 0
            per_set_size_sample_count = {}

            if len(set_size_options) > 0:
                num_samples_per_set_size = num_samples // len(set_size_options)

            for set_size in set_size_options:
                print('Generating {} trials for set size {}'.format(split, set_size))
                per_set_size_sample_count[set_size] = 0

                while per_set_size_sample_count[set_size] < num_samples_per_set_size:
                    
                    while True:
                        memory_array = {}

                        trial_counter = 0
                        while len(list(memory_array.keys())) < set_size:
                            sample_size = random.choice(self.bar_sizes)
                            sample_color = random.choice(self.bar_colors)
                            sample_orientation = random.choice(self.bar_orientations)
                            sample_gap = random.choice(self.bar_gaps)

                            if sample_size == 'small':
                                bar_size = self.small_bar_size
                            elif sample_size == 'large':
                                bar_size = self.large_bar_size

                            x1 = random.randint(0 + (bar_size//2) + self.img_edge_padding, 
                                                self.img_size[0] - bar_size - self.img_edge_padding)
                            y1 = random.randint(0 + (bar_size//2) + self.img_edge_padding, 
                                                self.img_size[1] - bar_size - self.img_edge_padding)

                            if sample_orientation == 'horizontal':
                                x2 = x1 + bar_size
                                y2 = y1
                            elif sample_orientation == 'vertical':
                                x2 = x1
                                y2 = y1 + bar_size

                            if sample_gap == 'broken':
                                if sample_size == 'large':
                                    gap_x1 = x1 + (x2-x1)*(2/5)
                                    gap_x2 = x1 + (x2-x1)*(3/5)
                                    gap_y1 = y1 + (y2-y1)*(2/5)
                                    gap_y2 = y1 + (y2-y1)*(3/5)
                                elif sample_size == 'small':
                                    gap_x1 = x1 + (x2-x1)*(1/3)
                                    gap_x2 = x1 + (x2-x1)*(2/3)
                                    gap_y1 = y1 + (y2-y1)*(1/3)
                                    gap_y2 = y1 + (y2-y1)*(2/3)
                                gap_coords = (gap_x1, gap_y1, gap_x2, gap_y2)
                            else:
                                gap_coords = None
                            
                            overlap = False
                            for cell in memory_array.values():
                                bbox = cell['bar_coords']
                                if ((abs(x1-bbox[0]) < bar_size + self.bar_padding) and 
                                    (abs(y1-bbox[1]) < bar_size + self.bar_padding)):
                                    overlap = True
                                    break

                            if not overlap:
                                memory_array[len(memory_array.keys())+1] = {'bar_coords': (x1, y1, x2, y2), 
                                                                                'size': sample_size, 
                                                                                'color': sample_color, 
                                                                                'orientation': sample_orientation, 
                                                                                'gap': sample_gap, 
                                                                                'gap_coords': gap_coords}

                            trial_counter += 1
                            if trial_counter > self.max_tries:
                                break
                        
                        if len(list(memory_array.keys())) == set_size:
                            break
                    

                    ri = random.choice(ri_options)
                    gt = random.choice([0, 1])
                    conjunction_gt = None

                    if gt == 0:
                        probe_array = {}

                        for cell_idx, cell in memory_array.items():
                            probe_array[cell_idx] = cell

                    elif gt == 1:
                        if self.variant == 'Color':
                            probe_array = {}

                            for cell_idx, cell in memory_array.items():
                                probe_array[cell_idx] = cell.copy()

                            bar_change_index = random.choice(list(probe_array.keys()))
                            color_choice = random.choice(self.bar_colors)
                            while color_choice == probe_array[bar_change_index]['color']:
                                color_choice = random.choice(self.bar_colors)

                            probe_array[bar_change_index]['color'] = color_choice
                        
                        elif self.variant == 'Orientation':
                            probe_array = {}

                            for cell_idx, cell in memory_array.items():
                                probe_array[cell_idx] = cell.copy()

                            bar_change_index = random.choice(list(probe_array.keys()))
                            orientation_choice = random.choice(self.bar_orientations)
                            while orientation_choice == probe_array[bar_change_index]['orientation']:
                                orientation_choice = random.choice(self.bar_orientations)

                            x1, y1, x2, y2 = probe_array[bar_change_index]['bar_coords']

                            if probe_array[bar_change_index]['gap'] == 'broken':
                                gap_x1, gap_y1, gap_x2, gap_y2 = probe_array[bar_change_index]['gap_coords']
                            elif probe_array[bar_change_index]['gap'] == 'continuous':
                                mod_gap_coords = None

                            if probe_array[bar_change_index]['size'] == 'small':
                                bar_size = self.small_bar_size
                            elif probe_array[bar_change_index]['size'] == 'large':
                                bar_size = self.large_bar_size

                            if orientation_choice == 'horizontal':
                                mod_bar_coords = (x1-(bar_size/2), y1+(bar_size/2), x2+(bar_size/2), y2-(bar_size/2)) 
                                if probe_array[bar_change_index]['gap'] == 'broken':
                                    mod_gap_coords = (gap_x1-(self.gap_size/2), gap_y1+(self.gap_size/2), 
                                                      gap_x2+(self.gap_size/2), gap_y2-(self.gap_size/2))
                            elif orientation_choice == 'vertical':
                                mod_bar_coords = (x1+(bar_size/2), y1-(bar_size/2), x2-(bar_size/2), y2+(bar_size/2))
                                if probe_array[bar_change_index]['gap'] == 'broken':
                                    mod_gap_coords = (gap_x1+(self.gap_size/2), gap_y1-(self.gap_size/2), 
                                                      gap_x2-(self.gap_size/2), gap_y2+(self.gap_size/2))
                                    
                            probe_array[bar_change_index]['orientation'] = orientation_choice
                            probe_array[bar_change_index]['bar_coords'] = mod_bar_coords
                            probe_array[bar_change_index]['gap_coords'] = mod_gap_coords

                        elif self.variant == 'Size':
                            probe_array = {}

                            for cell_idx, cell in memory_array.items():
                                probe_array[cell_idx] = cell.copy()

                            bar_change_index = random.choice(list(probe_array.keys()))
                            size_choice = random.choice(self.bar_sizes)
                            while size_choice == probe_array[bar_change_index]['size']:
                                size_choice = random.choice(self.bar_sizes)

                            x1, y1, x2, y2 = probe_array[bar_change_index]['bar_coords']
                            orientation = probe_array[bar_change_index]['orientation']

                            if size_choice == 'small':
                                bar_size = self.small_bar_size
                                if orientation == 'horizontal':
                                    mod_bar_coords = (x1+(bar_size/4), y1, x2-(bar_size/4), y2)
                                elif orientation == 'vertical':
                                    mod_bar_coords = (x1, y1+(bar_size/4), x2, y2-(bar_size/4))
                            elif size_choice == 'large':
                                bar_size = self.large_bar_size
                                if orientation == 'horizontal':
                                    mod_bar_coords = (x1-(bar_size/4), y1, x2+(bar_size/4), y2)
                                elif orientation == 'vertical':
                                    mod_bar_coords = (x1, y1-(bar_size/4), x2, y2+(bar_size/4))

                            probe_array[bar_change_index]['size'] = size_choice
                            probe_array[bar_change_index]['bar_coords'] = mod_bar_coords

                        elif self.variant == 'Gap':
                            probe_array = {}

                            for cell_idx, cell in memory_array.items():
                                probe_array[cell_idx] = cell.copy()

                            bar_change_index = random.choice(list(probe_array.keys()))
                            gap_choice = random.choice(self.bar_gaps)
                            while gap_choice == probe_array[bar_change_index]['gap']:
                                gap_choice = random.choice(self.bar_gaps)

                            x1, y1, x2, y2 = probe_array[bar_change_index]['bar_coords']

                            if gap_choice == 'broken':
                                if probe_array[bar_change_index]['size'] == 'large':
                                    gap_x1 = x1 + (x2-x1)*(2/5)
                                    gap_x2 = x1 + (x2-x1)*(3/5)
                                    gap_y1 = y1 + (y2-y1)*(2/5)
                                    gap_y2 = y1 + (y2-y1)*(3/5)
                                elif probe_array[bar_change_index]['size'] == 'small':
                                    gap_x1 = x1 + (x2-x1)*(1/3)
                                    gap_x2 = x1 + (x2-x1)*(2/3)
                                    gap_y1 = y1 + (y2-y1)*(1/3)
                                    gap_y2 = y1 + (y2-y1)*(2/3)
                                mod_gap_coords = (gap_x1, gap_y1, gap_x2, gap_y2)
                            elif gap_choice == 'continuous':
                                mod_gap_coords = None

                            probe_array[bar_change_index]['gap'] = gap_choice
                            probe_array[bar_change_index]['gap_coords'] = mod_gap_coords

                        elif self.variant == 'Conjunction':
                            conjunction_gt = random.choice(['Color', 'Orientation', 
                                                            'Size', 'Gap'])

                            probe_array = {}

                            for cell_idx, cell in memory_array.items():
                                probe_array[cell_idx] = cell.copy()

                            if conjunction_gt == 'Color':
                                bar_change_index = random.choice(list(probe_array.keys()))
                                color_choice = random.choice(self.bar_colors)
                                while color_choice == probe_array[bar_change_index]['color']:
                                    color_choice = random.choice(self.bar_colors)

                                probe_array[bar_change_index]['color'] = color_choice

                            elif conjunction_gt == 'Orientation':
                                bar_change_index = random.choice(list(probe_array.keys()))
                                orientation_choice = random.choice(self.bar_orientations)
                                while orientation_choice == probe_array[bar_change_index]['orientation']:
                                    orientation_choice = random.choice(self.bar_orientations)
                                
                                x1, y1, x2, y2 = probe_array[bar_change_index]['bar_coords']

                                if probe_array[bar_change_index]['gap'] == 'broken':
                                    gap_x1, gap_y1, gap_x2, gap_y2 = probe_array[bar_change_index]['gap_coords']
                                elif probe_array[bar_change_index]['gap'] == 'continuous':
                                    mod_gap_coords = None

                                if probe_array[bar_change_index]['size'] == 'small':
                                    bar_size = self.small_bar_size
                                elif probe_array[bar_change_index]['size'] == 'large':
                                    bar_size = self.large_bar_size

                                if orientation_choice == 'horizontal':
                                    mod_bar_coords = (x1-(bar_size/2), y1+(bar_size/2), 
                                                      x2+(bar_size/2), y2-(bar_size/2))
                                    if probe_array[bar_change_index]['gap'] == 'broken':
                                        mod_gap_coords = (gap_x1-(self.gap_size/2), gap_y1+(self.gap_size/2), 
                                                          gap_x2+(self.gap_size/2), gap_y2-(self.gap_size/2))
                                elif orientation_choice == 'vertical':
                                    mod_bar_coords = (x1+(bar_size/2), y1-(bar_size/2), 
                                                      x2-(bar_size/2), y2+(bar_size/2))
                                    if probe_array[bar_change_index]['gap'] == 'broken':
                                        mod_gap_coords = (gap_x1+(self.gap_size/2), gap_y1-(self.gap_size/2), 
                                                          gap_x2-(self.gap_size/2), gap_y2+(self.gap_size/2))
                                        
                                probe_array[bar_change_index]['orientation'] = orientation_choice
                                probe_array[bar_change_index]['bar_coords'] = mod_bar_coords
                                probe_array[bar_change_index]['gap_coords'] = mod_gap_coords

                            elif conjunction_gt == 'Size':
                                bar_change_index = random.choice(list(probe_array.keys()))
                                size_choice = random.choice(self.bar_sizes)
                                while size_choice == probe_array[bar_change_index]['size']:
                                    size_choice = random.choice(self.bar_sizes)

                                x1, y1, x2, y2 = probe_array[bar_change_index]['bar_coords']
                                orientation = probe_array[bar_change_index]['orientation']

                                if size_choice == 'small':
                                    bar_size = self.small_bar_size
                                    if orientation == 'horizontal':
                                        mod_bar_coords = (x1+(bar_size/4), y1, x2-(bar_size/4), y2)
                                    elif orientation == 'vertical':
                                        mod_bar_coords = (x1, y1+(bar_size/4), x2, y2-(bar_size/4))
                                elif size_choice == 'large':
                                    bar_size = self.large_bar_size
                                    if orientation == 'horizontal':
                                        mod_bar_coords = (x1-(bar_size/4), y1, x2+(bar_size/4), y2)
                                    elif orientation == 'vertical':
                                        mod_bar_coords = (x1, y1-(bar_size/4), x2, y2+(bar_size/4))

                                probe_array[bar_change_index]['size'] = size_choice
                                probe_array[bar_change_index]['bar_coords'] = mod_bar_coords

                            elif conjunction_gt == 'Gap':
                                bar_change_index = random.choice(list(probe_array.keys()))
                                gap_choice = random.choice(self.bar_gaps)
                                while gap_choice == probe_array[bar_change_index]['gap']:
                                    gap_choice = random.choice(self.bar_gaps)

                                x1, y1, x2, y2 = probe_array[bar_change_index]['bar_coords']

                                if gap_choice == 'broken':
                                    if probe_array[bar_change_index]['size'] == 'large':
                                        gap_x1 = x1 + (x2-x1)*(2/5)
                                        gap_x2 = x1 + (x2-x1)*(3/5)
                                        gap_y1 = y1 + (y2-y1)*(2/5)
                                        gap_y2 = y1 + (y2-y1)*(3/5)
                                    elif probe_array[bar_change_index]['size'] == 'small':
                                        gap_x1 = x1 + (x2-x1)*(1/3)
                                        gap_x2 = x1 + (x2-x1)*(2/3)
                                        gap_y1 = y1 + (y2-y1)*(1/3)
                                        gap_y2 = y1 + (y2-y1)*(2/3)
                                    mod_gap_coords = (gap_x1, gap_y1, gap_x2, gap_y2)
                                elif gap_choice == 'continuous':
                                    mod_gap_coords = None    

                                probe_array[bar_change_index]['gap'] = gap_choice
                                probe_array[bar_change_index]['gap_coords'] = mod_gap_coords
                        
                    
                    memory_stim_fnames = [split+'_'+str(overall_sample_count).zfill(6)+
                                          '_memory.png']
                    
                    probe_stim_fnames = [split+'_'+str(overall_sample_count).zfill(6)+
                                         '_probe.png']

                    trials[split].append({'set_size': set_size, 
                                          'memory_array': memory_array, 
                                          'probe_array': probe_array, 
                                          'retention_interval': ri, 
                                          'conjunction_gt': conjunction_gt, 
                                          'gt': gt, 
                                          'trial_type': split, 
                                          'trial_id': split+'_'+str(overall_sample_count).zfill(6), 
                                          'memory_stim_fnames': memory_stim_fnames, 
                                          'probe_stim_fnames': probe_stim_fnames})
                    
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
                memory_array = trial['memory_array']
                probe_array = trial['probe_array']

                memory_stim_list = self.draw_memory_stim(memory_array)
                probe_stim_list = self.draw_probe_stim(probe_array)

                if self.write:
                    memory_stim_fnames = trial['memory_stim_fnames']
                    probe_stim_fnames = trial['probe_stim_fnames']

                    for memory_stim, memory_stim_fname in zip(memory_stim_list, memory_stim_fnames):
                        memory_stim.save(os.path.join(write_dir, memory_stim_fname))
                    
                    for probe_stim, probe_stim_fname in zip(probe_stim_list, probe_stim_fnames):
                        probe_stim.save(os.path.join(write_dir, probe_stim_fname))

                else:
                    trials_stim = {}
                    trials_stim['memory_stim_list'] = memory_stim_list
                    trials_stim['probe_stim_list'] = probe_stim_list

                    trials_stim_list[split].append(trials_stim)

        if self.write:
            blank_stim = self.draw_blank_stim()
            blank_stim.save(os.path.join(self.task_data_path, 'blank.png'))
        else:
            return trials_stim_list
        
    def draw_blank_stim(self):
        blank_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
        return blank_stim

    def draw_memory_stim(self, memory_array):
        memory_stim_list = []

        memory_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
        draw = ImageDraw.Draw(memory_stim)

        for cell in memory_array.values():
            coords = cell['bar_coords']
            draw.line(coords, fill=self.color_map[cell['color']], width=self.bar_width)
            if cell['gap'] == 'broken':
                gap_coords = cell['gap_coords']
                draw.line(gap_coords, fill=self.color_map['black'], width=self.bar_width)
        
        memory_stim_list.append(memory_stim)

        return memory_stim_list

    def draw_probe_stim(self, probe_array):
        probe_stim_list = []

        probe_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
        draw = ImageDraw.Draw(probe_stim)

        for cell in probe_array.values():
            coords = cell['bar_coords']
            draw.line(coords, fill=self.color_map[cell['color']], width=self.bar_width)
            if cell['gap'] == 'broken':
                gap_coords = cell['gap_coords']
                draw.line(gap_coords, fill=self.color_map['black'], width=self.bar_width)

        probe_stim_list.append(probe_stim)

        return probe_stim_list