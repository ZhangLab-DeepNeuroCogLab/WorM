import os
import cv2
import json
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

class Spatial_Memory_Updating_DataGen:
    '''
    Task: Spatial Memory Updating
    '''

    def __init__(self, data_path, grid_size, set_size_options, 
                 presentation_time_options, num_updates_options, 
                 gen_random_trials, held_out_set_sizes, held_out_num_updates, 
                 num_samples, img_size, write):

        self.data_path = data_path
        self.grid_size = grid_size
        self.num_samples = num_samples
        self.img_size = (img_size, img_size)
        self.write = write

        self.box_draw_radius = 35
        self.box_size = 15 # must be divisible by 10
        self.marker_radius = 1.5
        self.arrow_width = 1
        self.arrow_tip_length = 0.4
        self.outline_width = 1
        self.bg_color = 'gray' # 'gray'
        self.box_color = 'green' # 'white'
        self.probe_color = 'red' # 'black'
        self.outline_color = 'white'
        self.marker_color = 'red' # 'black'
        self.arrow_color = (30, 30, 30) # black

        self.color_map = {'red': '#ee1d23', 'blue': '#015fae', 'violet': '#bc1b8d', 
                          'green': '#3ab54a', 'yellow': 'yellow', 'black': 'black', 
                          'white': 'white', 'brown': '#b78565', 'gray': '#d6d6d6'}

        self.task_data_path = os.path.join(self.data_path, 'Spatial_Memory_Updating')

        if gen_random_trials:
            self.train_data_dir = os.path.join(self.task_data_path, 'train')
            self.test_data_dir = os.path.join(self.task_data_path, 'test')
            self.gen_test_data_dir = os.path.join(self.task_data_path, 'gen_test')

            self.train_num_samples = num_samples
            self.test_num_samples = int(num_samples*0.1)
            self.gen_test_num_samples = int(num_samples*0.1)

            assert self.train_num_samples % (len(set_size_options)*len(presentation_time_options)) == 0
            assert self.test_num_samples % (len(set_size_options)*len(presentation_time_options)) == 0
            if len(held_out_set_sizes) > 0:
                assert self.gen_test_num_samples % (len(held_out_set_sizes)*len(presentation_time_options)) == 0

            trials = self.gen_random_trials(set_size_options, held_out_set_sizes, 
                                            num_updates_options, presentation_time_options)
            
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
                          num_updates_options, presentation_time_options):
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
            per_condition_sample_count = {}

            if len(set_size_options) > 0 and len(presentation_time_options) > 0:
                num_samples_per_condition = num_samples // (len(set_size_options)*len(presentation_time_options))
        
            for set_size in set_size_options:
                for presentation_time in tqdm(presentation_time_options):
                    print('Generating {} trials for set size {}, presentation time {}'.format(split, set_size, 
                                                                                              presentation_time))
                    per_condition_sample_count[(set_size, presentation_time)] = 0
                    
                    while per_condition_sample_count[(set_size, presentation_time)] < num_samples_per_condition:
                        num_updates = random.choice(num_updates_options)
                        box_center_coords, box_grid_center_coords = self.make_box_grids(set_size)
                        
                        inital_marker_state = {}
                        for box_index in range(set_size):
                            row = random.randint(0, self.grid_size-1)
                            col = random.randint(0, self.grid_size-1)
                            inital_marker_state[box_index] = (row*self.grid_size) + col

                        updates = []
                        marker_states = [inital_marker_state]

                        for update_index in range(num_updates):
                            current_state = marker_states[-1]
                            invalid_update = True

                            cell_index = current_state[update_index%set_size]
                            cell_row = cell_index//self.grid_size
                            cell_col = cell_index%self.grid_size

                            while invalid_update:
                                invalid_update = False
                                update_direction = random.choice(['up_down', 'down_up', 
                                                                  'left_right', 'right_left', 
                                                                  'upleft_downright', 'downright_upleft', 
                                                                  'upright_downleft', 'downleft_upright'])

                                if cell_row == 0:
                                    if (update_direction == 'down_up' or 
                                        update_direction == 'downleft_upright' or 
                                        update_direction == 'downright_upleft'):
                                        invalid_update = True
                                if cell_row == self.grid_size - 1:
                                    if (update_direction == 'up_down' or 
                                        update_direction == 'upleft_downright' or 
                                        update_direction == 'upright_downleft'):
                                        invalid_update = True
                                if cell_col == 0:
                                    if (update_direction == 'right_left' or 
                                        update_direction == 'upright_downleft' or 
                                        update_direction == 'downright_upleft'):
                                        invalid_update = True
                                if cell_col == self.grid_size - 1:
                                    if (update_direction == 'left_right' or 
                                        update_direction == 'upleft_downright' or 
                                        update_direction == 'downleft_upright'):
                                        invalid_update = True

                            updates.append([update_index%set_size, update_direction])

                            next_state = current_state.copy()
                            if update_direction == 'up_down':
                                cell_row += 1
                            
                            elif update_direction == 'down_up':
                                cell_row -= 1
                                
                            elif update_direction == 'left_right':
                                cell_col += 1
                                
                            elif update_direction == 'right_left':
                                cell_col -= 1
                                
                            elif update_direction == 'upleft_downright':
                                cell_row += 1
                                cell_col += 1
                                
                            elif update_direction == 'downright_upleft':
                                cell_row -= 1
                                cell_col -= 1
                                
                            elif update_direction == 'upright_downleft':
                                cell_row += 1
                                cell_col -= 1
                                
                            elif update_direction == 'downleft_upright':
                                cell_row -= 1
                                cell_col += 1

                            next_state[update_index%set_size] = cell_row*self.grid_size + cell_col
                            marker_states.append(next_state)

                        final_marker_state = marker_states[-1]

                        probe_order = random.sample(range(set_size), set_size)
                        probe_gt = [final_marker_state[box_index] for box_index in probe_order]

                        memory_stim_fnames = [split+'_'+str(overall_sample_count).zfill(6)+'_memory.png']
                        update_stim_fnames = [split+'_'+str(overall_sample_count).zfill(6)+
                                              '_update_'+str(update_index).zfill(2)+'.png' 
                                              for update_index in range(num_updates)]
                        probe_stim_fnames = [split+'_'+str(overall_sample_count).zfill(6)+
                                             '_probe_'+str(probe_index).zfill(2)+'.png' 
                                             for probe_index in range(set_size)]
                        
                        trials[split].append({'set_size': set_size, 
                                              'grid_size': self.grid_size,
                                              'num_updates': num_updates, 
                                              'presentation_time': presentation_time,
                                              'box_center_coords': box_center_coords, 
                                              'box_grid_center_coords': box_grid_center_coords, 
                                              'inital_marker_state': inital_marker_state, 
                                              'final_marker_state': final_marker_state,
                                              'marker_states': marker_states, 
                                              'updates': updates, 
                                              'probe_order': probe_order,
                                              'probe_gt': probe_gt,
                                              'trial_type': split, 
                                              'trial_id': split+'_'+str(overall_sample_count).zfill(6), 
                                              'memory_stim_fnames': memory_stim_fnames, 
                                              'update_stim_fnames': update_stim_fnames, 
                                              'probe_stim_fnames': probe_stim_fnames})
                        
                        overall_sample_count += 1
                        per_condition_sample_count[(set_size, presentation_time)] += 1

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
                box_center_coords = trial['box_center_coords']
                box_grid_center_coords = trial['box_grid_center_coords']
                markers = trial['inital_marker_state']
                updates = trial['updates']
                probe_order = trial['probe_order']

                memory_stim_list = self.draw_memory_stim(box_center_coords, box_grid_center_coords, markers)
                update_stim_list = self.draw_update_stim(box_center_coords, updates)
                probe_stim_list = self.draw_probe_stim(box_center_coords, probe_order)
                
                if self.write:
                    memory_stim_fnames = trial['memory_stim_fnames']
                    update_stim_fnames = trial['update_stim_fnames']
                    probe_stim_fnames = trial['probe_stim_fnames']

                    for memory_stim, memory_stim_fname in zip(memory_stim_list, memory_stim_fnames):
                        memory_stim.save(os.path.join(write_dir, memory_stim_fname))
                    for update_stim, update_stim_fname in zip(update_stim_list, update_stim_fnames):
                        update_stim.save(os.path.join(write_dir, update_stim_fname))
                    for probe_stim, probe_stim_fname in zip(probe_stim_list, probe_stim_fnames):
                        probe_stim.save(os.path.join(write_dir, probe_stim_fname))

                else:
                    trial_stim = {}
                    trial_stim['memory_stim_list'] = memory_stim_list
                    trial_stim['update_stim_list'] = update_stim_list
                    trial_stim['probe_stim_list'] = probe_stim_list

                    trials_stim_list[split].append(trial_stim)

        if not self.write:
            return trials_stim_list


    def gen_held_out_trials(self, set_size_options, num_updates_options, 
                                  train_size, test_size):
        
        # Held out conditions
        # 1. Set size
        # 2. Number of updates
        # 3. Box Index 1 - Initial State (0, 0), (0, 1), (1, 0), (1, 1)
        # 4. Box Index 3 - First Update up_down, left_right, upleft_downright

        raise NotImplementedError

    def draw_memory_stim(self, box_center_coords, box_grid_center_coords, markers):
        memory_stim_list = []
        memory_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
        draw = ImageDraw.Draw(memory_stim)

        for box_index, box_center_coord in box_center_coords.items():
            draw.rectangle((box_center_coord[0] - 0.5*self.box_size, 
                            box_center_coord[1] - 0.5*self.box_size,
                            box_center_coord[0] + 0.5*self.box_size,
                            box_center_coord[1] + 0.5*self.box_size),
                            fill=self.color_map[self.box_color], 
                            outline=self.color_map[self.outline_color], 
                            width=self.outline_width)

        for box_index, cell_index in markers.items():
            marker_coord = box_grid_center_coords[box_index][cell_index]

            draw.rectangle((marker_coord[0] - self.marker_radius, marker_coord[1] - self.marker_radius, 
                            marker_coord[0] + self.marker_radius, marker_coord[1] + self.marker_radius), 
                            fill=self.color_map[self.marker_color])

            # draw.ellipse((marker_coord[0] - self.marker_radius, marker_coord[1] - self.marker_radius, 
            #               marker_coord[0] + self.marker_radius, marker_coord[1] + self.marker_radius), 
            #               fill=self.color_map[self.marker_color])
            
        memory_stim_list.append(memory_stim)
            
        return memory_stim_list
    
    def draw_update_stim(self, box_center_coords, updates):
        update_stim_list = []
        for update in updates:
            update_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
            draw = ImageDraw.Draw(update_stim)

            for box_index, box_center_coord in box_center_coords.items():
                draw.rectangle((box_center_coord[0] - 0.5*self.box_size, 
                                box_center_coord[1] - 0.5*self.box_size,
                                box_center_coord[0] + 0.5*self.box_size,
                                box_center_coord[1] + 0.5*self.box_size),
                                fill=self.color_map[self.box_color], outline=self.color_map[self.outline_color])

            box_index = update[0]
            update_direction = update[1]
            if update_direction == 'up_down':
                arrow_start_coord = (box_center_coords[box_index][0], 
                                     box_center_coords[box_index][1] - 0.3*self.box_size)

                arrow_end_coord = (box_center_coords[box_index][0], 
                                   box_center_coords[box_index][1] + 0.3*self.box_size)

            elif update_direction == 'down_up':
                arrow_start_coord = (box_center_coords[box_index][0], 
                                     box_center_coords[box_index][1] + 0.3*self.box_size)
                                
                arrow_end_coord = (box_center_coords[box_index][0], 
                                   box_center_coords[box_index][1] - 0.3*self.box_size)
                                  
            elif update_direction == 'left_right':
                arrow_start_coord = (box_center_coords[box_index][0] - 0.3*self.box_size, 
                                     box_center_coords[box_index][1]) 
                                
                arrow_end_coord = (box_center_coords[box_index][0] + 0.3*self.box_size, 
                                   box_center_coords[box_index][1])

            elif update_direction == 'right_left':
                arrow_start_coord = (box_center_coords[box_index][0] + 0.3*self.box_size, 
                                     box_center_coords[box_index][1]) 
                
                arrow_end_coord = (box_center_coords[box_index][0] - 0.3*self.box_size, 
                                   box_center_coords[box_index][1])

            elif update_direction == 'upleft_downright':
                arrow_start_coord = (box_center_coords[box_index][0] - 0.3*self.box_size, 
                                     box_center_coords[box_index][1] - 0.3*self.box_size)
                
                arrow_end_coord = (box_center_coords[box_index][0] + 0.3*self.box_size, 
                                   box_center_coords[box_index][1] + 0.3*self.box_size)

            elif update_direction == 'downright_upleft':
                arrow_start_coord = (box_center_coords[box_index][0] + 0.3*self.box_size, 
                                     box_center_coords[box_index][1] + 0.3*self.box_size)
                                
                arrow_end_coord = (box_center_coords[box_index][0] - 0.3*self.box_size, 
                                   box_center_coords[box_index][1] - 0.3*self.box_size)

            elif update_direction == 'upright_downleft':
                arrow_start_coord = (box_center_coords[box_index][0] + 0.3*self.box_size, 
                                     box_center_coords[box_index][1] - 0.3*self.box_size)

                arrow_end_coord = (box_center_coords[box_index][0] - 0.3*self.box_size, 
                                   box_center_coords[box_index][1] + 0.3*self.box_size)
            
            elif update_direction == 'downleft_upright':
                arrow_start_coord = (box_center_coords[box_index][0] - 0.3*self.box_size, 
                                     box_center_coords[box_index][1] + 0.3*self.box_size)

                arrow_end_coord = (box_center_coords[box_index][0] + 0.3*self.box_size, 
                                   box_center_coords[box_index][1] - 0.3*self.box_size)
                
            else:
                raise ValueError('Invalid update direction')
            
            arrow_start_coord = tuple(map(int, arrow_start_coord))
            arrow_end_coord = tuple(map(int, arrow_end_coord))

            cv_image = np.array(update_stim)
            cv_image = cv2.arrowedLine(cv_image, arrow_start_coord, arrow_end_coord, 
                                       self.arrow_color, self.arrow_width, 
                                       tipLength=self.arrow_tip_length)
            update_stim = Image.fromarray(cv_image)

            update_stim_list.append(update_stim)
            
        return update_stim_list
    
    def draw_probe_stim(self, box_center_coords, probe_order):
        probe_stim_list = []
        for probe_index in probe_order:
            probe_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
            draw = ImageDraw.Draw(probe_stim)

            for box_index, box_center_coord in box_center_coords.items():
                if box_index == probe_index:
                    draw.rectangle((box_center_coord[0] - 0.5*self.box_size, 
                                    box_center_coord[1] - 0.5*self.box_size,
                                    box_center_coord[0] + 0.5*self.box_size,
                                    box_center_coord[1] + 0.5*self.box_size),
                                    fill=self.color_map[self.probe_color], outline=self.color_map[self.outline_color])
                else:
                    draw.rectangle((box_center_coord[0] - 0.5*self.box_size, 
                                    box_center_coord[1] - 0.5*self.box_size,
                                    box_center_coord[0] + 0.5*self.box_size,
                                    box_center_coord[1] + 0.5*self.box_size),
                                    fill=self.color_map[self.box_color], outline=self.color_map[self.outline_color])

            probe_stim_list.append(probe_stim)        
        
        return probe_stim_list

    def make_box_grids(self, set_size):
        box_center_coords = {}
        box_grid_center_coords = {}

        if set_size == 1:
            box_center_coords[0] = [self.img_size[0] // 2, self.img_size[1] // 2]
        elif set_size % 2 == 0 or set_size in [3, 5, 7]:
            for i in range(set_size):
                theta = 2 * np.pi * i / set_size
                x = self.img_size[0] // 2 + self.box_draw_radius * np.cos(theta)
                y = self.img_size[1] // 2 + self.box_draw_radius * np.sin(theta)
                box_center_coords[i] = [x, y]
        else:
            raise NotImplementedError
        
        for index, coord in box_center_coords.items():
            grid_center_coords = {}
            cell_size = self.box_size // self.grid_size

            start_x = coord[0] - 0.5*cell_size*(self.grid_size - 1)
            start_y = coord[1] - 0.5*cell_size*(self.grid_size - 1)

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell_center_x = start_x + j*cell_size
                    cell_center_y = start_y + i*cell_size
                    grid_center_coords[(i*self.grid_size) + j] = [cell_center_x, cell_center_y]

            box_grid_center_coords[index] = grid_center_coords

        return box_center_coords, box_grid_center_coords