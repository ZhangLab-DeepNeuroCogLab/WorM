import os
import cv2
import json
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

class Complex_Span_DataGen:
    '''
    Task: Complex Span
    '''

    def __init__(self, data_path, num_storage_options, num_distractor_options, 
                 visual_memory_grid_size, 
                 spatial_distractor_grid_size_options, 
                 spatial_distractor_set_size_options, 
                 spatial_distractor_symmetry_offset_options, 
                 gen_random_trials, 
                 num_samples, img_size, write):
        
        self.data_path = data_path
        self.num_samples = num_samples
        self.img_size = (img_size, img_size)
        self.write = write

        self.visual_memory_grid_size = visual_memory_grid_size

        self.bg_color = 'gray'
        self.active_cell_color = 'red'
        self.inactive_cell_color = 'white'
        self.cell_outline_color = 'black'
        self.spatial_probe_outline_color = 'red'
        self.marker_color = 'red'
        self.spatial_mem_border_color = 'green'
        self.spatial_mem_border_width = 2
        self.spatial_mem_border_padding = 6
        self.marker_padding = 6
        self.memory_stim_padding = 9
        self.mem_rect_radius = 3
        self.stim_img_padding = 3
        self.rect_radius = 2
        self.probe_stim_padding = 2
        self.probe_recall_rect_radius = 0
        self.arrow_width = 2
        self.arrow_tip_length = 0.4
        self.arrow_color = (30, 30, 30)

        self.color_map = {'red': '#ee1d23', 'blue': '#015fae', 'violet': '#bc1b8d', 
                        'green': '#3ab54a', 'yellow': 'yellow', 'black': 'black', 
                        'white': 'white', 'brown': '#b78565', 'gray': '#d6d6d6'}
        
        self.red_colors = ['#880808', '#EE4B2B', '#C41E3A', '#D2042D', '#DC143C', '#8B0000', '#800000']
        self.blue_colors = ['#0000FF', '#0096FF', '#0047AB', '#1C05B3', '#00008B', '#3F00FF', '#0437F2']

        self.task_data_path = os.path.join(self.data_path, 'Complex_Span')

        if gen_random_trials:
            self.train_data_dir = os.path.join(self.task_data_path, 'train')
            self.test_data_dir = os.path.join(self.task_data_path, 'test')
            self.gen_test_data_dir = os.path.join(self.task_data_path, 'gen_test')

            self.train_num_samples = num_samples
            self.test_num_samples = int(num_samples * 0.1)
            self.gen_test_num_samples = 0 # int(num_samples * 0.1)

            assert self.train_num_samples % len(num_distractor_options) == 0
            assert self.test_num_samples % len(num_distractor_options) == 0

            trials = self.gen_random_trials(num_storage_options, 
                                            num_distractor_options, 
                                            spatial_distractor_grid_size_options, 
                                            spatial_distractor_set_size_options, 
                                            spatial_distractor_symmetry_offset_options)
            
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
        
    def gen_random_trials(self, num_storage_options, num_distractor_options,
                          spatial_distractor_grid_size_options, 
                          spatial_distractor_set_size_options, 
                          spatial_distractor_symmetry_offset_options):
        
        variations = ['spatial-spatial', 'spatial-visual', 'visual-spatial', 'visual-visual']
        
        trials = {'train': [], 'test': [], 'gen_test': []}

        for split in ['train', 'test', 'gen_test']:
            print('Generating {} trials'.format(split))
            if split == 'train':
                num_samples = self.train_num_samples
                num_storage = num_storage_options
                num_distractors = num_distractor_options
                spatial_distractor_grid_size = spatial_distractor_grid_size_options
                spatial_distractor_set_size = spatial_distractor_set_size_options
            elif split == 'test':
                num_samples = self.test_num_samples
                num_storage = num_storage_options
                num_distractors = num_distractor_options
                spatial_distractor_grid_size = spatial_distractor_grid_size_options
                spatial_distractor_set_size = spatial_distractor_set_size_options
            elif split == 'gen_test':
                num_samples = self.gen_test_num_samples

            overall_sample_count = 0
            per_condition_sample_count = {}

            num_samples_per_condition = num_samples // (len(num_distractors)*len(variations))

            for variation in variations:
                for num_distractor in num_distractors:
                    print('Generating {} {} trials'.format(variation, num_distractor))
                    per_condition_sample_count[(variation, num_distractor)] = 0

                    while per_condition_sample_count[(variation, num_distractor)] < num_samples_per_condition:
                        
                        mem_distractor_1 = None
                        mem_distractor_2 = None
                        distractor_grid_size = None
                        
                        if variation == 'spatial-spatial':
                            memory_1_gt = random.choice(['012', '210', '345', '543', 
                                                         '678', '876', '036', '630', 
                                                         '147', '741', '258', '852', 
                                                         '048', '840', '246', '642'])
                            
                            memory_2_gt = random.choice(['012', '210', '345', '543', 
                                                         '678', '876', '036', '630', 
                                                         '147', '741', '258', '852', 
                                                         '048', '840', '246', '642'])
                            
                            recall_gt_1 = random.randint(0, 7)
                            recall_gt_2 = random.randint(0, 7)

                            distractor_gts = []
                            distractor_gt_items = []

                            for _ in range(num_distractor*2):
                                distractor_gt = random.choice([0, 1])

                                distractor_grid_size = random.choice(spatial_distractor_grid_size)
                                distractor_set_size = random.choice(spatial_distractor_set_size)

                                left_grid_cell_idxs = [((random.choice(range(distractor_grid_size))*distractor_grid_size) + 
                                                random.choice(range(distractor_grid_size//2))) 
                                                for _ in range(distractor_set_size // 2)]
                                            
                                right_grid_cell_idxs = [(idx + (distractor_grid_size - 2*(idx%distractor_grid_size)) - 1) 
                                                        for idx in left_grid_cell_idxs]
                                
                                if distractor_gt == 1:
                                    symmetry_offset = 0
                                    gt_item = left_grid_cell_idxs + right_grid_cell_idxs
                                else:
                                    symmetry_offset = random.choice(spatial_distractor_symmetry_offset_options)
                                    while symmetry_offset > distractor_set_size:
                                        symmetry_offset = random.choice(spatial_distractor_symmetry_offset_options)

                                    right_inactive_grid_cell_idxs = [idx for idx in range(distractor_grid_size**2) 
                                                                    if (idx not in right_grid_cell_idxs) and 
                                                                    (idx%distractor_grid_size >= distractor_grid_size//2)]

                                    right_grid_cell_idxs = random.sample(right_grid_cell_idxs, 
                                                                        len(right_grid_cell_idxs)-symmetry_offset//2)

                                    sample_inactive_right_grid_cell_idxs = random.sample(right_inactive_grid_cell_idxs, 
                                                                                        symmetry_offset//2)
                                    
                                    gt_item = left_grid_cell_idxs + right_grid_cell_idxs + sample_inactive_right_grid_cell_idxs

                                random.shuffle(gt_item)

                                distractor_gts.append(distractor_gt)
                                distractor_gt_items.append(gt_item)


                        elif variation == 'spatial-visual':
                            memory_1_gt = random.choice(['012', '210', '345', '543', 
                                                        '678', '876', '036', '630', 
                                                        '147', '741', '258', '852', 
                                                        '048', '840', '246', '642'])
                            
                            memory_2_gt = random.choice(['012', '210', '345', '543', 
                                                        '678', '876', '036', '630', 
                                                        '147', '741', '258', '852', 
                                                        '048', '840', '246', '642'])
                            
                            recall_gt_1 = random.randint(0, 7)
                            recall_gt_2 = random.randint(0, 7)

                            distractor_gts = []
                            distractor_gt_items = []

                            for _ in range(num_distractor*2):
                                distractor_gt = random.choice([0, 1])

                                if distractor_gt == 0:
                                    distractor_item = random.choice(self.red_colors)
                                else:
                                    distractor_item = random.choice(self.blue_colors)

                                distractor_gts.append(distractor_gt)
                                distractor_gt_items.append(distractor_item)
                                

                        elif variation == 'visual-visual':
                            memory_1_gt = random.sample(range(self.visual_memory_grid_size**2), 
                                                        self.visual_memory_grid_size**2//2)
                            
                            memory_2_gt = random.sample(range(self.visual_memory_grid_size**2), 
                                                        self.visual_memory_grid_size**2//2)
                            
                            recall_gt_1 = random.randint(0, 7)
                            recall_gt_2 = random.randint(0, 7)

                            mem_distractor_diff = 6
                            mem_distractor_1 = []
                            for _ in range(7):
                                memory_gt_dark_cells = memory_1_gt
                                memory_gt_white_cells = [cell for cell in range(self.visual_memory_grid_size**2)
                                                            if cell not in memory_gt_dark_cells]
                                
                                distractor_dark_cells = random.sample(memory_gt_dark_cells, 
                                                                    len(memory_gt_dark_cells)-mem_distractor_diff//2)
                                sample_memory_gt_white_cells = random.sample(memory_gt_white_cells,
                                                                        mem_distractor_diff//2)

                                distractor_item = distractor_dark_cells + sample_memory_gt_white_cells
                                mem_distractor_1.append(distractor_item)
                            
                            mem_distractor_2 = []
                            for _ in range(7):
                                memory_gt_dark_cells = memory_2_gt
                                memory_gt_white_cells = [cell for cell in range(self.visual_memory_grid_size**2)
                                                            if cell not in memory_gt_dark_cells]
                                
                                distractor_dark_cells = random.sample(memory_gt_dark_cells, 
                                                                    len(memory_gt_dark_cells)-mem_distractor_diff//2)
                                sample_memory_gt_white_cells = random.sample(memory_gt_white_cells,
                                                                        mem_distractor_diff//2)

                                distractor_item = distractor_dark_cells + sample_memory_gt_white_cells
                                mem_distractor_2.append(distractor_item)


                            distractor_gts = []
                            distractor_gt_items = []

                            for _ in range(num_distractor*2):
                                distractor_gt = random.choice([0, 1])

                                if distractor_gt == 0:
                                    distractor_item = random.choice(self.red_colors)
                                else:
                                    distractor_item = random.choice(self.blue_colors)

                                distractor_gts.append(distractor_gt)
                                distractor_gt_items.append(distractor_item)


                        elif variation == 'visual-spatial':
                            memory_1_gt = random.sample(range(self.visual_memory_grid_size**2), 
                                                        self.visual_memory_grid_size**2//2)
                            
                            memory_2_gt = random.sample(range(self.visual_memory_grid_size**2), 
                                                        self.visual_memory_grid_size**2//2)
                            
                            recall_gt_1 = random.randint(0, 7)
                            recall_gt_2 = random.randint(0, 7)

                            mem_distractor_diff = 6
                            mem_distractor_1 = []
                            for _ in range(7):
                                memory_gt_dark_cells = memory_1_gt
                                memory_gt_white_cells = [cell for cell in range(self.visual_memory_grid_size**2)
                                                            if cell not in memory_gt_dark_cells]
                                
                                distractor_dark_cells = random.sample(memory_gt_dark_cells, 
                                                                    len(memory_gt_dark_cells)-mem_distractor_diff//2)
                                sample_memory_gt_white_cells = random.sample(memory_gt_white_cells,
                                                                        mem_distractor_diff//2)

                                distractor_item = distractor_dark_cells + sample_memory_gt_white_cells
                                mem_distractor_1.append(distractor_item)
                            
                            mem_distractor_2 = []
                            for _ in range(7):
                                memory_gt_dark_cells = memory_2_gt
                                memory_gt_white_cells = [cell for cell in range(self.visual_memory_grid_size**2)
                                                            if cell not in memory_gt_dark_cells]
                                
                                distractor_dark_cells = random.sample(memory_gt_dark_cells, 
                                                                    len(memory_gt_dark_cells)-mem_distractor_diff//2)
                                sample_memory_gt_white_cells = random.sample(memory_gt_white_cells,
                                                                        mem_distractor_diff//2)

                                distractor_item = distractor_dark_cells + sample_memory_gt_white_cells
                                mem_distractor_2.append(distractor_item)


                            distractor_gts = []
                            distractor_gt_items = []

                            for _ in range(num_distractor*2):
                                distractor_gt = random.choice([0, 1])

                                distractor_grid_size = random.choice(spatial_distractor_grid_size)
                                distractor_set_size = random.choice(spatial_distractor_set_size)

                                left_grid_cell_idxs = [((random.choice(range(distractor_grid_size))*distractor_grid_size) + 
                                                random.choice(range(distractor_grid_size//2))) 
                                                for _ in range(distractor_set_size // 2)]
                                            
                                right_grid_cell_idxs = [(idx + (distractor_grid_size - 2*(idx%distractor_grid_size)) - 1) 
                                                        for idx in left_grid_cell_idxs]
                                
                                if distractor_gt == 1:
                                    symmetry_offset = 0
                                    gt_item = left_grid_cell_idxs + right_grid_cell_idxs
                                else:
                                    symmetry_offset = random.choice(spatial_distractor_symmetry_offset_options)
                                    while symmetry_offset > distractor_set_size:
                                        symmetry_offset = random.choice(spatial_distractor_symmetry_offset_options)

                                    right_inactive_grid_cell_idxs = [idx for idx in range(distractor_grid_size**2) 
                                                                    if (idx not in right_grid_cell_idxs) and 
                                                                    (idx%distractor_grid_size >= distractor_grid_size//2)]

                                    right_grid_cell_idxs = random.sample(right_grid_cell_idxs, 
                                                                        len(right_grid_cell_idxs)-symmetry_offset//2)

                                    sample_inactive_right_grid_cell_idxs = random.sample(right_inactive_grid_cell_idxs, 
                                                                                        symmetry_offset//2)
                                    
                                    gt_item = left_grid_cell_idxs + right_grid_cell_idxs + sample_inactive_right_grid_cell_idxs

                                random.shuffle(gt_item)

                                distractor_gts.append(distractor_gt)
                                distractor_gt_items.append(gt_item)



                        memory_stim_1_fnames = [split+'_'+str(overall_sample_count).zfill(6)+
                                                '_memory_1_'+str(i).zfill(3)+'.png' 
                                                for i in range(3)]
                        
                        memory_stim_2_fnames = [split+'_'+str(overall_sample_count).zfill(6)+
                                                '_memory_2_'+str(i).zfill(3)+'.png' 
                                                for i in range(3)]
                        
                        distractor_stim_1_fnames = [split+'_'+str(overall_sample_count).zfill(6)+
                                                '_distractor_1_'+str(i).zfill(3)+'.png' 
                                                for i in range(num_distractor)]
                        
                        distractor_stim_2_fnames = [split+'_'+str(overall_sample_count).zfill(6)+
                                                '_distractor_2_'+str(i).zfill(3)+'.png' 
                                                for i in range(num_distractor)]
                        
                        probe_stim_fnames = [split+'_'+str(overall_sample_count).zfill(6)+
                                                '_probe_'+str(i)+'.png' 
                                                for i in range(2)]
                        
                        trials[split].append({'memory_1_gt': memory_1_gt, 
                                                'memory_2_gt': memory_2_gt, 
                                                'recall_gt_1': recall_gt_1, 
                                                'recall_gt_2': recall_gt_2, 
                                                'mem_distractor_1': mem_distractor_1, 
                                                'mem_distractor_2': mem_distractor_2, 
                                                'distractor_grid_size': distractor_grid_size, 
                                                'distractor_gts': distractor_gts, 
                                                'distractor_gt_items': distractor_gt_items, 
                                                'memory_stim_1_fnames': memory_stim_1_fnames, 
                                                'memory_stim_2_fnames': memory_stim_2_fnames, 
                                                'distractor_stim_1_fnames': distractor_stim_1_fnames, 
                                                'distractor_stim_2_fnames': distractor_stim_2_fnames, 
                                                'probe_stim_fnames': probe_stim_fnames,
                                                'num_distractor': num_distractor, 
                                                'variation': variation, 
                                                'trial_type': split, 
                                                'trial_id': split+'_'+str(overall_sample_count).zfill(6)})
                        
                        overall_sample_count += 1
                        per_condition_sample_count[(variation, num_distractor)] += 1

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
                variation = trial['variation']
                memory_1_gt = trial['memory_1_gt']
                memory_2_gt = trial['memory_2_gt']
                distractor_gt_items = trial['distractor_gt_items']
                distractor_grid_size = trial['distractor_grid_size']
                num_distractor = trial['num_distractor']
                mem_distractor_1 = trial['mem_distractor_1']
                mem_distractor_2 = trial['mem_distractor_2']
                recall_gt_1 = trial['recall_gt_1']
                recall_gt_2 = trial['recall_gt_2']

                if variation in ['visual-visual', 'visual-spatial']:
                    memory_stim_list_1, memory_stim_list_2 = self.draw_memory_visual_stim(memory_1_gt, memory_2_gt)
                elif variation in ['spatial-visual', 'spatial-spatial']:
                    memory_stim_list_1, memory_stim_list_2 = self.draw_memory_spatial_stim(memory_1_gt, memory_2_gt)

                if variation in ['visual-visual', 'spatial-visual']:
                    distractor_stim_list = self.draw_distractor_visual_stim(distractor_gt_items)
                elif variation in ['visual-spatial', 'spatial-spatial']:
                    distractor_stim_list = self.draw_distractor_spatial_stim(distractor_gt_items, 
                                                                             distractor_grid_size)
                    
                if variation in ['spatial-visual', 'spatial-spatial']:
                    probe_stim_1, probe_stim_2 = self.draw_spatial_probe(memory_1_gt, memory_2_gt, recall_gt_1, recall_gt_2)
                elif variation in ['visual-visual', 'visual-spatial']:
                    probe_stim_1, probe_stim_2 = self.draw_visual_probe(memory_1_gt, memory_2_gt, 
                                                                       mem_distractor_1, mem_distractor_2, 
                                                                       recall_gt_1, recall_gt_2)
                    
                if self.write:
                    memory_stim_1_fnames = trial['memory_stim_1_fnames']
                    memory_stim_2_fnames = trial['memory_stim_2_fnames']
                    distractor_stim_1_fnames = trial['distractor_stim_1_fnames']
                    distractor_stim_2_fnames = trial['distractor_stim_2_fnames']

                    probe_stim_fnames = trial['probe_stim_fnames']

                    if variation in ['visual-visual', 'visual-spatial']:
                        memory_stim_list_1 = [memory_stim_list_1] * 3
                        memory_stim_list_2 = [memory_stim_list_2] * 3
                        for memory_stim, memory_stim_fname in zip(memory_stim_list_1, memory_stim_1_fnames):
                            memory_stim.save(os.path.join(write_dir, memory_stim_fname))
                        for memory_stim, memory_stim_fname in zip(memory_stim_list_2, memory_stim_2_fnames):
                            memory_stim.save(os.path.join(write_dir, memory_stim_fname))
                    elif variation in ['spatial-visual', 'spatial-spatial']:
                        for memory_stim, memory_stim_fname in zip(memory_stim_list_1, memory_stim_1_fnames):
                            memory_stim.save(os.path.join(write_dir, memory_stim_fname))
                        for memory_stim, memory_stim_fname in zip(memory_stim_list_2, memory_stim_2_fnames):
                            memory_stim.save(os.path.join(write_dir, memory_stim_fname))

                    for distractor_stim, distractor_stim_fname in zip(distractor_stim_list[:num_distractor], distractor_stim_1_fnames):
                        distractor_stim.save(os.path.join(write_dir, distractor_stim_fname))
                    for distractor_stim, distractor_stim_fname in zip(distractor_stim_list[num_distractor:], distractor_stim_2_fnames):
                        distractor_stim.save(os.path.join(write_dir, distractor_stim_fname))

                    probe_stim_1.save(os.path.join(write_dir, probe_stim_fnames[0]))
                    probe_stim_2.save(os.path.join(write_dir, probe_stim_fnames[1]))

        if self.write:
            blank_stim = self.draw_blank_stim()
            blank_stim.save(os.path.join(self.task_data_path, 'blank.png'))


    def draw_blank_stim(self):
        blank_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
        
        return blank_stim
    
    def draw_spatial_probe(self, memory_1_gt, memory_2_gt, recall_gt_1, recall_gt_2):
        if memory_1_gt in ['012', '678']:               
            memory_1_gttt = '345'
        elif memory_1_gt in ['210', '876']:
            memory_1_gttt = '543'
        elif memory_1_gt in ['036', '258']:
            memory_1_gttt = '147'
        elif memory_1_gt in ['630', '852']:
            memory_1_gttt = '741'
        else:   
            memory_1_gttt = memory_1_gt


        if memory_2_gt in ['012', '678']:               
            memory_2_gttt = '345'
        elif memory_2_gt in ['210', '876']:
            memory_2_gttt = '543'
        elif memory_2_gt in ['036', '258']:
            memory_2_gttt = '147'
        elif memory_2_gt in ['630', '852']:
            memory_2_gttt = '741'
        else:
            memory_2_gttt = memory_2_gt


        probe_1_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
        draw = ImageDraw.Draw(probe_1_stim)

        big_grid_cell_size_x_px = self.img_size[0] // 3
        big_grid_cell_size_y_px = self.img_size[1] // 3

        probe_stim_x_padding = self.probe_stim_padding

        grid_size_px = (self.img_size[0]-(2*4*probe_stim_x_padding)) // 3
        grid_cell_size_px = grid_size_px // 3

        probe_stim_y_padding = (big_grid_cell_size_y_px - grid_size_px) // 2

        distractor_1 = ['345', '543', '147', '741', '048', '840', '246', '642']
        distractor_1.remove(memory_1_gttt)
        random.shuffle(distractor_1)

        coords_overall = []
        distractor_count = 0


        for i in range(8):
            row = i // 3
            col = i % 3

            for cell_index in range(3**2):
                cell_x = cell_index % 3
                cell_y = cell_index // 3
                
                if i in [6, 7]:
                    cell_x = (col*big_grid_cell_size_x_px + 
                            cell_x*grid_cell_size_px + probe_stim_x_padding + 17)
                else:
                    cell_x = (col*big_grid_cell_size_x_px + 
                            cell_x*grid_cell_size_px + probe_stim_x_padding + 1)
                cell_y = (row*big_grid_cell_size_y_px + 
                        cell_y*grid_cell_size_px + probe_stim_y_padding)
                
                draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                   radius=self.probe_recall_rect_radius, fill=self.color_map[self.active_cell_color], 
                                   outline=self.color_map[self.spatial_probe_outline_color])


        for i in range(8):
            row = i // 3
            col = i % 3

            if i == recall_gt_1:

                for cell_index in range(3**2):
                    cell_x = cell_index % 3
                    cell_y = cell_index // 3

                    if i in [6, 7]:
                        cell_x = (col*big_grid_cell_size_x_px + 
                                cell_x*grid_cell_size_px + probe_stim_x_padding + 17)
                    else:
                        cell_x = (col*big_grid_cell_size_x_px + 
                                cell_x*grid_cell_size_px + probe_stim_x_padding + 1)
                    cell_y = (row*big_grid_cell_size_y_px + 
                            cell_y*grid_cell_size_px + probe_stim_y_padding)
                    
                    if cell_index == int(memory_1_gttt[0]):
                        coord_0 = (cell_x+(grid_cell_size_px//2), cell_y+(grid_cell_size_px//2))
                    elif cell_index == int(memory_1_gttt[2]):
                        coord_1 = (cell_x+(grid_cell_size_px//2), cell_y+(grid_cell_size_px//2))

            else:
                for cell_index in range(3**2):
                    cell_x = cell_index % 3
                    cell_y = cell_index // 3

                    if i in [6, 7]:
                        cell_x = (col*big_grid_cell_size_x_px + 
                                cell_x*grid_cell_size_px + probe_stim_x_padding + 17)
                    else:
                        cell_x = (col*big_grid_cell_size_x_px + 
                                cell_x*grid_cell_size_px + probe_stim_x_padding + 1)
                    cell_y = (row*big_grid_cell_size_y_px + 
                            cell_y*grid_cell_size_px + probe_stim_y_padding)
                    
                    if cell_index == int(distractor_1[distractor_count][0]):
                        coord_0 = (cell_x+(grid_cell_size_px//2), cell_y+(grid_cell_size_px//2))
                    elif cell_index == int(distractor_1[distractor_count][2]):
                        coord_1 = (cell_x+(grid_cell_size_px//2), cell_y+(grid_cell_size_px//2))


                distractor_count += 1

            coords = [coord_0, coord_1]
            coords_overall.append(coords)

            cv_image = np.array(probe_1_stim)
            for coords in coords_overall:
                cv_image = cv2.arrowedLine(cv_image, coords[0], coords[1], 
                                           self.arrow_color, self.arrow_width, 
                                            tipLength=self.arrow_tip_length)
            
            probe_1_stim = Image.fromarray(cv_image)



        probe_2_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
        draw = ImageDraw.Draw(probe_2_stim)

        big_grid_cell_size_x_px = self.img_size[0] // 3
        big_grid_cell_size_y_px = self.img_size[1] // 3

        probe_stim_x_padding = self.probe_stim_padding

        grid_size_px = (self.img_size[0]-(2*4*probe_stim_x_padding)) // 3
        grid_cell_size_px = grid_size_px // 3

        probe_stim_y_padding = (big_grid_cell_size_y_px - grid_size_px) // 2

        distractor_2 = ['345', '543', '147', '741', '048', '840', '246', '642']
        distractor_2.remove(memory_2_gttt)
        random.shuffle(distractor_2)

        coords_overall = []
        distractor_count = 0


        for i in range(8):
            row = i // 3
            col = i % 3

            for cell_index in range(3**2):
                cell_x = cell_index % 3
                cell_y = cell_index // 3
                
                if i in [6, 7]:
                    cell_x = (col*big_grid_cell_size_x_px + 
                            cell_x*grid_cell_size_px + probe_stim_x_padding + 17)
                else:
                    cell_x = (col*big_grid_cell_size_x_px + 
                            cell_x*grid_cell_size_px + probe_stim_x_padding + 1)
                cell_y = (row*big_grid_cell_size_y_px + 
                        cell_y*grid_cell_size_px + probe_stim_y_padding)
                
                draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                   radius=self.probe_recall_rect_radius, fill=self.color_map[self.active_cell_color], 
                                   outline=self.color_map[self.spatial_probe_outline_color])


        for i in range(8):
            row = i // 3
            col = i % 3

            if i == recall_gt_2:
                for cell_index in range(3**2):
                    cell_x = cell_index % 3
                    cell_y = cell_index // 3

                    if i in [6, 7]:
                        cell_x = (col*big_grid_cell_size_x_px + 
                                cell_x*grid_cell_size_px + probe_stim_x_padding + 17)
                    else:
                        cell_x = (col*big_grid_cell_size_x_px + 
                                cell_x*grid_cell_size_px + probe_stim_x_padding + 1)
                    cell_y = (row*big_grid_cell_size_y_px + 
                            cell_y*grid_cell_size_px + probe_stim_y_padding)
                    
                    if cell_index == int(memory_2_gttt[0]):
                        coord_0 = (cell_x+(grid_cell_size_px//2), cell_y+(grid_cell_size_px//2))
                    elif cell_index == int(memory_2_gttt[2]):
                        coord_1 = (cell_x+(grid_cell_size_px//2), cell_y+(grid_cell_size_px//2))
                    
                    draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                   radius=self.probe_recall_rect_radius, fill=self.color_map[self.active_cell_color], 
                                   outline=self.color_map[self.spatial_probe_outline_color])

            else:
                for cell_index in range(3**2):
                    cell_x = cell_index % 3
                    cell_y = cell_index // 3

                    if i in [6, 7]:
                        cell_x = (col*big_grid_cell_size_x_px + 
                                cell_x*grid_cell_size_px + probe_stim_x_padding + 17)
                    else:
                        cell_x = (col*big_grid_cell_size_x_px + 
                                cell_x*grid_cell_size_px + probe_stim_x_padding + 1)
                    cell_y = (row*big_grid_cell_size_y_px + 
                            cell_y*grid_cell_size_px + probe_stim_y_padding)
                    
                    if cell_index == int(distractor_2[distractor_count][0]):
                        coord_0 = (cell_x+(grid_cell_size_px//2), cell_y+(grid_cell_size_px//2))
                    elif cell_index == int(distractor_2[distractor_count][2]):
                        coord_1 = (cell_x+(grid_cell_size_px//2), cell_y+(grid_cell_size_px//2))

                    draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                   radius=self.probe_recall_rect_radius, fill=self.color_map[self.active_cell_color], 
                                   outline=self.color_map[self.spatial_probe_outline_color])

                distractor_count += 1

            coords = [coord_0, coord_1]
            coords_overall.append(coords)

            cv_image = np.array(probe_2_stim)
            for coords in coords_overall:
                cv_image = cv2.arrowedLine(cv_image, coords[0], coords[1], 
                                           self.arrow_color, self.arrow_width, 
                                            tipLength=self.arrow_tip_length)
            
            probe_2_stim = Image.fromarray(cv_image)

        return probe_1_stim, probe_2_stim

    
    def draw_visual_probe(self, memory_1_gt, memory_2_gt, mem_distractor_1, mem_distractor_2, 
                          recall_gt_1, recall_gt_2):
        probe_1_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
        draw = ImageDraw.Draw(probe_1_stim)

        big_grid_cell_size_x_px = self.img_size[0] // 3
        big_grid_cell_size_y_px = self.img_size[1] // 3

        probe_stim_x_padding = self.probe_stim_padding

        grid_size_px = (self.img_size[0]-(2*4*probe_stim_x_padding)) // 3
        grid_cell_size_px = grid_size_px // self.visual_memory_grid_size

        probe_stim_y_padding = (big_grid_cell_size_y_px - grid_size_px) // 2

        distractor_counter = 0

        for i in range(8):
            row = i // 3
            col = i % 3

            if i == recall_gt_1:
                for cell_index in range(self.visual_memory_grid_size**2):
                    cell_x = cell_index % self.visual_memory_grid_size
                    cell_y = cell_index // self.visual_memory_grid_size

                    if i in [6, 7]:
                        cell_x = (col*big_grid_cell_size_x_px + 
                                cell_x*grid_cell_size_px + probe_stim_x_padding + 17)
                    else:
                        cell_x = (col*big_grid_cell_size_x_px + 
                                cell_x*grid_cell_size_px + probe_stim_x_padding + 1)
                    cell_y = (row*big_grid_cell_size_y_px + 
                            cell_y*grid_cell_size_px + probe_stim_y_padding)

                    if cell_index in memory_1_gt:
                        draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                    radius=self.probe_recall_rect_radius, fill=self.color_map[self.active_cell_color], outline=self.color_map[self.cell_outline_color])
                    else:
                        draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                    radius=self.probe_recall_rect_radius, fill=self.color_map[self.inactive_cell_color], outline=self.color_map[self.cell_outline_color])
                        
            else:
                for cell_index in range(self.visual_memory_grid_size**2):
                    cell_x = cell_index % self.visual_memory_grid_size
                    cell_y = cell_index // self.visual_memory_grid_size

                    if i in [6, 7]:
                        cell_x = (col*big_grid_cell_size_x_px + 
                            cell_x*grid_cell_size_px + probe_stim_x_padding + 17)
                    else:
                        cell_x = (col*big_grid_cell_size_x_px + 
                                cell_x*grid_cell_size_px + probe_stim_x_padding + 1)
                    cell_y = (row*big_grid_cell_size_y_px + 
                            cell_y*grid_cell_size_px + probe_stim_y_padding)

                    if cell_index in mem_distractor_1[distractor_counter]:
                        draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                    radius=self.probe_recall_rect_radius, fill=self.color_map[self.active_cell_color], outline=self.color_map[self.cell_outline_color])
                    else:
                        draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                    radius=self.probe_recall_rect_radius, fill=self.color_map[self.inactive_cell_color], outline=self.color_map[self.cell_outline_color])

                distractor_counter += 1


        probe_2_stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
        draw = ImageDraw.Draw(probe_2_stim)

        big_grid_cell_size_x_px = self.img_size[0] // 3
        big_grid_cell_size_y_px = self.img_size[1] // 3

        probe_stim_x_padding = self.probe_stim_padding

        grid_size_px = (self.img_size[0]-(2*4*probe_stim_x_padding)) // 3
        grid_cell_size_px = grid_size_px // self.visual_memory_grid_size

        probe_stim_y_padding = (big_grid_cell_size_y_px - grid_size_px) // 2

        distractor_counter = 0

        for i in range(8):
            row = i // 3
            col = i % 3

            if i == recall_gt_2:
                for cell_index in range(self.visual_memory_grid_size**2):
                    cell_x = cell_index % self.visual_memory_grid_size
                    cell_y = cell_index // self.visual_memory_grid_size

                    if i in [6, 7]:
                        cell_x = (col*big_grid_cell_size_x_px + 
                            cell_x*grid_cell_size_px + probe_stim_x_padding + 17)
                    else:
                        cell_x = (col*big_grid_cell_size_x_px + 
                                cell_x*grid_cell_size_px + probe_stim_x_padding + 1)
                    cell_y = (row*big_grid_cell_size_y_px + 
                            cell_y*grid_cell_size_px + probe_stim_y_padding)

                    if cell_index in memory_2_gt:
                        draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                    radius=self.probe_recall_rect_radius, fill=self.color_map[self.active_cell_color], outline=self.color_map[self.cell_outline_color])
                    else:
                        draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                    radius=self.probe_recall_rect_radius, fill=self.color_map[self.inactive_cell_color], outline=self.color_map[self.cell_outline_color])
                        
            else:
                for cell_index in range(self.visual_memory_grid_size**2):
                    cell_x = cell_index % self.visual_memory_grid_size
                    cell_y = cell_index // self.visual_memory_grid_size

                    if i in [6, 7]:
                        cell_x = (col*big_grid_cell_size_x_px + 
                            cell_x*grid_cell_size_px + probe_stim_x_padding + 17)
                    else:
                        cell_x = (col*big_grid_cell_size_x_px + 
                                cell_x*grid_cell_size_px + probe_stim_x_padding + 1)
                    cell_y = (row*big_grid_cell_size_y_px + 
                            cell_y*grid_cell_size_px + probe_stim_y_padding)

                    if cell_index in mem_distractor_2[distractor_counter]:
                        draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                    radius=self.probe_recall_rect_radius, fill=self.color_map[self.active_cell_color], outline=self.color_map[self.cell_outline_color])
                    else:
                        draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                    radius=self.probe_recall_rect_radius, fill=self.color_map[self.inactive_cell_color], outline=self.color_map[self.cell_outline_color])

                distractor_counter += 1

        return probe_1_stim, probe_2_stim


    def draw_memory_visual_stim(self, memory_gt_1, memory_gt_2):
        memory_stim_1 = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
        draw = ImageDraw.Draw(memory_stim_1)

        grid_size_px = self.img_size[0] - 2*self.memory_stim_padding
        grid_cell_size_px = grid_size_px // self.visual_memory_grid_size

        for cell_index in range(self.visual_memory_grid_size**2):
            cell_x = cell_index % self.visual_memory_grid_size
            cell_y = cell_index // self.visual_memory_grid_size

            cell_x = cell_x * grid_cell_size_px + self.memory_stim_padding
            cell_y = cell_y * grid_cell_size_px + self.memory_stim_padding

            if cell_index in memory_gt_1:
                draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                radius=self.mem_rect_radius,
                                fill=self.color_map[self.active_cell_color], 
                                outline=self.color_map[self.cell_outline_color])
            else:
                draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                radius=self.mem_rect_radius,
                                fill=self.color_map[self.inactive_cell_color], 
                                outline=self.color_map[self.cell_outline_color])
                
        
        memory_stim_2 = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
        draw = ImageDraw.Draw(memory_stim_2)

        grid_size_px = self.img_size[0] - 2*self.memory_stim_padding
        grid_cell_size_px = grid_size_px // self.visual_memory_grid_size

        for cell_index in range(self.visual_memory_grid_size**2):
            cell_x = cell_index % self.visual_memory_grid_size
            cell_y = cell_index // self.visual_memory_grid_size

            cell_x = cell_x * grid_cell_size_px + self.memory_stim_padding
            cell_y = cell_y * grid_cell_size_px + self.memory_stim_padding

            if cell_index in memory_gt_2:
                draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                radius=self.mem_rect_radius,
                                fill=self.color_map[self.active_cell_color], 
                                outline=self.color_map[self.cell_outline_color])
            else:
                draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                                radius=self.mem_rect_radius,
                                fill=self.color_map[self.inactive_cell_color], 
                                outline=self.color_map[self.cell_outline_color])
                
        return memory_stim_1, memory_stim_2
    

    def draw_memory_spatial_stim(self, memory_gt_1, memory_gt_2):
        memory_stim_1_list = []
        for i in range(3):
            memory_stim_1 = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
            draw = ImageDraw.Draw(memory_stim_1)

            grid_size_px = self.img_size[0] - 2*self.memory_stim_padding
            grid_cell_size_px = grid_size_px // 3

            draw.rectangle([self.spatial_mem_border_padding, self.spatial_mem_border_padding, 
                            self.img_size[0]-self.spatial_mem_border_padding, self.img_size[1]-self.spatial_mem_border_padding],
                            outline=self.color_map[self.spatial_mem_border_color], 
                            width=self.spatial_mem_border_width)

            for cell_index in range(3**2):
                cell_x = cell_index % 3
                cell_y = cell_index // 3

                cell_x = cell_x * grid_cell_size_px + self.memory_stim_padding
                cell_y = cell_y * grid_cell_size_px + self.memory_stim_padding

                if cell_index == int(memory_gt_1[i]):
                    # draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                    #                 radius=self.mem_rect_radius,
                    #                 fill=self.color_map[self.active_cell_color], 
                    #                 outline=self.color_map[self.cell_outline_color])
                    
                    draw.ellipse([cell_x+self.marker_padding, cell_y+self.marker_padding, 
                                  cell_x+grid_cell_size_px-self.marker_padding, 
                                  cell_y+grid_cell_size_px-self.marker_padding], 
                          fill=self.color_map[self.marker_color])

                else:
                    # Do nothing
                    pass

            memory_stim_1_list.append(memory_stim_1)


        memory_stim_2_list = []
        for i in range(3):
            memory_stim_2 = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
            draw = ImageDraw.Draw(memory_stim_2)

            grid_size_px = self.img_size[0] - 2*self.memory_stim_padding
            grid_cell_size_px = grid_size_px // 3

            draw.rectangle([self.spatial_mem_border_padding, self.spatial_mem_border_padding, 
                            self.img_size[0]-self.spatial_mem_border_padding, self.img_size[1]-self.spatial_mem_border_padding],
                            outline=self.color_map[self.spatial_mem_border_color], 
                            width=self.spatial_mem_border_width)

            for cell_index in range(3**2):
                cell_x = cell_index % 3
                cell_y = cell_index // 3

                cell_x = cell_x * grid_cell_size_px + self.memory_stim_padding
                cell_y = cell_y * grid_cell_size_px + self.memory_stim_padding

                if cell_index == int(memory_gt_2[i]):
                    # draw.rounded_rectangle([cell_x, cell_y, cell_x+grid_cell_size_px, cell_y+grid_cell_size_px], 
                    #                 radius=self.mem_rect_radius,
                    #                 fill=self.color_map[self.active_cell_color], 
                    #                 outline=self.color_map[self.cell_outline_color])
                    
                    draw.ellipse([cell_x+self.marker_padding, cell_y+self.marker_padding, 
                                  cell_x+grid_cell_size_px-self.marker_padding, 
                                  cell_y+grid_cell_size_px-self.marker_padding], 
                          fill=self.color_map[self.marker_color])

                else:
                    # Do nothing
                    pass

            memory_stim_2_list.append(memory_stim_2)

        return memory_stim_1_list, memory_stim_2_list
    

    def draw_distractor_visual_stim(self, distractor_gt_items):
        distractor_stim_list = []

        for item in distractor_gt_items:
            stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
            draw = ImageDraw.Draw(stim)

            draw.rectangle([self.spatial_mem_border_padding, self.spatial_mem_border_padding, 
                            self.img_size[0]-self.spatial_mem_border_padding, self.img_size[1]-self.spatial_mem_border_padding],
                            outline=self.color_map['black'], 
                            fill=item, 
                            width=self.spatial_mem_border_width)

            distractor_stim_list.append(stim)

        return distractor_stim_list

    def draw_distractor_spatial_stim(self, distractor_gt_items, distractor_grid_size):
        distractor_stim_list = []

        for item in distractor_gt_items:
            stim = Image.new('RGB', self.img_size, color=self.color_map[self.bg_color])
            draw = ImageDraw.Draw(stim)

            grid_size_px = self.img_size[0] - 2*self.stim_img_padding
            grid_cell_size_px = grid_size_px // distractor_grid_size

            for cell in range(distractor_grid_size**2):
                cell_x = cell % distractor_grid_size
                cell_y = cell // distractor_grid_size

                cell_x_px = cell_x * grid_cell_size_px + self.stim_img_padding
                cell_y_px = cell_y * grid_cell_size_px + self.stim_img_padding

                if cell in item:
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
                    
            distractor_stim_list.append(stim)

        return distractor_stim_list