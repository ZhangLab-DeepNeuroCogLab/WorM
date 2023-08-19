from torch.utils.data import DataLoader
from torch.utils.data import random_split

from src.data.dataset import Spatial_Coordination_Dataset
from src.data.dataset import Spatial_Free_Recall_Dataset
from src.data.dataset import Spatial_Integration_Dataset
from src.data.dataset import Spatial_Memory_Updating_Dataset
from src.data.dataset import Spatial_Task_Switching_Dataset
from src.data.dataset import Complex_Span_Dataset
from src.data.dataset import Visual_Item_Recognition_Dataset
from src.data.dataset import Visual_Serial_Recall_Recognition_Dataset
from src.data.dataset import Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset


def get_multitask_dataloader(config):

    # SC Task
    SC_Task_TrainVal_Dataset = Spatial_Coordination_Dataset(data_path=config.data_folder, 
                                                            max_seq_len=config.max_seq_len,
                                                            grid_size=10, 
                                                            list_length_options=[10, 12, 14, 16, 18],
                                                            symmetry_offset_options=[2, 4, 6, 8],
                                                            gen_random_trials=True, 
                                                            held_out_list_lengths=[], 
                                                            num_samples=config.samples_per_task, 
                                                            img_size=config.img_size, 
                                                            rs_img_size=config.rs_img_size, 
                                                            write=True, show_gt_pattern=False,
                                                            split='train')
    
    SC_Task_Test_Dataset = Spatial_Coordination_Dataset(data_path=config.data_folder, 
                                                        max_seq_len=config.max_seq_len,
                                                        rs_img_size=config.rs_img_size,
                                                        show_gt_pattern=False,
                                                        split='test')
    
    if config.gen_test:
        SC_Task_Gen_Test_Dataset = Spatial_Coordination_Dataset(data_path=config.data_folder,
                                                                max_seq_len=config.max_seq_len,
                                                                rs_img_size=config.rs_img_size,
                                                                show_gt_pattern=False,
                                                                split='gen_test')
        
    SC_Task_Train_Dataset, SC_Task_Val_Dataset = random_split(
        SC_Task_TrainVal_Dataset, [int(0.9*len(SC_Task_TrainVal_Dataset)), int(0.1*len(SC_Task_TrainVal_Dataset))])
    
    sc_task_train_dataloader = DataLoader(SC_Task_Train_Dataset, batch_size=config.batch_size, 
                                          shuffle=True, drop_last=True, num_workers=config.num_workers)
    sc_task_val_dataloader = DataLoader(SC_Task_Val_Dataset, batch_size=config.batch_size, 
                                        shuffle=False, drop_last=False, num_workers=config.num_workers)
    sc_task_test_dataloader = DataLoader(SC_Task_Test_Dataset, batch_size=config.batch_size, 
                                         shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    if config.gen_test:
        sc_task_gen_test_dataloader = DataLoader(SC_Task_Gen_Test_Dataset, batch_size=config.batch_size,
                                                    shuffle=False, drop_last=False, num_workers=config.num_workers)

    # SFR Task
    SFR_Task_TrainVal_Dataset = Spatial_Free_Recall_Dataset(data_path=config.data_folder, 
                                                            max_seq_len=config.max_seq_len,
                                                            grid_size=10, 
                                                            set_size_options=[30], 
                                                            list_length_options=[1, 2, 3, 4, 5, 
                                                                                 6, 7, 8, 10, 12,
                                                                                 15, 18], 
                                                            gen_random_trials=True, 
                                                            held_out_set_sizes=[], 
                                                            held_out_list_lengths=[], 
                                                            num_samples=config.samples_per_task, 
                                                            img_size=config.img_size, 
                                                            rs_img_size=config.rs_img_size, 
                                                            write=True, split='train')
    
    SFR_Task_Test_Dataset = Spatial_Free_Recall_Dataset(data_path=config.data_folder,
                                                        max_seq_len=config.max_seq_len,
                                                        rs_img_size=config.rs_img_size,
                                                        split='test')
    
    if config.gen_test:
        SFR_Task_Gen_Test_Dataset = Spatial_Free_Recall_Dataset(data_path=config.data_folder, 
                                                                max_seq_len=config.max_seq_len,
                                                                rs_img_size=config.rs_img_size, 
                                                                split='gen_test')
    
    SFR_Task_Train_Dataset, SFR_Task_Val_Dataset = random_split(
        SFR_Task_TrainVal_Dataset, [int(0.9*len(SFR_Task_TrainVal_Dataset)), int(0.1*len(SFR_Task_TrainVal_Dataset))])
    
    sfr_task_train_dataloader = DataLoader(SFR_Task_Train_Dataset, batch_size=config.batch_size,
                                           shuffle=True, drop_last=True, num_workers=config.num_workers)
    sfr_task_val_dataloader = DataLoader(SFR_Task_Val_Dataset, batch_size=config.batch_size,
                                         shuffle=False, drop_last=False, num_workers=config.num_workers)
    sfr_task_test_dataloader = DataLoader(SFR_Task_Test_Dataset, batch_size=config.batch_size, 
                                          shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    if config.gen_test:
        sfr_task_gen_test_dataloader = DataLoader(SFR_Task_Gen_Test_Dataset, batch_size=config.batch_size, 
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    # SI Task
    SI_Task_TrainVal_Dataset = Spatial_Integration_Dataset(data_path=config.data_folder, 
                                                           max_seq_len=config.max_seq_len,
                                                           grid_size_options=[4], 
                                                           pattern_size_options=[12], 
                                                           part_size_options=[1, 2, 3, 
                                                                              4, 6, 12], 
                                                           distractor_difference_options=[1, 2, 3, 4], 
                                                           gen_random_trials=True, 
                                                           num_samples=config.samples_per_task, 
                                                           img_size=config.img_size, 
                                                           rs_img_size=config.rs_img_size, 
                                                           write=True, split='train')
    
    SI_Task_Test_Dataset = Spatial_Integration_Dataset(data_path=config.data_folder,
                                                       max_seq_len=config.max_seq_len,
                                                       rs_img_size=config.rs_img_size,
                                                       split='test')

    if config.gen_test:
        SI_Task_Gen_Test_Dataset = Spatial_Integration_Dataset(data_path=config.data_folder, 
                                                            max_seq_len=config.max_seq_len, 
                                                            rs_img_size=config.rs_img_size,
                                                            split='gen_test')
    
    SI_Task_Train_Dataset, SI_Task_Val_Dataset = random_split(
        SI_Task_TrainVal_Dataset, [int(0.9*len(SI_Task_TrainVal_Dataset)), int(0.1*len(SI_Task_TrainVal_Dataset))])
    
    si_task_train_dataloader = DataLoader(SI_Task_Train_Dataset, batch_size=config.batch_size,
                                            shuffle=True, drop_last=True, num_workers=config.num_workers)
    si_task_val_dataloader = DataLoader(SI_Task_Val_Dataset, batch_size=config.batch_size,
                                            shuffle=False, drop_last=False, num_workers=config.num_workers)
    si_task_test_dataloader = DataLoader(SI_Task_Test_Dataset, batch_size=config.batch_size,
                                            shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    if config.gen_test:
        si_task_gen_test_dataloader = DataLoader(SI_Task_Gen_Test_Dataset, batch_size=config.batch_size, 
                                                    shuffle=False, drop_last=False, num_workers=config.num_workers)
    

    # SMU Task
    SMU_Task_TrainVal_Dataset = Spatial_Memory_Updating_Dataset(data_path=config.data_folder, 
                                                                max_seq_len=config.max_seq_len,
                                                                grid_size=3, 
                                                                set_size_options=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                                presentation_time_options=[1], 
                                                                num_updates_options=[8],
                                                                gen_random_trials=True, 
                                                                held_out_set_sizes=[], 
                                                                num_samples=config.samples_per_task, 
                                                                img_size=config.img_size, 
                                                                rs_img_size=config.rs_img_size, 
                                                                write=True, split='train')
    
    SMU_Task_Test_Dataset = Spatial_Memory_Updating_Dataset(data_path=config.data_folder, 
                                                            max_seq_len=config.max_seq_len, 
                                                            rs_img_size=config.rs_img_size,
                                                            split='test')
    
    if config.gen_test:
        SMU_Task_Gen_Test_Dataset = Spatial_Memory_Updating_Dataset(data_path=config.data_folder, 
                                                                    max_seq_len=config.max_seq_len, 
                                                                    rs_img_size=config.rs_img_size,
                                                                    split='gen_test')
    
    SMU_Task_Train_Dataset, SMU_Task_Val_Dataset = random_split(
        SMU_Task_TrainVal_Dataset, [int(0.9*len(SMU_Task_TrainVal_Dataset)), int(0.1*len(SMU_Task_TrainVal_Dataset))])
    
    smu_task_train_dataloader = DataLoader(SMU_Task_Train_Dataset, batch_size=config.batch_size,
                                            shuffle=True, drop_last=True, num_workers=config.num_workers)
    smu_task_val_dataloader = DataLoader(SMU_Task_Val_Dataset, batch_size=config.batch_size,
                                            shuffle=False, drop_last=False, num_workers=config.num_workers)
    smu_task_test_dataloader = DataLoader(SMU_Task_Test_Dataset, batch_size=config.batch_size,
                                            shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    if config.gen_test:
        smu_task_gen_test_dataloader = DataLoader(SMU_Task_Gen_Test_Dataset, batch_size=config.batch_size, 
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    

    # STS Task
    STS_Task_TrainVal_Dataset = Spatial_Task_Switching_Dataset(data_path=config.data_folder, 
                                                                max_seq_len=config.max_seq_len,
                                                                variant='Cued', 
                                                                trial_lengths_options=[20], 
                                                                gen_random_trials=True, 
                                                                held_out_trial_lengths=[], 
                                                                num_samples=config.samples_per_task, 
                                                                img_size=config.img_size, 
                                                                rs_img_size=config.rs_img_size, 
                                                                write=True, split='train')
    
    STS_Task_Test_Dataset = Spatial_Task_Switching_Dataset(data_path=config.data_folder, 
                                                            max_seq_len=config.max_seq_len,
                                                            variant='Cued', 
                                                            rs_img_size=config.rs_img_size,
                                                            split='test')
    
    STS_Task_Gen_Test_Dataset = Spatial_Task_Switching_Dataset(data_path=config.data_folder, 
                                                                max_seq_len=config.max_seq_len,
                                                                variant='Cued',
                                                                rs_img_size=config.rs_img_size,
                                                                split='gen_test')
    
    STS_Task_Train_Dataset, STS_Task_Val_Dataset = random_split(
        STS_Task_TrainVal_Dataset, [int(0.9*len(STS_Task_TrainVal_Dataset)), int(0.1*len(STS_Task_TrainVal_Dataset))])
    
    sts_task_train_dataloader = DataLoader(STS_Task_Train_Dataset, batch_size=config.batch_size,
                                            shuffle=True, drop_last=True, num_workers=config.num_workers)
    sts_task_val_dataloader = DataLoader(STS_Task_Val_Dataset, batch_size=config.batch_size,
                                            shuffle=False, drop_last=False, num_workers=config.num_workers)
    sts_task_test_dataloader = DataLoader(STS_Task_Test_Dataset, batch_size=config.batch_size,
                                            shuffle=False, drop_last=False, num_workers=config.num_workers)
    sts_task_gen_test_dataloader = DataLoader(STS_Task_Gen_Test_Dataset, batch_size=config.batch_size,
                                                    shuffle=False, drop_last=False, num_workers=config.num_workers)
    

    # VIR Task
    VIR_Task_TrainVal_Dataset = Visual_Item_Recognition_Dataset(data_path=config.data_folder, 
                                                                            max_seq_len=config.max_seq_len,
                                                                            grid_size=6, 
                                                                            list_length_options=[4, 6, 8, 10], 
                                                                            distractor_difference_options=[4], 
                                                                            ri_options=[0, 2, 4, 5, 6], 
                                                                            gen_random_trials=True, 
                                                                            held_out_list_lengths=[], 
                                                                            num_samples=config.samples_per_task, 
                                                                            img_size=config.img_size,
                                                                            rs_img_size=config.rs_img_size, 
                                                                            write=True, split='train')

    VIR_Task_Test_Dataset = Visual_Item_Recognition_Dataset(data_path=config.data_folder, 
                                                                    max_seq_len=config.max_seq_len,
                                                                    rs_img_size=config.rs_img_size,
                                                                    split='test')

    if config.gen_test: 
        VIR_Task_Gen_Test_Dataset = Visual_Item_Recognition_Dataset(data_path=config.data_folder,
                                                                            max_seq_len=config.max_seq_len,
                                                                            rs_img_size=config.rs_img_size,
                                                                            split='gen_test')
        
    VIR_Task_Train_Dataset, VIR_Task_Val_Dataset = random_split(
        VIR_Task_TrainVal_Dataset, [int(0.9*len(VIR_Task_TrainVal_Dataset)), int(0.1*len(VIR_Task_TrainVal_Dataset))])
    
    vir_task_train_dataloader = DataLoader(VIR_Task_Train_Dataset, batch_size=config.batch_size,
                                                shuffle=True, drop_last=True, num_workers=config.num_workers)
    vir_task_val_dataloader = DataLoader(VIR_Task_Val_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    vir_task_test_dataloader = DataLoader(VIR_Task_Test_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    if config.gen_test:
        vir_task_gen_test_dataloader = DataLoader(VIR_Task_Gen_Test_Dataset, batch_size=config.batch_size,
                                                            shuffle=False, drop_last=False, num_workers=config.num_workers)


    # VSR Task
    VSR_Task_TrainVal_Dataset = Visual_Serial_Recall_Recognition_Dataset(data_path=config.data_folder, 
                                                                         max_seq_len=config.max_seq_len,
                                                                         probe_variant='Recall', 
                                                                         grid_size=6, 
                                                                         list_length_options=[2, 3, 4, 5, 
                                                                                              6, 7, 8, 9], 
                                                                         distractor_difference_options=[], 
                                                                         gen_random_trials=True, 
                                                                         held_out_list_lengths=[], 
                                                                         num_samples=config.samples_per_task, 
                                                                         img_size=config.img_size, 
                                                                         rs_img_size=config.rs_img_size, 
                                                                         write=True, split='train')
    
    VSR_Task_Test_Dataset = Visual_Serial_Recall_Recognition_Dataset(data_path=config.data_folder, 
                                                                     max_seq_len=config.max_seq_len,
                                                                     probe_variant='Recall', 
                                                                     rs_img_size=config.rs_img_size,
                                                                     split='test')
    
    if config.gen_test:
        VSR_Task_Gen_Test_Dataset = Visual_Serial_Recall_Recognition_Dataset(data_path=config.data_folder, 
                                                                            max_seq_len=config.max_seq_len,
                                                                            probe_variant='Recall', 
                                                                            rs_img_size=config.rs_img_size,
                                                                            split='gen_test')
        
    VSR_Task_Train_Dataset, VSR_Task_Val_Dataset = random_split(
        VSR_Task_TrainVal_Dataset, [int(0.9*len(VSR_Task_TrainVal_Dataset)), int(0.1*len(VSR_Task_TrainVal_Dataset))])
    
    vsr_task_train_dataloader = DataLoader(VSR_Task_Train_Dataset, batch_size=config.batch_size,
                                           shuffle=True, drop_last=True, num_workers=config.num_workers)
    vsr_task_val_dataloader = DataLoader(VSR_Task_Val_Dataset, batch_size=config.batch_size,
                                             shuffle=False, drop_last=False, num_workers=config.num_workers)
    vsr_task_test_dataloader = DataLoader(VSR_Task_Test_Dataset, batch_size=config.batch_size, 
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    if config.gen_test:
        vsr_task_gen_test_dataloader = DataLoader(VSR_Task_Gen_Test_Dataset, batch_size=config.batch_size, 
                                                    shuffle=False, drop_last=False, num_workers=config.num_workers)
    

    # VSRec Task
    VSRec_Task_TrainVal_Dataset = Visual_Serial_Recall_Recognition_Dataset(data_path=config.data_folder, 
                                                                           max_seq_len=config.max_seq_len,
                                                                           probe_variant='Recognition', grid_size=6, 
                                                                           list_length_options=[2, 3, 4, 5, 
                                                                                                6, 7, 8, 9], 
                                                                           distractor_difference_options=[2, 4, 6, 8, 10], 
                                                                           gen_random_trials=True, 
                                                                           held_out_list_lengths=[], 
                                                                           num_samples=config.samples_per_task, 
                                                                           img_size=config.img_size, 
                                                                           rs_img_size=config.rs_img_size, 
                                                                           write=True, split='train')
    
    VSRec_Task_Test_Dataset = Visual_Serial_Recall_Recognition_Dataset(data_path=config.data_folder, 
                                                                       max_seq_len=config.max_seq_len, 
                                                                       probe_variant='Recognition', 
                                                                       rs_img_size=config.rs_img_size,
                                                                       split='test')
    
    if config.gen_test:
        VSRec_Task_Gen_Test_Dataset = Visual_Serial_Recall_Recognition_Dataset(data_path=config.data_folder, 
                                                                            max_seq_len=config.max_seq_len, 
                                                                            probe_variant='Recognition', 
                                                                            rs_img_size=config.rs_img_size,
                                                                            split='gen_test')

    VSRec_Task_Train_Dataset, VSRec_Task_Val_Dataset = random_split(
        VSRec_Task_TrainVal_Dataset, [int(0.9*len(VSRec_Task_TrainVal_Dataset)), int(0.1*len(VSRec_Task_TrainVal_Dataset))])
    
    vsrec_task_train_dataloader = DataLoader(VSRec_Task_Train_Dataset, batch_size=config.batch_size,
                                                shuffle=True, drop_last=True, num_workers=config.num_workers)
    vsrec_task_val_dataloader = DataLoader(VSRec_Task_Val_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    vsrec_task_test_dataloader = DataLoader(VSRec_Task_Test_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    if config.gen_test:
        vsrec_task_gen_test_dataloader = DataLoader(VSRec_Task_Gen_Test_Dataset, batch_size=config.batch_size, 
                                                    shuffle=False, drop_last=False, num_workers=config.num_workers)
    

    # CD Color Task
    CD_Color_Task_TrainVal_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder, 
                                                                                                    max_seq_len=config.max_seq_len, 
                                                                                                    variant='Color', 
                                                                                                    set_size_options=[2, 4, 6, 8, 10, 12], 
                                                                                                    ri_options=[0, 6, 12, 18], 
                                                                                                    gen_random_trials=True, 
                                                                                                    held_out_set_size_options=[], 
                                                                                                    held_out_ri_options=[], 
                                                                                                    num_samples=config.samples_per_task, 
                                                                                                    img_size=config.img_size, 
                                                                                                    rs_img_size=config.rs_img_size, 
                                                                                                    write=True, split='train')
    
    CD_Color_Task_Test_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                    max_seq_len=config.max_seq_len, 
                                                                                                    variant='Color', 
                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                    split='test')
    

    if config.gen_test:
        CD_Color_Task_Gen_Test_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder, 
                                                                                                            max_seq_len=config.max_seq_len,
                                                                                                            variant='Color',
                                                                                                            rs_img_size=config.rs_img_size,
                                                                                                            split='gen_test')
        
    CD_Color_Task_Train_Dataset, CD_Color_Task_Val_Dataset = random_split(
        CD_Color_Task_TrainVal_Dataset, [int(0.9*len(CD_Color_Task_TrainVal_Dataset)), int(0.1*len(CD_Color_Task_TrainVal_Dataset))])
    
    cd_color_task_train_dataloader = DataLoader(CD_Color_Task_Train_Dataset, batch_size=config.batch_size, 
                                                shuffle=True, drop_last=True, num_workers=config.num_workers)
    cd_color_task_val_dataloader = DataLoader(CD_Color_Task_Val_Dataset, batch_size=config.batch_size, 
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    cd_color_task_test_dataloader = DataLoader(CD_Color_Task_Test_Dataset, batch_size=config.batch_size, 
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    if config.gen_test:
        cd_color_task_gen_test_dataloader = DataLoader(CD_Color_Task_Gen_Test_Dataset, batch_size=config.batch_size, 
                                                    shuffle=False, drop_last=False, num_workers=config.num_workers)
    

    # CD Orientation Task
    CD_Orientation_Task_TrainVal_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                            max_seq_len=config.max_seq_len,
                                                                                                            variant='Orientation',
                                                                                                            set_size_options=[2, 4, 6, 8, 10, 12],
                                                                                                            ri_options=[0, 6, 12, 18],
                                                                                                            gen_random_trials=True,
                                                                                                            held_out_set_size_options=[],
                                                                                                            held_out_ri_options=[],
                                                                                                            num_samples=config.samples_per_task,
                                                                                                            img_size=config.img_size, 
                                                                                                            rs_img_size=config.rs_img_size,
                                                                                                            write=True, split='train')
    
    CD_Orientation_Task_Test_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                            max_seq_len=config.max_seq_len,
                                                                                                            variant='Orientation',
                                                                                                            rs_img_size=config.rs_img_size,
                                                                                                            split='test')
    
    if config.gen_test:
        CD_Orientation_Task_Gen_Test_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                                    max_seq_len=config.max_seq_len,
                                                                                                                    variant='Orientation',
                                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                                    split='gen_test')
        
    CD_Orientation_Task_Train_Dataset, CD_Orientation_Task_Val_Dataset = random_split(
        CD_Orientation_Task_TrainVal_Dataset, [int(0.9*len(CD_Orientation_Task_TrainVal_Dataset)), int(0.1*len(CD_Orientation_Task_TrainVal_Dataset))])
    
    cd_orientation_task_train_dataloader = DataLoader(CD_Orientation_Task_Train_Dataset, batch_size=config.batch_size,
                                                shuffle=True, drop_last=True, num_workers=config.num_workers)
    cd_orientation_task_val_dataloader = DataLoader(CD_Orientation_Task_Val_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    cd_orientation_task_test_dataloader = DataLoader(CD_Orientation_Task_Test_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    if config.gen_test:
        cd_orientation_task_gen_test_dataloader = DataLoader(CD_Orientation_Task_Gen_Test_Dataset, batch_size=config.batch_size,
                                                    shuffle=False, drop_last=False, num_workers=config.num_workers)
    

    # CD Size Task
    CD_Size_Task_TrainVal_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                    max_seq_len=config.max_seq_len,
                                                                                                    variant='Size',
                                                                                                    set_size_options=[2, 4, 6, 8, 10, 12],
                                                                                                    ri_options=[0, 6, 12, 18],
                                                                                                    gen_random_trials=True,
                                                                                                    held_out_set_size_options=[],
                                                                                                    held_out_ri_options=[],
                                                                                                    num_samples=config.samples_per_task,
                                                                                                    img_size=config.img_size, 
                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                    write=True, split='train')
    
    CD_Size_Task_Test_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                    max_seq_len=config.max_seq_len,
                                                                                                    variant='Size',
                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                    split='test')
    
    if config.gen_test:
        CD_Size_Task_Gen_Test_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                            max_seq_len=config.max_seq_len,
                                                                                                            variant='Size',
                                                                                                            rs_img_size=config.rs_img_size,
                                                                                                            split='gen_test')
        
    CD_Size_Task_Train_Dataset, CD_Size_Task_Val_Dataset = random_split(
        CD_Size_Task_TrainVal_Dataset, [int(0.9*len(CD_Size_Task_TrainVal_Dataset)), int(0.1*len(CD_Size_Task_TrainVal_Dataset))])
    
    cd_size_task_train_dataloader = DataLoader(CD_Size_Task_Train_Dataset, batch_size=config.batch_size,
                                                shuffle=True, drop_last=True, num_workers=config.num_workers)
    cd_size_task_val_dataloader = DataLoader(CD_Size_Task_Val_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    cd_size_task_test_dataloader = DataLoader(CD_Size_Task_Test_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    if config.gen_test:
        cd_size_task_gen_test_dataloader = DataLoader(CD_Size_Task_Gen_Test_Dataset, batch_size=config.batch_size,
                                                    shuffle=False, drop_last=False, num_workers=config.num_workers)
    

    # CD Gap Task
    CD_Gap_Task_TrainVal_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                    max_seq_len=config.max_seq_len,
                                                                                                    variant='Gap',
                                                                                                    set_size_options=[2, 4, 6, 8, 10, 12],
                                                                                                    ri_options=[0, 6, 12, 18],
                                                                                                    gen_random_trials=True,
                                                                                                    held_out_set_size_options=[],
                                                                                                    held_out_ri_options=[],
                                                                                                    num_samples=config.samples_per_task,
                                                                                                    img_size=config.img_size, 
                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                    write=True, split='train')
    
    CD_Gap_Task_Test_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                    max_seq_len=config.max_seq_len,
                                                                                                    variant='Gap',
                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                    split='test')
    
    if config.gen_test:
        CD_Gap_Task_Gen_Test_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                            max_seq_len=config.max_seq_len,
                                                                                                            variant='Gap',
                                                                                                            rs_img_size=config.rs_img_size,
                                                                                                            split='gen_test')
    
    CD_Gap_Task_Train_Dataset, CD_Gap_Task_Val_Dataset = random_split(
        CD_Gap_Task_TrainVal_Dataset, [int(0.9*len(CD_Gap_Task_TrainVal_Dataset)), int(0.1*len(CD_Gap_Task_TrainVal_Dataset))])
    
    cd_gap_task_train_dataloader = DataLoader(CD_Gap_Task_Train_Dataset, batch_size=config.batch_size,
                                                shuffle=True, drop_last=True, num_workers=config.num_workers)
    cd_gap_task_val_dataloader = DataLoader(CD_Gap_Task_Val_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    cd_gap_task_test_dataloader = DataLoader(CD_Gap_Task_Test_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    if config.gen_test:
        cd_gap_task_gen_test_dataloader = DataLoader(CD_Gap_Task_Gen_Test_Dataset, batch_size=config.batch_size,
                                                    shuffle=False, drop_last=False, num_workers=config.num_workers)
    

    # CD Conjunction Task
    CD_Conj_Task_TrainVal_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                    max_seq_len=config.max_seq_len,
                                                                                                    variant='Conjunction',
                                                                                                    set_size_options=[2, 4, 6, 8, 10, 12],
                                                                                                    ri_options=[0, 6, 12, 18],
                                                                                                    gen_random_trials=True,
                                                                                                    held_out_set_size_options=[],
                                                                                                    held_out_ri_options=[],
                                                                                                    num_samples=config.samples_per_task,
                                                                                                    img_size=config.img_size, 
                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                    write=True, split='train')
    
    CD_Conj_Task_Test_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                    max_seq_len=config.max_seq_len,
                                                                                                    variant='Conjunction',
                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                    split='test')
    
    if config.gen_test:
        CD_Conj_Task_Gen_Test_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                            max_seq_len=config.max_seq_len,
                                                                                                            variant='Conjunction',
                                                                                                            rs_img_size=config.rs_img_size,
                                                                                                            split='gen_test')
        
    CD_Conj_Task_Train_Dataset, CD_Conj_Task_Val_Dataset = random_split(
        CD_Conj_Task_TrainVal_Dataset, [int(0.9*len(CD_Conj_Task_TrainVal_Dataset)), int(0.1*len(CD_Conj_Task_TrainVal_Dataset))])
    
    cd_conj_task_train_dataloader = DataLoader(CD_Conj_Task_Train_Dataset, batch_size=config.batch_size,
                                                shuffle=True, drop_last=True, num_workers=config.num_workers)
    cd_conj_task_val_dataloader = DataLoader(CD_Conj_Task_Val_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    cd_conj_task_test_dataloader = DataLoader(CD_Conj_Task_Test_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    if config.gen_test:
        cd_conj_task_gen_test_dataloader = DataLoader(CD_Conj_Task_Gen_Test_Dataset, batch_size=config.batch_size,
                                                    shuffle=False, drop_last=False, num_workers=config.num_workers)


    # CS Task
    CS_Task_TrainVal_Dataset = CS_Dataset(data_path=config.data_folder, 
                                                          max_seq_len=config.max_seq_len,
                                                          num_storage_options=[2], 
                                                          num_distractor_options=[0, 1, 3, 5], 
                                                          visual_memory_grid_size=4, 
                                                          spatial_distractor_grid_size_options=[10], 
                                                          spatial_distractor_set_size_options=[20], 
                                                          spatial_distractor_symmetry_offset_options=[4], 
                                                          gen_random_trials=True, 
                                                          num_samples=config.samples_per_task, 
                                                          img_size=config.img_size, 
                                                          rs_img_size=config.rs_img_size, 
                                                          write=True, split='train')
    
    CS_Task_Test_Dataset = CS_Dataset(data_path=config.data_folder, 
                                                      max_seq_len=config.max_seq_len,
                                                      rs_img_size=config.rs_img_size,
                                                      split='test')
    
    if config.gen_test:
        CS_Task_Gen_Test_Dataset = CS_Dataset(data_path=config.data_folder, 
                                                              max_seq_len=config.max_seq_len,
                                                              rs_img_size=config.rs_img_size,
                                                              split='gen_test')
        
    CS_Task_Train_Dataset, CS_Task_Val_Dataset = random_split(
        CS_Task_TrainVal_Dataset, [int(0.9*len(CS_Task_TrainVal_Dataset)), int(0.1*len(CS_Task_TrainVal_Dataset))])
    
    cs_task_train_dataloader = DataLoader(CS_Task_Train_Dataset, batch_size=config.batch_size,
                                           shuffle=True, drop_last=True, num_workers=config.num_workers)
    cs_task_val_dataloader = DataLoader(CS_Task_Val_Dataset, batch_size=config.batch_size,
                                             shuffle=False, drop_last=False, num_workers=config.num_workers)
    cs_task_test_dataloader = DataLoader(CS_Task_Test_Dataset, batch_size=config.batch_size, 
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    if config.gen_test:
        cs_task_gen_test_dataloader = DataLoader(CS_Task_Gen_Test_Dataset, batch_size=config.batch_size, 
                                                    shuffle=False, drop_last=False, num_workers=config.num_workers)


    multitask_train_dataloader = {
        'SC_Task': sc_task_train_dataloader,
        'SFR_Task': sfr_task_train_dataloader,
        'SI_Task': si_task_train_dataloader, 
        'SMU_Task': smu_task_train_dataloader, 
        'STS_Task': sts_task_train_dataloader, 
        'VIR_Task': vir_task_train_dataloader,
        'VSR_Task': vsr_task_train_dataloader,
        'VSRec_Task': vsrec_task_train_dataloader, 
        'CD_Color_Task': cd_color_task_train_dataloader, 
        'CD_Orientation_Task': cd_orientation_task_train_dataloader, 
        'CD_Size_Task': cd_size_task_train_dataloader, 
        'CD_Gap_Task': cd_gap_task_train_dataloader, 
        'CD_Conj_Task': cd_conj_task_train_dataloader, 
        'CS_Task': cs_task_train_dataloader
    }

    multitask_val_dataloader = {
        'SC_Task': sc_task_val_dataloader, 
        'SFR_Task': sfr_task_val_dataloader, 
        'SI_Task': si_task_val_dataloader, 
        'SMU_Task': smu_task_val_dataloader, 
        'STS_Task': sts_task_val_dataloader, 
        'VIR_Task': vir_task_val_dataloader, 
        'VSR_Task': vsr_task_val_dataloader, 
        'VSRec_Task': vsrec_task_val_dataloader, 
        'CD_Color_Task': cd_color_task_val_dataloader, 
        'CD_Orientation_Task': cd_orientation_task_val_dataloader, 
        'CD_Size_Task': cd_size_task_val_dataloader, 
        'CD_Gap_Task': cd_gap_task_val_dataloader, 
        'CD_Conj_Task': cd_conj_task_val_dataloader, 
        'CS_Task': cs_task_val_dataloader
    }

    multitask_test_dataloader = {
        'SC_Task': sc_task_test_dataloader, 
        'SFR_Task': sfr_task_test_dataloader, 
        'SI_Task': si_task_test_dataloader, 
        'SMU_Task': smu_task_test_dataloader, 
        'STS_Task': sts_task_test_dataloader, 
        'VIR_Task': vir_task_test_dataloader, 
        'VSR_Task': vsr_task_test_dataloader, 
        'VSRec_Task': vsrec_task_test_dataloader, 
        'CD_Color_Task': cd_color_task_test_dataloader, 
        'CD_Orientation_Task': cd_orientation_task_test_dataloader, 
        'CD_Size_Task': cd_size_task_test_dataloader, 
        'CD_Gap_Task': cd_gap_task_test_dataloader, 
        'CD_Conj_Task': cd_conj_task_test_dataloader, 
        'CS_Task': cs_task_test_dataloader
    }

    if config.gen_test:
        multitask_gen_test_dataloader = {
            'SC_Task': sc_task_gen_test_dataloader, 
            'SFR_Task': sfr_task_gen_test_dataloader, 
            'SI_Task': si_task_gen_test_dataloader, 
            'SMU_Task': smu_task_gen_test_dataloader, 
            'STS_Task': sts_task_gen_test_dataloader, 
            'VIR_Task': vir_task_gen_test_dataloader, 
            'VSR_Task': vsr_task_gen_test_dataloader, 
            'VSRec_Task': vsrec_task_gen_test_dataloader, 
            'CD_Color_Task': cd_color_task_gen_test_dataloader, 
            'CD_Orientation_Task': cd_orientation_task_gen_test_dataloader, 
            'CD_Size_Task': cd_size_task_gen_test_dataloader, 
            'CD_Gap_Task': cd_gap_task_gen_test_dataloader, 
            'CD_Conj_Task': cd_conj_task_gen_test_dataloader, 
            'CS_Task': cs_task_gen_test_dataloader
        }

    if config.gen_test:
        return multitask_train_dataloader, multitask_val_dataloader, multitask_test_dataloader, multitask_gen_test_dataloader
    else:
        return multitask_train_dataloader, multitask_val_dataloader, multitask_test_dataloader
    

def get_train_multitask_dataloader(config):

    # SC Task
    SC_Task_Train_Dataset = Spatial_Coordination_Dataset(data_path=config.data_folder, 
                                                        max_seq_len=config.max_seq_len,
                                                        rs_img_size=config.rs_img_size,
                                                        show_gt_pattern=False,
                                                        split='train')

    sc_task_train_dataloader = DataLoader(SC_Task_Train_Dataset, batch_size=config.batch_size, 
                                         shuffle=False, drop_last=False, num_workers=config.num_workers)

    # SFR Task    
    SFR_Task_Train_Dataset = Spatial_Free_Recall_Dataset(data_path=config.data_folder,
                                                        max_seq_len=config.max_seq_len,
                                                        rs_img_size=config.rs_img_size,
                                                        split='train')

    sfr_task_train_dataloader = DataLoader(SFR_Task_Train_Dataset, batch_size=config.batch_size, 
                                          shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    # SI Task
    SI_Task_Train_Dataset = Spatial_Integration_Dataset(data_path=config.data_folder,
                                                       max_seq_len=config.max_seq_len,
                                                       rs_img_size=config.rs_img_size,
                                                       split='train')

    si_task_train_dataloader = DataLoader(SI_Task_Train_Dataset, batch_size=config.batch_size,
                                            shuffle=False, drop_last=False, num_workers=config.num_workers)    

    # SMU Task
    SMU_Task_Train_Dataset = Spatial_Memory_Updating_Dataset(data_path=config.data_folder, 
                                                            max_seq_len=config.max_seq_len, 
                                                            rs_img_size=config.rs_img_size,
                                                            split='train')
    
    smu_task_train_dataloader = DataLoader(SMU_Task_Train_Dataset, batch_size=config.batch_size,
                                            shuffle=False, drop_last=False, num_workers=config.num_workers)

    # STS Task
    STS_Task_Train_Dataset = Spatial_Task_Switching_Dataset(data_path=config.data_folder, 
                                                            max_seq_len=config.max_seq_len,
                                                            variant='Cued', 
                                                            rs_img_size=config.rs_img_size,
                                                            split='train')
    
    sts_task_train_dataloader = DataLoader(STS_Task_Train_Dataset, batch_size=config.batch_size,
                                            shuffle=False, drop_last=False, num_workers=config.num_workers)    

    # VIR Task
    VIR_Task_Train_Dataset = Visual_Item_Recognition_Dataset(data_path=config.data_folder, 
                                                                    max_seq_len=config.max_seq_len,
                                                                    rs_img_size=config.rs_img_size,
                                                                    split='train')

    vir_task_train_dataloader = DataLoader(VIR_Task_Train_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    # VSR Task
    VSR_Task_Train_Dataset = Visual_Serial_Recall_Recognition_Dataset(data_path=config.data_folder, 
                                                                     max_seq_len=config.max_seq_len,
                                                                     probe_variant='Recall', 
                                                                     rs_img_size=config.rs_img_size,
                                                                     split='train')
    
    vsr_task_train_dataloader = DataLoader(VSR_Task_Train_Dataset, batch_size=config.batch_size, 
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)    

    # VSRec Task
    VSRec_Task_Train_Dataset = Visual_Serial_Recall_Recognition_Dataset(data_path=config.data_folder, 
                                                                       max_seq_len=config.max_seq_len, 
                                                                       probe_variant='Recognition', 
                                                                       rs_img_size=config.rs_img_size,
                                                                       split='train')
    
    vsrec_task_train_dataloader = DataLoader(VSRec_Task_Train_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)

    # CD Color Task
    CD_Color_Task_Train_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                    max_seq_len=config.max_seq_len, 
                                                                                                    variant='Color', 
                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                    split='train')
    
    cd_color_task_train_dataloader = DataLoader(CD_Color_Task_Train_Dataset, batch_size=config.batch_size, 
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    # CD Orientation Task
    CD_Orientation_Task_Train_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                            max_seq_len=config.max_seq_len,
                                                                                                            variant='Orientation',
                                                                                                            rs_img_size=config.rs_img_size,
                                                                                                            split='train')
    
    cd_orientation_task_train_dataloader = DataLoader(CD_Orientation_Task_Train_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)

    # CD Size Task
    CD_Size_Task_Train_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                    max_seq_len=config.max_seq_len,
                                                                                                    variant='Size',
                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                    split='train')
    
    cd_size_task_train_dataloader = DataLoader(CD_Size_Task_Train_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers) 

    # CD Gap Task
    CD_Gap_Task_Train_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                    max_seq_len=config.max_seq_len,
                                                                                                    variant='Gap',
                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                    split='train')
    
    cd_gap_task_train_dataloader = DataLoader(CD_Gap_Task_Train_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    # CD Conjunction Task
    CD_Conj_Task_Train_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                    max_seq_len=config.max_seq_len,
                                                                                                    variant='Conjunction',
                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                    split='train')
    
    cd_conj_task_train_dataloader = DataLoader(CD_Conj_Task_Train_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)

    # CS Task
    CS_Task_Train_Dataset = CS_Dataset(data_path=config.data_folder, 
                                                      max_seq_len=config.max_seq_len,
                                                      rs_img_size=config.rs_img_size,
                                                      split='train')
    
    cs_task_train_dataloader = DataLoader(CS_Task_Train_Dataset, batch_size=config.batch_size, 
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    
    multitask_train_dataloader = {
        'SC_Task': sc_task_train_dataloader, 
        'SFR_Task': sfr_task_train_dataloader, 
        'SI_Task': si_task_train_dataloader, 
        'SMU_Task': smu_task_train_dataloader, 
        'STS_Task': sts_task_train_dataloader, 
        'VIR_Task': vir_task_train_dataloader, 
        'VSR_Task': vsr_task_train_dataloader, 
        'VSRec_Task': vsrec_task_train_dataloader, 
        'CD_Color_Task': cd_color_task_train_dataloader, 
        'CD_Orientation_Task': cd_orientation_task_train_dataloader, 
        'CD_Size_Task': cd_size_task_train_dataloader, 
        'CD_Gap_Task': cd_gap_task_train_dataloader, 
        'CD_Conj_Task': cd_conj_task_train_dataloader, 
        'CS_Task': cs_task_train_dataloader
    }

    return multitask_train_dataloader


def get_test_multitask_dataloader(config):

    # SC Task
    SC_Task_Test_Dataset = Spatial_Coordination_Dataset(data_path=config.data_folder, 
                                                        max_seq_len=config.max_seq_len,
                                                        rs_img_size=config.rs_img_size,
                                                        show_gt_pattern=False,
                                                        split='test')

    sc_task_test_dataloader = DataLoader(SC_Task_Test_Dataset, batch_size=config.batch_size, 
                                         shuffle=False, drop_last=False, num_workers=config.num_workers)

    # SFR Task    
    SFR_Task_Test_Dataset = Spatial_Free_Recall_Dataset(data_path=config.data_folder,
                                                        max_seq_len=config.max_seq_len,
                                                        rs_img_size=config.rs_img_size,
                                                        split='test')

    sfr_task_test_dataloader = DataLoader(SFR_Task_Test_Dataset, batch_size=config.batch_size, 
                                          shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    # SI Task
    SI_Task_Test_Dataset = Spatial_Integration_Dataset(data_path=config.data_folder,
                                                       max_seq_len=config.max_seq_len,
                                                       rs_img_size=config.rs_img_size,
                                                       split='test')

    si_task_test_dataloader = DataLoader(SI_Task_Test_Dataset, batch_size=config.batch_size,
                                            shuffle=False, drop_last=False, num_workers=config.num_workers)    

    # SMU Task
    SMU_Task_Test_Dataset = Spatial_Memory_Updating_Dataset(data_path=config.data_folder, 
                                                            max_seq_len=config.max_seq_len, 
                                                            rs_img_size=config.rs_img_size,
                                                            split='test')
    
    smu_task_test_dataloader = DataLoader(SMU_Task_Test_Dataset, batch_size=config.batch_size,
                                            shuffle=False, drop_last=False, num_workers=config.num_workers)

    # STS Task
    STS_Task_Test_Dataset = Spatial_Task_Switching_Dataset(data_path=config.data_folder, 
                                                            max_seq_len=config.max_seq_len,
                                                            variant='Cued', 
                                                            rs_img_size=config.rs_img_size,
                                                            split='test')
    
    sts_task_test_dataloader = DataLoader(STS_Task_Test_Dataset, batch_size=config.batch_size,
                                            shuffle=False, drop_last=False, num_workers=config.num_workers)    

    # VIR Task
    VIR_Task_Test_Dataset = Visual_Item_Recognition_Dataset(data_path=config.data_folder, 
                                                                    max_seq_len=config.max_seq_len,
                                                                    rs_img_size=config.rs_img_size,
                                                                    split='test')

    vir_task_test_dataloader = DataLoader(VIR_Task_Test_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    # VSR Task
    VSR_Task_Test_Dataset = Visual_Serial_Recall_Recognition_Dataset(data_path=config.data_folder, 
                                                                     max_seq_len=config.max_seq_len,
                                                                     probe_variant='Recall', 
                                                                     rs_img_size=config.rs_img_size,
                                                                     split='test')
    
    vsr_task_test_dataloader = DataLoader(VSR_Task_Test_Dataset, batch_size=config.batch_size, 
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)    

    # VSRec Task
    VSRec_Task_Test_Dataset = Visual_Serial_Recall_Recognition_Dataset(data_path=config.data_folder, 
                                                                       max_seq_len=config.max_seq_len, 
                                                                       probe_variant='Recognition', 
                                                                       rs_img_size=config.rs_img_size,
                                                                       split='test')
    
    vsrec_task_test_dataloader = DataLoader(VSRec_Task_Test_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)

    # CD Color Task
    CD_Color_Task_Test_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                    max_seq_len=config.max_seq_len, 
                                                                                                    variant='Color', 
                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                    split='test')
    
    cd_color_task_test_dataloader = DataLoader(CD_Color_Task_Test_Dataset, batch_size=config.batch_size, 
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    # CD Orientation Task
    CD_Orientation_Task_Test_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                            max_seq_len=config.max_seq_len,
                                                                                                            variant='Orientation',
                                                                                                            rs_img_size=config.rs_img_size,
                                                                                                            split='test')
    
    cd_orientation_task_test_dataloader = DataLoader(CD_Orientation_Task_Test_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)

    # CD Size Task
    CD_Size_Task_Test_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                    max_seq_len=config.max_seq_len,
                                                                                                    variant='Size',
                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                    split='test')
    
    cd_size_task_test_dataloader = DataLoader(CD_Size_Task_Test_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers) 

    # CD Gap Task
    CD_Gap_Task_Test_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                    max_seq_len=config.max_seq_len,
                                                                                                    variant='Gap',
                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                    split='test')
    
    cd_gap_task_test_dataloader = DataLoader(CD_Gap_Task_Test_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    # CD Conjunction Task
    CD_Conj_Task_Test_Dataset = Color_Orientation_Size_Gap_Conjunction_Change_Detection_Dataset(data_path=config.data_folder,
                                                                                                    max_seq_len=config.max_seq_len,
                                                                                                    variant='Conjunction',
                                                                                                    rs_img_size=config.rs_img_size,
                                                                                                    split='test')
    
    cd_conj_task_test_dataloader = DataLoader(CD_Conj_Task_Test_Dataset, batch_size=config.batch_size,
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)

    # CS Task
    CS_Task_Test_Dataset = CS_Dataset(data_path=config.data_folder, 
                                                      max_seq_len=config.max_seq_len,
                                                      rs_img_size=config.rs_img_size,
                                                      split='test')
    
    cs_task_test_dataloader = DataLoader(CS_Task_Test_Dataset, batch_size=config.batch_size, 
                                                shuffle=False, drop_last=False, num_workers=config.num_workers)
    
    
    multitask_test_dataloader = {
        'SC_Task': sc_task_test_dataloader, 
        'SFR_Task': sfr_task_test_dataloader, 
        'SI_Task': si_task_test_dataloader, 
        'SMU_Task': smu_task_test_dataloader, 
        'STS_Task': sts_task_test_dataloader, 
        'VIR_Task': vir_task_test_dataloader, 
        'VSR_Task': vsr_task_test_dataloader, 
        'VSRec_Task': vsrec_task_test_dataloader, 
        'CD_Color_Task': cd_color_task_test_dataloader, 
        'CD_Orientation_Task': cd_orientation_task_test_dataloader, 
        'CD_Size_Task': cd_size_task_test_dataloader, 
        'CD_Gap_Task': cd_gap_task_test_dataloader, 
        'CD_Conj_Task': cd_conj_task_test_dataloader, 
        'CS_Task': cs_task_test_dataloader
    }

    return multitask_test_dataloader
