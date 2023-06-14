import argparse

def build_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", type=str, default="./wm_bench_data/")
    parser.add_argument("--log_folder", type=str, default="./log/")
    parser.add_argument("--model_folder", type=str, default="./model/")
    parser.add_argument("--output_folder", type=str, default="./output/")

    parser.add_argument("--stage", type=str, default="Train")
    parser.add_argument("--num_tasks", type=int, default=14)
    parser.add_argument("--gen_test", type=int, default=0)

    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--resume_epoch", type=int, default=0)
    parser.add_argument("--resume_run_name", type=str, default="")
    parser.add_argument("--resume_wandb_id", type=str, default="")
    
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--rs_img_size", type=int, default=32)
    parser.add_argument("--num_input_channels", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--task_embedding", type=str, default="Learned")
    parser.add_argument("--task_embedding_given", type=str, default="All_TS")

    parser.add_argument("--use_cnn", type=int, default=1)
    parser.add_argument("--mem_architecture", type=str, default="GRU")
    parser.add_argument("--mem_input_size", type=int, default=512)
    parser.add_argument("--mem_hidden_size", type=int, default=96)
    parser.add_argument("--mem_num_layers", type=int, default=1)
    parser.add_argument("--trf_dim_ff", type=int, default=2048)

    parser.add_argument("--projection", type=str, default="linear")
    parser.add_argument("--classifier", type=str, default="linear")

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--samples_per_task", type=int, default=96000)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_extracted_feats", type=int, default=0)
    parser.add_argument("--test_interval", type=int, default=5)

    return parser