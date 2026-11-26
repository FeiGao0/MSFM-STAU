import os
import argparse
import numpy as np
import torch
import random
from core.data_provider import datasets_factory
from core.models.model_factory import Model
import core.trainer as trainer

#Set random number seed
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='STAU')
parser.add_argument('--dataset', type=str, default='tec')
parser.add_argument('--is_training', type=str, default='True')#False
args_main = parser.parse_args()
args_main.tied = True

if args_main.dataset == 'mnist':
    from configs.mnist_configs import configs
elif args_main.dataset == 'tec':
    from configs.tec_configs import configs

parser_dataset = configs()
parser_dataset.add_argument('--device', type=str, default='cuda')
args = parser_dataset.parse_args()
args.tied = True
if args_main.is_training == 'True':
    args.is_training = 'True'
else:
    args.is_training = 'False'


def schedule_sampling(eta, itr, channel, batch_size):
    zeros = np.zeros((batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    #print('eta: ', eta)
    random_flip = np.random.random_sample(
        (batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_height // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * channel))
    zeros = np.zeros((args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * channel))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_height // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * channel))
    return eta, real_input_flag


def train_wrapper(model):
    begin = 0
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        model.load(args.pretrained_model)
        begin = int(args.pretrained_model.split('-')[-1])
    else:
        print("No pre training weights found or training from scratch, iteration starts from 0")

    train_input_handle = datasets_factory.data_provider(configs=args,
                                                        data_train_path=args.data_train_path,
                                                        dataset=args.dataset,
                                                        data_test_path=args.data_val_path,
                                                        batch_size=args.batch_size,
                                                        is_training=True,
                                                        is_shuffle=True,
                                                        mode='train')
    
    val_input_handle = datasets_factory.data_provider(configs=args,
                                                      data_train_path=args.data_train_path,
                                                      dataset=args.dataset,
                                                      data_test_path=args.data_val_path,
                                                      batch_size=args.batch_size,
                                                      is_training=False,
                                                      is_shuffle=False,
                                                      mode='val')
    
    eta = args.sampling_start_value
    eta -= (begin * args.sampling_changing_rate)
    itr = begin

    # Initialization of Early Stop Mechanism===
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10

    for epoch in range(0, args.max_epoches):
        if itr > args.max_iterations:
            break
        for ims in train_input_handle:
            if itr > args.max_iterations:
                break
            batch_size = ims.shape[0]
            eta, real_input_flag = schedule_sampling(eta, itr, args.img_channel, batch_size)

            if itr % args.test_interval == 0 and itr > begin:
                print('Validate:')
                val_loss = trainer.test(model, val_input_handle, args, itr, is_val=True)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    model.save(itr)
                    print(f"==> Discover a better model! Saved (val_loss: {val_loss:.6f})")
                else:
                    patience_counter += 1
                    print(f"==> Verify that Loss has not decreased, patience value: {patience_counter}/{early_stop_patience}")
                    
                # 触发早停
                if patience_counter >= early_stop_patience:
                    print(f"\n Verification set loss has been continuous {early_stop_patience} times without a decrease, training ended")
                    return
                    
            trainer.train(model, ims, real_input_flag, args, itr)

            
            itr += 1


def test_wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(configs=args,
                                                       data_train_path=args.data_train_path,
                                                       dataset=args.dataset,
                                                       data_test_path=args.data_test_path,
                                                       batch_size=args.batch_size,
                                                       is_training=False,
                                                       is_shuffle=False,
                                                       mode='test')
    itr = 1
    for i in range(itr):
        trainer.test(model, test_input_handle, args, itr, is_val=False)


if __name__ == '__main__':
    print('Initializing models')
    if args.is_training == 'True':
        args.is_training = True
    else:
        args.is_training = False

    model = Model(args)

    if args.is_training:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.gen_frm_dir):
            os.makedirs(args.gen_frm_dir)
        train_wrapper(model)
    else:
        if not os.path.exists(args.gen_frm_dir):
            os.makedirs(args.gen_frm_dir)
        test_wrapper(model)