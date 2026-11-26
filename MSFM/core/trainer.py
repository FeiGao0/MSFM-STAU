import os
import os.path
import numpy as np
from core.utils import preprocess
import torch
import codecs

def train(model, ims, real_input_flag, configs, itr):
    _, loss_l1, loss_l2 = model.train(ims, real_input_flag, itr)
    if itr % configs.display_interval == 0:
        print('itr: ' + str(itr),
              'training L1 loss: ' + str(loss_l1), 'training L2 loss: ' + str(loss_l2))

        loss_file = os.path.join(configs.save_dir, 'train_loss.txt')
        with open(loss_file, 'a') as f:
            f.write(f"{itr}, {loss_l1}, {loss_l2}\n")

def test(model, test_input_handle, configs, itr, is_val=False):

    if is_val:
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for data in test_input_handle:
                batch_size = data.shape[0]
                real_input_flag = np.zeros(
                    (batch_size,
                     configs.total_length - configs.input_length - 1,
                     configs.img_height // configs.patch_size,
                     configs.img_width // configs.patch_size,
                     configs.patch_size ** 2 * configs.img_channel))
                

                img_gen = model.test(data, real_input_flag) # 形状: (B, 23, C, H, W)
                

                target = data[:, 1:].detach().cpu().numpy() 

                loss = np.mean(np.abs(target - img_gen))
                total_loss += loss * batch_size
                count += batch_size
                
        avg_loss = total_loss / count
        print(f'--> Validation L1 Loss: {avg_loss:.6f}')
        

        model.network.train()
        return avg_loss

    # ==== test mode ====
    print('test...')
    res_path = os.path.join(configs.gen_frm_dir, str(itr))

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    f = codecs.open(os.path.join(res_path, 'performance.txt'), 'w+')
    f.truncate()

    avg_mse = 0
    avg_mae = 0
    batch_id = 0
    img_mse, img_mae = [], []
    mse_list, mae_list = [], []

    all_predictions = []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        img_mae.append(0)
        mse_list.append(0)
        mae_list.append(0)

    for data in test_input_handle:
        print(batch_id)

        batch_size = data.shape[0]
        real_input_flag = np.zeros(
            (batch_size,
             configs.total_length - configs.input_length - 1,
             configs.img_height // configs.patch_size,
             configs.img_width // configs.patch_size,
             configs.patch_size ** 2 * configs.img_channel))
        output_length = configs.total_length - configs.input_length
        img_gen = model.test(data, real_input_flag)

        # ：(B, T, C, H, W) -> (B, T, H, W, C)
        img_gen = img_gen.transpose(0, 1, 3, 4, 2)
        test_ims = data.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)

        output_length = min(output_length, configs.total_length - 1)
        test_ims = preprocess.reshape_patch_back(test_ims, configs.patch_size)
        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        img_out = img_gen[:, -output_length:, :]

        clipped_img_out = np.clip(img_out, 0.0, 1.0)
        all_predictions.append(clipped_img_out)

        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :]
            gx = clipped_img_out[:, i, :]

            mse = np.square(x - gx).sum() / batch_size
            mae = np.abs(x - gx).sum() / batch_size

            img_mse[i] += mse
            img_mae[i] += mae
            mse_list[i] = mse
            mae_list[i] = mae

            avg_mse += mse
            avg_mae += mae

        f.writelines(str(batch_id) + ',' + str(mse_list) + ',' + str(mae_list) + '\n')
        batch_id = batch_id + 1

    f.close()


    if len(all_predictions) > 0:
        final_preds = np.concatenate(all_predictions, axis=0)
        final_preds = final_preds.transpose(0, 1, 4, 2, 3) # (N, T, H, W, C) -> (N, T, C, H, W)

        half_len = final_preds.shape[0] // 2
        pred_2020 = final_preds[:half_len]
        pred_2024 = final_preds[half_len:]

        np_save_dir = os.path.join(configs.gen_frm_dir, 'npy_output')
        if not os.path.exists(np_save_dir):
            os.makedirs(np_save_dir)

        np.save(os.path.join(np_save_dir, 'pred_2020_all.npy'), pred_2020)
        np.save(os.path.join(np_save_dir, 'pred_2024_all.npy'), pred_2024)
        print(f"\n=== The predicted data has been saved ===")
        print(f"2020: {os.path.join(np_save_dir, 'pred_2020.npy')}, shape: {pred_2020.shape}")
        print(f"2024: {os.path.join(np_save_dir, 'pred_2024.npy')}, shape: {pred_2024.shape}")

    with codecs.open(os.path.join(res_path, 'data.txt'), 'w+') as data_write:
        data_write.truncate()

        avg_mse = avg_mse / (batch_id * output_length)
        print('mse per frame: ' + str(avg_mse))
        for i in range(configs.total_length - configs.input_length):
            print(img_mse[i] / batch_id)
            img_mse[i] = img_mse[i] / batch_id
        data_write.writelines(str(avg_mse) + '\n')
        data_write.writelines(str(img_mse) + '\n')

        avg_mae = avg_mae / (batch_id * output_length)
        print('mae per frame: ' + str(avg_mae))
        for i in range(configs.total_length - configs.input_length):
            print(img_mae[i] / batch_id)
            img_mae[i] = img_mae[i] / batch_id
        data_write.writelines(str(avg_mae) + '\n')
        data_write.writelines(str(img_mae) + '\n')