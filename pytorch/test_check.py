import torch
from Gan_Pytorch import GAN


def load():
        checkpoints = torch.load(checkpoint_path, map_location='cuda:0')
        # gen_model.load_state_dict(checkpoints['state_dict'])
        # dis_model.load_state_dict(checkpoints['state_dict1'])
        # gen_optim.load_state_dict(checkpoints['optimizer'])
        # dis_optim.load_state_dict(checkpoints['optimizer1'])
        best_fid = checkpoints['best_fid']
        start_epoch = checkpoints['epoch']

if __name__ == '__main__':
        # epochs = 10000
        # batch_size = 192
        # lr = 1e-5
        # input_dims = 100
        # input_shape = (28, 28, 1)
        # output_shape = 1
        # log = []
        # best_fid = torch.finfo(torch.float64).max
        # # 如果cuda可用，则使用cuda加速训练
        # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # # device = 'cpu'
        checkpoint_path = f'./check_point/curr_checkpoint_Inter.pth.tar'
        # gan_model = GAN(input_shape, input_dims, output_shape, device)
        load()