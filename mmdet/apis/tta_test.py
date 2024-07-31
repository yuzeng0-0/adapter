import mmcv
import tempfile
from copy import deepcopy
import torch
from mmcv.image import tensor2imgs
import numpy as np




def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """
    
    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            dir = './.temp', suffix='.npy', delete=False).name
    print(temp_file_name)
    np.save(temp_file_name, array)
    return temp_file_name

def single_gpu_tent(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    origin_model = deepcopy(model)
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    seq_name = dataset.img_dir.split('/')[-1]
    param_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.requires_grad and ("norm" in name or "bn" in name):         # 只更新norm和bn的参数
                param_list.append(param)
                # print (name)
            else:
                param.requires_grad=False
    optimizer = torch.optim.Adam(param_list, lr=0.00006/8, betas=(0.9, 0.999))
    delta_sum = 0
    for i, data in enumerate(data_loader):

        gt = data['gt_semantic_seg'][0].to(data['img'][0]).to(torch.long)
        del data['gt_semantic_seg']

        with torch.no_grad():
            acc_origin = origin_model.forward(return_loss=True, img=data['img'][0], img_metas=data['img_metas'][0].data[0], gt_semantic_seg=gt)
            acc_adaption = model.forward(return_loss=True, img=data['img'][0], img_metas=data['img_metas'][0].data[0], gt_semantic_seg=gt)
            acc = round(acc_adaption['decode.acc_seg'].item(),2)
            acc_pre = round(acc_origin['decode.acc_seg'].item(),2)
            acc_delta = round(acc - acc_pre,2)
            delta_sum = round(delta_sum + acc_delta,2)
            print(seq_name,'delta_sum',delta_sum,'acc_delta:',acc_delta, 'acc_origin:',acc_pre,'loss_adaption:',acc)
        
            result,_ = model(return_loss=False, **data)

                
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)

        if isinstance(result, list):
            loss = model.forward(return_loss=True, img=data['img'][0], img_metas=data['img_metas'][0].data[0], gt_semantic_seg=torch.from_numpy(np.stack(result)).cuda().unsqueeze(1))
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            loss = model(return_loss=True, img=data['img'][0], img_metas=data['img_metas'][0].data[0], gt_semantic_seg=result)
            if efficient_test:
                result = np2tmp(result)
            results.append(result)


        torch.mean(loss["decode.loss_seg"]).backward()
        optimizer.step()
        optimizer.zero_grad()



        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results