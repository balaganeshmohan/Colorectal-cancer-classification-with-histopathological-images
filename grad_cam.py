from fastai.vision import *
from fastai.callbacks import *
from fastai import *
import torchvision.models
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import torch.nn as nn


class GradCam():


    '''The below 3 functions implements GRAD-CAM visualization technique.
    Code gently borrowed from https://www.kaggle.com/qitvision/a-complete-ml-pipeline-fast-ai
    '''
    def prediction_overview(interp:ClassificationInterpretation,
                       classes=['MSIH', 'nonMSIH']):
        top_loss_val, top_loss_idx = interp.top_losses()
        fig, ax = plt.subplots(3,4, figsize=(14,14))
        fig.suptitle('Predicted / Actual / Loss / Probability',fontsize=20)
    
    #random samples
        for i in range(4):
            random_index = randint(0, len(top_loss_idx))
            idx = top_loss_idx[random_index]
            im, cl = interp.data.dl(DatasetType.Valid).dataset[idx]
            im = image2np(im.data)
            cl = int(cl)
            ax[0,i].imshow(im)
            ax[0,i].set_xticks([])
            ax[0,i].set_yticks([])
            ax[0,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.preds[idx][cl]:.2f}')
            ax[0,0].set_ylabel('Random samples', fontsize=16, rotation=0, labelpad=80)
        
    #incorrect samples
        for i in range(4):
            idx = top_loss_idx[i]
            im, cl = interp.data.dl(DatasetType.Valid).dataset[idx]
            cl = int(cl)
            im = image2np(im.data)
            ax[1,i].imshow(im)
        
            ax[1,i].set_xticks([])
            ax[1,i].set_yticks([])
            ax[1,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.preds[idx][cl]:.2f}')
        
            ax[1,0].set_ylabel('Most incorrect\nsamples', fontsize=16, rotation=0, labelpad=80)
        
    #correct samples
    
        for i in range(4):
            idx = top_loss_idx[len(top_loss_idx) - i - 1]
            im,cl = interp.data.dl(DatasetType.Valid).dataset[idx]
            cl = int(cl)
            im = image2np(im.data)
            ax[2,i].imshow(im)
            ax[2,i].set_xticks([])
            ax[2,i].set_yticks([])
            ax[2,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.preds[idx][cl]:.2f}')
        ax[2,0].set_ylabel('Most correct\nsamples', fontsize=16, rotation=0, labelpad=80)



        def hooked_backward(m, oneBatch, cat):
            # we hook into the convolutional part = m[0] of the model
            with hook_output(m[0]) as hook_a: 
                with hook_output(m[0], grad=True) as hook_g:
                preds = m(oneBatch)
                preds[0,int(cat)].backward()
            return hook_a,hook_g


    def getHeatmap(val_index):
        """Returns the validation set image and the activation map"""
        #procure the model
        m = learn.model.eval()
        tensorImg,cl = data_128.valid_ds[val_index]

        #getting batches
        oneBatch,_ = data_128.one_item(tensorImg)
        oneBatch_im = vision.Image(data_128.denorm(oneBatch)[0])

        # RGB to grayscale
        cvIm = cv2.cvtColor(image2np(oneBatch_im.data), cv2.COLOR_RGB2GRAY)


        hook_a,hook_g = hooked_backward(m, oneBatch, cl)

        acts = hook_a.stored[0].cpu()


        # Grad-CAM
        grad = hook_g.stored[0][0].cpu()
        grad_chan = grad.mean(1).mean(1)
        grad.shape,grad_chan.shape
        mult = (acts*grad_chan[...,None,None]).mean(0)
        return mult, cvIm





    def plot_heatmap_overview(interp:ClassificationInterpretation, classes=['MSIH','nonMSIH']):
        # top losses will return all validation losses and indexes sorted by the largest first
        tl_val,tl_idx = interp.top_losses()
        #classes = interp.data.classes
        fig, ax = plt.subplots(3,4, figsize=(16,12))
        fig.suptitle('Grad-CAM\nPredicted / Actual / Loss / Probability',fontsize=20)
        # Random
        for i in range(4):
            random_index = randint(0,len(tl_idx))
            idx = tl_idx[random_index]
            act, im = getHeatmap(idx)
            H,W = im.shape
            _,cl = interp.data.dl(DatasetType.Valid).dataset[idx]
            cl = int(cl)
            ax[0,i].imshow(im)
            ax[0,i].imshow(im, cmap=plt.cm.gray)
            ax[0,i].imshow(act, alpha=0.5, extent=(0,H,W,0),
                            interpolation='bilinear', cmap='inferno')
            ax[0,i].set_xticks([])
            ax[0,i].set_yticks([])
            ax[0,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.preds[idx][cl]:.2f}')
            ax[0,0].set_ylabel('Random samples', fontsize=16, rotation=0, labelpad=80)
            # Most incorrect or top losses
            for i in range(4):
                idx = tl_idx[i]
                act, im = getHeatmap(idx)
                H,W = im.shape
                _,cl = interp.data.dl(DatasetType.Valid).dataset[idx]
                cl = int(cl)
                ax[1,i].imshow(im)
                ax[1,i].imshow(im, cmap=plt.cm.gray)
                ax[1,i].imshow(act, alpha=0.5, extent=(0,H,W,0),
                                interpolation='bilinear', cmap='inferno')
                ax[1,i].set_xticks([])
                ax[1,i].set_yticks([])
                ax[1,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.preds[idx][cl]:.2f}')
                ax[1,0].set_ylabel('Most incorrect\nsamples', fontsize=16, rotation=0, labelpad=80)
            # Most correct or least losses
            for i in range(4):
                idx = tl_idx[len(tl_idx) - i - 1]
                act, im = getHeatmap(idx)
                H,W = im.shape
                _,cl = interp.data.dl(DatasetType.Valid).dataset[idx]
                cl = int(cl)
                ax[2,i].imshow(im)
                ax[2,i].imshow(im, cmap=plt.cm.gray)
                ax[2,i].imshow(act, alpha=0.5, extent=(0,H,W,0),
                                    interpolation='bilinear', cmap='inferno')
                ax[2,i].set_xticks([])
                ax[2,i].set_yticks([])
                ax[2,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.preds[idx][cl]:.2f}')
                ax[2,0].set_ylabel('Most correct\nsamples', fontsize=16, rotation=0, labelpad=80)


def main(interp):
    plot = GradCam()
    plot.plot_heatmap_overview(interp, ['MSIH','nonMSIH'])


if __name__ == '__main__':
    main(interp)