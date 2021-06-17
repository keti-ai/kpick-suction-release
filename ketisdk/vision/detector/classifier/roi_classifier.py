from .classifier_base import BasCifarClassfier
import torch
import numpy as np


class RoiCifarClassfier(BasCifarClassfier):
    def predict_rois_m(self,im_tensors, boxes, inds):
        box_tensors = torch.from_numpy(boxes)
        ind_tensors = torch.from_numpy(inds)
        if self.use_cuda:
            im_tensors = [el.cuda() for el in im_tensors]
            ind_tensors, box_tensors = ind_tensors.cuda(), box_tensors.cuda()
        # im_tensors = im_tensors.view(1, *im_tensors.size())
        out =self.model((im_tensors, box_tensors, ind_tensors))

        # # visualize model
        # if not hasattr(self, 'do_torchviz'):
        #     from torchviz import make_dot
        #     make_dot(out, params=dict(list(self.model.named_parameters()))).render("rnn_torchviz", format="png")
        #     print('model saved...')
        #     self.do_torchviz = True
        return out.cpu().detach().numpy()


    def predict_tensor_rois(self, im_tensors, boxes, inds=None):
        if inds is None: inds = np.zeros((len(boxes),1), 'uint8')
        num_box = len(boxes)
        if num_box < self.args.net.test_batch: return self.predict_rois_m(im_tensors=im_tensors, boxes=boxes,
                                                                       inds=inds)

        # num_batch = int(np.ceil(num_box / self.args.test_batch))
        probs = []
        split_range = list(range(0, num_box,self.args.net.test_batch)) + [num_box,]
        for j, end in enumerate(split_range[1:]):
            start = 0 if j==0 else split_range[j]
            probs.append(self.predict_rois_m(im_tensors=im_tensors,
                                             boxes=boxes[start:end, :],
                                             inds=inds[start:end, :]))
        return np.vstack(probs)















