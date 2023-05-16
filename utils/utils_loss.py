import torch
import torch.nn.functional as F
import torch.nn as nn


class PaPiLoss(nn.Module):
    def __init__(self, predicted_score_cls, pseudo_label_weight = 0.99):
        super().__init__()
        self.predicted_score_cls1 = predicted_score_cls
        self.predicted_score_cls2 = predicted_score_cls

        self.init_predicted_score_cls = predicted_score_cls.detach()

        self.pseudo_label_weight = pseudo_label_weight
    

    def set_alpha(self, epoch, args):
        self.alpha = min((epoch / 10) * args.alpha_weight, args.alpha_weight)
    

    def set_pseudo_label_weight(self, epoch, args):
        start = args.pseudo_label_weight_range[0]
        end = args.pseudo_label_weight_range[1]
        self.pseudo_label_weight = 1. * epoch / args.epochs * (end - start) + start


    def update_weight_byclsout1(self, cls_predicted_score, batch_index, batch_partial_Y, args):
        with torch.no_grad():
            y_pred_raw_probas = torch.softmax(cls_predicted_score, dim = 1)
            
            revisedY_raw = batch_partial_Y.clone()
            revisedY_raw = revisedY_raw * y_pred_raw_probas
            revisedY_raw = revisedY_raw / revisedY_raw.sum(dim = 1).repeat(args.num_class, 1).transpose(0, 1)
            
            cls_pseudo_label = revisedY_raw.detach()
            
            self.predicted_score_cls1[batch_index, :] = self.pseudo_label_weight * self.predicted_score_cls1[batch_index, :] + \
                                                    (1 - self.pseudo_label_weight) * cls_pseudo_label
            

    def forward(self, cls_out1, cls_out2, logits_prot1, logits_prot2, logits_prot_1_mix, logits_prot_2_mix, idx_rp, Lambda, index, args, sim_criterion):
        y_pred_1_probas = torch.softmax(cls_out1, dim = 1)
        
        prot_pred_1_mix_probas_log = torch.log_softmax(torch.div(logits_prot_1_mix, args.tau_proto), dim = 1)
        prot_pred_2_mix_probas_log = torch.log_softmax(torch.div(logits_prot_2_mix, args.tau_proto), dim = 1)
        
        soft_positive_label_target1 = self.predicted_score_cls1[index, :].clone().detach()
        soft_positive_label_target1_rp = self.predicted_score_cls1[index[idx_rp], :].clone().detach()
        

        cls_loss_all_1 = soft_positive_label_target1 * torch.log(y_pred_1_probas)
        cls_loss_1 = - ((cls_loss_all_1).sum(dim=1)).mean()


        sim_loss_2_1 = Lambda * sim_criterion(prot_pred_1_mix_probas_log, soft_positive_label_target1) + \
                    (1 - Lambda) * sim_criterion(prot_pred_1_mix_probas_log, soft_positive_label_target1_rp)

        sim_loss_2_2 = Lambda * sim_criterion(prot_pred_2_mix_probas_log, soft_positive_label_target1) + \
                    (1 - Lambda) * sim_criterion(prot_pred_2_mix_probas_log, soft_positive_label_target1_rp)

        sim_loss_2 = sim_loss_2_1 + sim_loss_2_2
        

        return cls_loss_1, sim_loss_2, self.alpha

