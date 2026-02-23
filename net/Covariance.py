import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import functools


class SpatialCovarianceBlock(nn.Module):
    def __init__(self):
        super(SpatialCovarianceBlock, self).__init__()

    def cal_covariance(self, input):
        CovaMatrix_list = []
        for i in range(len(input)):
            support_set_sam = input[i]
            B, C, h, w = support_set_sam.size()
            local_feature_list = []

            for local_feature in support_set_sam:
                local_feature_np = local_feature.detach().cpu().numpy()
                transposed_tensor = np.transpose(local_feature_np, (1, 2, 0))
                reshaped_tensor = np.reshape(transposed_tensor, (h * w, C))

                for line in reshaped_tensor:
                    local_feature_list.append(line)

            local_feature_np = np.array(local_feature_list)
            # mean = np.mean(local_feature_np, axis=0)
            # local_feature_list = [x - mean for x in local_feature_list]

            covariance_matrix = np.cov(local_feature_np, rowvar=False)
            covariance_matrix = torch.from_numpy(covariance_matrix)
            CovaMatrix_list.append(covariance_matrix)

        return CovaMatrix_list



    def mahalanobis_similarity(self, input, CovaMatrix_list):
        B, C, h, w = input.size()
        mahalanobis = []

        for i in range(B):
            query_sam = input[i]
            query_sam = query_sam.view(C, -1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm
            mea_sim = torch.zeros(1, len(CovaMatrix_list) * h * w).cuda()
            for j in range(len(CovaMatrix_list)):
                covariance_matrix = CovaMatrix_list[j].float().cuda()
                diff = query_sam - torch.mean(query_sam, dim=1, keepdim=True)
                temp_dis = torch.matmul(torch.matmul(diff.T, covariance_matrix), diff)
                mea_sim[0, j * h * w:(j + 1) * h * w] = temp_dis.diag()

            mahalanobis.append(mea_sim.view(1, -1))

        mahalanobis = torch.cat(mahalanobis, 0)

        return mahalanobis


    def forward(self, x1, x2):

        CovaMatrix_list = self.cal_covariance(x2)
        maha_sim = self.mahalanobis_similarity(x1, CovaMatrix_list)

        return maha_sim

class ChannelCovarianceBlock(nn.Module):
    def __init__(self):
        super(ChannelCovarianceBlock, self).__init__()

    def cal_covariance(self, input): #For Support set of [(n_class * n_way) x batch x C x h x w]
        CovaMatrix_list = []
        for i in range(len(input)):
            support_set_sam = input[i]
            B, C, h, w = support_set_sam.size()
            local_feature_list = []

            for local_feature in support_set_sam:
                local_feature_np = local_feature.detach().cpu().numpy()
                reshaped_tensor = np.reshape(local_feature_np, (C, h * w))

                for line in reshaped_tensor:
                    local_feature_list.append(line)

            local_feature_np = np.array(local_feature_list)
            # mean = np.mean(local_feature_np, axis=0)
            # local_feature_list = [x - mean for x in local_feature_list]

            covariance_matrix = np.cov(local_feature_np, rowvar=False)
            covariance_matrix = torch.from_numpy(covariance_matrix)
            CovaMatrix_list.append(covariance_matrix)

        return CovaMatrix_list



    def mahalanobis_similarity(self, input, CovaMatrix_list):
        B, C, h, w = input.size()
        mahalanobis = []

        for i in range(B):
            query_sam = input[i]
            query_sam = query_sam.view(C, -1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm
            mea_sim = torch.zeros(1, len(CovaMatrix_list) * C).cuda()
            for j in range(len(CovaMatrix_list)):
                covariance_matrix = CovaMatrix_list[j].float().cuda()
                diff = query_sam - torch.mean(query_sam, dim=1, keepdim=True)
                temp_dis = torch.matmul(torch.matmul(diff, covariance_matrix), diff.T)
                mea_sim[0, (j * C) : ((j + 1) * C)] = temp_dis.diag()

            mahalanobis.append(mea_sim.view(1, -1))

        mahalanobis = torch.cat(mahalanobis, 0)

        return mahalanobis


    def forward(self, x1, x2):

        CovaMatrix_list = self.cal_covariance(x2)
        maha_sim = self.mahalanobis_similarity(x1, CovaMatrix_list)

        return maha_sim

