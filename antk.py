import torch
import math


class TNTK(torch.nn.Module):
    def __init__(self, num_layers, num_mlp_layers, scale, reg_lambda=1e-6):
        super(TNTK, self).__init__()
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.scale = scale
        assert (scale in ['uniform', 'degree'])
        self.reg_lambda = reg_lambda

        self.sigma_w2 = 2.
        self.sigma_b2 = 0.01

    def __adj(self, S, adj1, adj2):
        """
        go through one adj layer
        """
        tmp = adj1.mm(S)
        tmp = adj2.mm(tmp.transpose(0, 1)).transpose(0, 1)
        return tmp

    def __next_diag(self, S):
        """
        go through one normal layer, for diagonal element
        S: covariance of last layer
        """
        diag = torch.sqrt(S.diag())
        tmp = diag[:, None] * diag[None, :]
        S = S / tmp
        S = torch.clamp(S, -0.9999,
                        0.9999)  # smooth the value so the derivative will not lead into NAN: https://discuss.pytorch.org/t/nan-gradient-for-torch-cos-torch-acos/9617
        DS = (math.pi - torch.acos(S)) / math.pi
        S = (S * (math.pi - torch.acos(S)) + torch.sqrt(1 - torch.pow(S, 2))) / math.pi
        S = S * tmp

        return S, DS, diag

    def __next_diag_nngp(self, S):
        diag = torch.sqrt(S.diag())
        S = S * self.sigma_w2
        tmp = diag[:, None] * diag[None, :]
        S = S / tmp
        S = torch.clamp(S, -0.9999, 0.9999)
        theta = torch.acos(S)
        S = self.sigma_w2 * tmp * (torch.sin(theta) + (math.pi - theta) * torch.cos(theta)) / 2. / math.pi

        return S, diag

    def __next_nngp(self, S):
        # dense
        S = S * self.sigma_w2

        # relu
        S = torch.clamp(S, -0.9999, 0.9999)  # normalization
        theta = torch.acos(S)
        S = self.sigma_w2 * S * (torch.sin(theta) + (math.pi - theta) * torch.cos(theta)) / 2. / math.pi

        return S

    def __next(self, S):
        """
        go through one MLP layer
        """
        S = torch.clamp(S, -0.9999, 0.9999)
        DS = (math.pi - torch.acos(S)) / math.pi
        S = (S * (math.pi - torch.acos(S)) + torch.sqrt(1 - torch.pow(S, 2))) / math.pi
        return S, DS

    def diag(self, X, A):
        """
        compute the diagonal element of GNTK
        X: feature matrix
        A: adjacency matrix
        """

        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / torch.outer(A.sum(dim=1), A.sum(dim=0))

        diag_list = []
        diag_nngp_list = []
        nngp_xx_list = []
        sigma = torch.mm(X, X.T) + 0.0001
        sigma = scale_mat * self.__adj(sigma, A, A)
        ntk = torch.clone(sigma)
        nngp = torch.clone(sigma)

        for layer in range(1, self.num_layers):

            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma, diag = self.__next_diag(sigma)
                diag_list.append(diag)
                ntk = ntk * dot_sigma + sigma

                nngp, diag_nngp = self.__next_diag_nngp(nngp)
                diag_nngp_list.append(diag_nngp)

            if layer != self.num_layers - 1:
                nngp_xx_list.append(nngp)
                nngp_old = torch.clone(nngp)
                nngp = torch.matmul(torch.matmul(nngp, nngp), nngp.T)
                ntk = 2 * nngp + torch.matmul(torch.matmul(nngp_old, ntk * 2), nngp_old.T)

                sigma = self.__adj(sigma, A, A) * scale_mat
                ntk = self.__adj(ntk, A, A) * scale_mat
                nngp = self.__adj(nngp, A, A) * scale_mat

        return diag_list, diag_nngp_list, nngp_xx_list

    def forward(self, A_S, X_S, y_S, A_T, X_T, device):
        N, n = A_S.shape[0], A_S.shape[1]
        N_T, n_prime = A_T.shape[0], A_T.shape[1]

        assert A_S.shape == (N, n, n), "A_S shape wrong."
        assert A_T.shape == (N_T, n_prime, n_prime), "A_S shape wrong."

        A_S = 0.0001 * torch.eye(A_S.shape[1]).expand(A_S.shape[0], -1, -1).to(device) + A_S
        A_T = 0.0001 * torch.eye(A_T.shape[1]).expand(A_T.shape[0], -1, -1).to(device) + A_T

        diag_T_list = []
        diag_T_nngp = []
        nngp_xx_T_list = []
        for i in range(len(A_T)):
            diag, diag_nngp, nngp_xx = self.diag(X_T[i], A_T[i])
            diag = torch.cat(diag)
            diag_T_list.append(diag)
            diag_nngp = torch.cat(diag_nngp)
            diag_T_nngp.append(diag_nngp)
            nngp_xx = torch.cat(nngp_xx)
            nngp_xx_T_list.append(nngp_xx)
        diag_T = torch.cat(diag_T_list).view(N_T, self.num_mlp_layers * (self.num_layers - 1), -1)
        diag_T_nngp = torch.cat(diag_T_nngp).view(N_T, self.num_mlp_layers * (self.num_layers - 1), -1)
        nngp_xx_T = torch.cat(nngp_xx_T_list).view(N_T, self.num_layers - 2, n_prime, n_prime)
        assert diag_T.shape == (N_T, self.num_mlp_layers * (self.num_layers - 1), n_prime), "diag_T shape wrong."
        assert diag_T_nngp.shape == (
        N_T, self.num_mlp_layers * (self.num_layers - 1), n_prime), "diag_T_nngp shape wrong."
        assert nngp_xx_T.shape == (N_T, self.num_layers - 2, n_prime, n_prime), "nngp_xx_T shape wrong."

        diag_S_list = []
        diag_S_nngp = []
        nngp_xx_S_list = []
        for i in range(A_S.shape[0]):
            diag, diag_nngp, nngp_xx = self.diag(X_S[i], A_S[i])
            diag = torch.cat(diag)
            diag_S_list.append(diag)
            diag_nngp = torch.cat(diag_nngp)
            diag_S_nngp.append(diag_nngp)
            nngp_xx = torch.cat(nngp_xx)
            nngp_xx_S_list.append(nngp_xx)
        diag_S = torch.cat(diag_S_list).view(N, self.num_mlp_layers * (self.num_layers - 1), -1)
        diag_S_nngp = torch.cat(diag_S_nngp).view(N, self.num_mlp_layers * (self.num_layers - 1), -1)
        nngp_xx_S = torch.cat(nngp_xx_S_list).view(N, self.num_layers - 2, n, n)
        assert diag_S.shape == (N, self.num_mlp_layers * (self.num_layers - 1), n), "diag_S shape wrong."
        assert diag_S_nngp.shape == (N, self.num_mlp_layers * (self.num_layers - 1), n), "diag_S_nngp shape wrong."
        assert nngp_xx_S.shape == (N, self.num_layers - 2, n, n), "nngp_xx_S shape wrong."

        """
            Computing K_SS
        """
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / (torch.einsum('Na,Mb->NMab', A_S.sum(dim=2), A_S.sum(dim=1)))
            assert scale_mat.shape == (N, N, n, n), "scale_mat shape wrong."

        sigma = torch.einsum('Nab,Mbc->NMac', X_S, X_S.permute(0, 2, 1)) + 0.0001
        sigma = torch.einsum('Nab,NMbc,Mcd->NMad', A_S, sigma, A_S.permute(0, 2, 1)) * scale_mat
        assert sigma.shape == (N, N, n, n), "sigma shape wrong."

        ntk = torch.clone(sigma)
        nngp = torch.clone(sigma)

        cnt_mlp = 0
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                tmp = torch.einsum('Na,Mb->NMab', diag_S[:, cnt_mlp, :], diag_S[:, cnt_mlp, :]) + 0.000001
                assert tmp.shape == (N, N, n, n), "normalization matrix shape wrong."
                sigma = sigma / tmp
                sigma, dot_sigma = self.__next(sigma)
                sigma = sigma * tmp
                ntk = ntk * dot_sigma + sigma

                tmp_nngp = torch.einsum('Na,Mb->NMab', diag_S_nngp[:, cnt_mlp, :],
                                        diag_S_nngp[:, cnt_mlp, :]) + 0.000001
                nngp = nngp / tmp_nngp
                nngp = self.__next_nngp(nngp)
                assert tmp_nngp.shape == (N, N, n, n), "nngp normalization matrix shape wrong."
                cnt_mlp += 1

            if layer != self.num_layers - 1:
                nngp = torch.einsum('Nab,NMbc,Mcd->NMad',
                                    nngp_xx_S[:, layer - 1, :, :], nngp, nngp_xx_S[:, layer - 1, :, :].permute(0, 2, 1))
                ntk = 2 * nngp + torch.einsum('Nab,NMbc,Mcd->NMad',
                                              nngp_xx_S[:, layer - 1, :, :], ntk*2,
                                              nngp_xx_S[:, layer - 1, :, :].permute(0, 2, 1))

                ntk = torch.einsum('Nab,NMbc,Mcd->NMad', A_S, ntk, A_S.permute(0, 2, 1)) * scale_mat
                nngp = torch.einsum('Nab,NMbc,Mcd->NMad', A_S, nngp, A_S.permute(0, 2, 1)) * scale_mat
                sigma = torch.einsum('Nab,NMbc,Mcd->NMad', A_S, sigma, A_S.permute(0, 2, 1)) * scale_mat

        assert ntk.shape == (N, N, n, n), "ntk shape wrong."
        ntk = scale_mat ** (self.num_layers - 1) * ntk
        K_SS = ntk.mean(dim=(2, 3))  # (N, N)
        assert K_SS.shape == (N, N), "K_SS shape wrong."

        """
            Computing K_ST column by column
        """
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / (torch.einsum('Na,Mb->NMab', A_S.sum(dim=2), A_T.sum(dim=1)))
            assert scale_mat.shape == (N, N_T, n, n_prime), "K_ST scale_mat shape wrong."

        sigma = torch.einsum('Nab,Mbc->NMac', X_S, X_T.permute(0, 2, 1)) + 0.0001
        sigma = torch.einsum('Nab,NMbc,Mcd->NMad', A_S, sigma, A_T.permute(0, 2, 1)) * scale_mat
        assert sigma.shape == (N, N_T, n, n_prime), "K_ST sigma shape wrong."
        ntk = torch.clone(sigma)
        nngp = torch.clone(sigma)

        cnt_mlp = 0
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                tmp = torch.einsum('Na,Mb->NMab', diag_S[:, cnt_mlp, :], diag_T[:, cnt_mlp, :]) + 0.000001
                assert tmp.shape == (N, N_T, n, n_prime), "normalization matrix shape wrong."
                sigma = sigma / tmp
                sigma, dot_sigma = self.__next(sigma)
                sigma = sigma * tmp
                ntk = ntk * dot_sigma + sigma

                tmp_nngp = torch.einsum('Na,Mb->NMab', diag_S_nngp[:, cnt_mlp, :],
                                        diag_T_nngp[:, cnt_mlp, :]) + 0.000001
                nngp = nngp / tmp_nngp
                nngp = self.__next_nngp(nngp)
                assert tmp_nngp.shape == (N, N_T, n, n_prime), "nngp normalization matrix shape wrong."
                cnt_mlp += 1

            if layer != self.num_layers - 1:
                nngp = torch.einsum('Nab,NMbc,Mcd->NMad',
                                    nngp_xx_S[:, layer - 1, :, :], nngp, nngp_xx_T[:, layer - 1, :, :].permute(0, 2, 1))
                ntk = 2 * nngp + torch.einsum('Nab,NMbc,Mcd->NMad',
                                              nngp_xx_S[:, layer - 1, :, :], ntk * 2,
                                              nngp_xx_T[:, layer - 1, :, :].permute(0, 2, 1))

                ntk = torch.einsum('Nab,NMbc,Mcd->NMad', A_S, ntk, A_T.permute(0, 2, 1)) * scale_mat
                nngp = torch.einsum('Nab,NMbc,Mcd->NMad', A_S, nngp, A_T.permute(0, 2, 1)) * scale_mat
                sigma = torch.einsum('Nab,NMbc,Mcd->NMad', A_S, sigma, A_T.permute(0, 2, 1)) * scale_mat
        assert ntk.shape == (N, N_T, n, n_prime), "K_ST ntk shape wrong."
        ntk = scale_mat ** (self.num_layers - 1) * ntk
        K_ST = ntk.mean(dim=(2, 3))  # (N, N_T)
        assert K_ST.shape == (N, N_T), "K_ST shape wrong."

        """
            Prediction
        """
        KSS_reg = K_SS + self.reg_lambda * torch.trace(K_SS) / N * torch.eye(N).to(device)
        KSS_inverse_yS = torch.linalg.solve(KSS_reg, y_S)
        pred = K_ST.permute(1, 0).mm(KSS_inverse_yS)

        return pred, K_SS
