import torch
import numpy as np
import torch.nn as nn
import my_ops
import os

is_cuda = torch.cuda.is_available()

#parameters
params = {'n_numbers': 40, #sample dimension
          'mu_lr': 0.1,
          'sigma_lr': 1e-6,
          'batch_size': 10,
          'prob_inc': 1.0,
          'samples_per_num_train': 5, #number of noise samples for each phi representation in training == replications of mu in training
          'n_units': 32, #dimension of the hidden layer
          'keep_prob': 1.,
          'eps': 1e-8,
          'min_eps':0.0001,
          'eps_0': 1.,
          'epsilon': 12,
          'min_range_test': 0, #minimum range of distribution of test set
          'max_range_test': 1, #maximum range of distribution of test set
          'train_noise_factor': True, #add noise to mu
          'test_set_size': 1,
          'noise_type': 'gumbel',
          'anneal_rate': 1e-5,
          'n_epochs': 2000, #notice early stopping condition
          }

def ensure_dir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def to_var(x):
    if is_cuda:
        x = x.cuda()
    return x

dir_path = os.path.dirname(os.path.realpath(__file__))
trained_models_path = os.path.join(dir_path, 'Model_pkls')
ensure_dir(trained_models_path)

reports_path = os.path.join(dir_path, 'Reports')
ensure_dir(reports_path)

"""Model class for encoder"""
class Encoding_Net(nn.Module):
    def __init__(self, latent_dim, output_dim, dropout_prob, samples_per_num):
        """
        in_flattened_vector: input flattened vector
        latent_dim: number of neurons in latent layer
        output_dim: dimension of log alpha square matrix
        """
        super(Encoding_Net, self).__init__()
        self.output_dim = output_dim
        self.samples_per_num = samples_per_num

        self.encoder = nn.Sequential(
            # net: output of the first neural network that connects numbers to a
            # 'latent' representation.
            # activation_fn: ReLU is default hence it is specified here
            # dropout p – probability of an element to be zeroed
            nn.Linear(1, latent_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            # now those latent representation are connected to rows of the matrix
            # log_alpha.
            nn.Linear(latent_dim, output_dim),
            nn.Dropout(p=dropout_prob))

    def forward(self, x):
        # each number is processed with the same network, so data is reshaped
        # so that numbers occupy the 'batch' position.
        phi_x = self.encoder(x.view(-1, 1))
        # reshape to cubic
        phi_x = phi_x.reshape(-1, self.output_dim, self.output_dim)
        return phi_x

class Sigma_Net(nn.Module) :
    def __init__(self, d):
        """
        in_flattened_vector: input flattened vector

        """
        super(Sigma_Net, self).__init__()
        self.d = d

        self.sigma_encoder = nn.Sequential(
            nn.Linear(self.d , 1),
            nn.Softplus()
            )

    def forward(self, x):
        # each number is processed with the same network, so data is reshaped
        # so that numbers occupy the 'batch' position.
        sigma_x = self.sigma_encoder(x)
        return sigma_x

class Direct_Permutation():
    def __init__(self,params):
        # Create the neural network
        self.dropout_prob = 1. - params['keep_prob']
        self.latent_dim = params['n_units']
        self.output_dim = params['n_numbers']
        self.eps = params['eps']
        self.n_epochs = params['n_epochs']
        self.n_numbers = params['n_numbers']
        self.epsilon = params['epsilon']
        self.batch_size = params['batch_size']
        self.annealing_rate = params['anneal_rate']
        self.eps_0, self.ANNEAL_RATE, self.min_eps = params['eps_0'], params['anneal_rate'], params['min_eps']
        self.samples_per_num_train = params['samples_per_num_train']
        self.train_noise_factor = params['train_noise_factor']
        self.noise_type = params['noise_type']

        self.perm_encoder = Encoding_Net(latent_dim=self.latent_dim, output_dim=self.output_dim, dropout_prob=self.dropout_prob, samples_per_num = self.samples_per_num_train)
        self.sigma_encoder = Sigma_Net(d = self.n_numbers)

        if torch.cuda.is_available():
            self.perm_encoder.cuda()
            self.sigma_encoder.cuda()

        self.optimizer_phi = torch.optim.Adam(self.perm_encoder.parameters(), lr=params['mu_lr'], eps=self.eps)
        self.optimizer_sigma = torch.optim.SGD(self.sigma_encoder.parameters(), lr=params['sigma_lr'])
        self.training_iter = 0

    # Training process
    def train_model(self, train_set):

        # tiled variables, to compare to many permutations
        train_ordered, train_random, train_perms = train_set[0], train_set[1], train_set[2]

        train_ordered_tiled = train_ordered.repeat(self.samples_per_num_train, 1)
        train_random_tiled = train_random.repeat(self.samples_per_num_train, 1)
        train_perms_tiled = train_perms.repeat(self.samples_per_num_train, 1)

        train_ordered_tiled, train_random_tiled = self.dataset_loader(train_ordered_tiled), self.dataset_loader(train_random_tiled)

        train_loss_history = []
        sigma_history = []
        epoch_history = []

        # Early stopping initialization
        epochs_no_improve = 0
        max_epochs_stop = 50
        best_loss = 99999

        for epoch in range(self.n_epochs):
            epoch_history.append(epoch)
            # Training phase of permutation encoder net
            self.perm_encoder.train()
            self.sigma_encoder.train()

            x_in = to_var(train_random).detach()
            train_ordered_tiled = to_var(train_ordered_tiled).detach()

            #obtain log alpha, log_alpha_w_noise
            log_alpha = self.perm_encoder(x_in)
            sigma = self.sigma_encoder(x_in)

            log_alpha_w_noise, noise_sigma = self.my_phi_and_sigma_gamma(log_alpha, self.samples_per_num_train, self.noise_type, self.train_noise_factor, sigma)

            # Solve a matching problem for a batch of matrices.
            hungarian_matching_index_solution_z_phi_gamma = my_ops.hungarian_matching(log_alpha_w_noise)

            hungarian_matching_permutation_matrix = my_ops.my_listperm2matperm(hungarian_matching_index_solution_z_phi_gamma)

            hungarian_matching_permutation = my_ops.my_invert_listperm(hungarian_matching_index_solution_z_phi_gamma)

            train_l2_loss, _, _, _, _ = self.build_l2s_loss(train_ordered_tiled, train_random_tiled, hungarian_matching_permutation, train_perms_tiled)
            encoder_gradient_direction_matrix = self.direction_encoder_gradient_calcuate(log_alpha_w_noise, hungarian_matching_permutation_matrix, train_perms, train_l2_loss, train_random, self.samples_per_num_train)

            #calculate loss to optimize phi encoder
            encoder_gradient_direction_matrix = (1. /1.) * encoder_gradient_direction_matrix
            encoder_loss = torch.sum(log_alpha_w_noise * to_var(encoder_gradient_direction_matrix))

            #calculate loss to optimize sigma encoder
            sigma_loss_calc = noise_sigma * to_var(encoder_gradient_direction_matrix)
            sigma_loss = torch.mean(sigma_loss_calc)

            #optimize on phi
            self.optimizer_phi.zero_grad()
            #optimize on sigma
            self.optimizer_sigma.zero_grad()

            encoder_loss.backward(retain_graph=True)
            sigma_loss.backward()

            self.optimizer_phi.step()
            self.optimizer_sigma.step()

            self.training_iter += 1
            if self.training_iter % 50 == 0:
                print("Epoch ", str(self.training_iter), 'Loss ', str("{:.7f}".format(torch.mean(train_l2_loss).item())))

            sigma_history.append(torch.mean(sigma).item())
            train_loss_history.append(torch.mean(train_l2_loss).item())

            # Update the progress bar.
            if torch.mean(train_l2_loss).item() < best_loss:
                best_loss = torch.mean(train_l2_loss).item()
                epochs_no_improve = 0
                torch.save(self.perm_encoder.state_dict(), os.path.join(trained_models_path,  str(self.n_numbers) + '_' + str(self.samples_per_num_train) + '_' + 'best_state_dict.pkl'))
            elif torch.mean(train_l2_loss).item() >= best_loss:
                epochs_no_improve += 1
                if epochs_no_improve >= max_epochs_stop:
                    torch.save(self.perm_encoder.state_dict(), os.path.join(trained_models_path, str(self.n_numbers) + '_' + str(self.samples_per_num_train) + '_' + 'best_state_dict.pkl'))
                    print("Early Stopping! Total epochs:",epoch, "training loss ", torch.mean(train_l2_loss).item())
                    print('Training completed')
                    return train_loss_history, epoch_history, sigma_history
        print('Training completed')
        return train_loss_history, epoch_history, sigma_history

    def direction_encoder_gradient_calcuate(self, log_alpha_w_noise, log_alpha_w_noise_permutation_matrix, train_perms, l2_diff, x_sampled, samples_per_num_train) :
        with torch.no_grad():
            log_alpha_w_noise_w_e_theta = log_alpha_w_noise.clone()
            reattempt = True

            batches = train_perms.size()[0]
            while reattempt:
                # associate the perturbation to its correlated position in log_alpha_w_noise according to the ground truth permutation
                for column_ind in range(self.n_numbers):
                    gt_selected_pos = train_perms.data[:, column_ind]

                    for b_index in range(batches):
                        for row_index in range(self.n_numbers):
                            if row_index == gt_selected_pos[b_index]:
                                for s in range(samples_per_num_train):
                                    # first position for batch
                                    # positive augmentation
                                    log_alpha_w_noise_w_e_theta[b_index + s*batches, row_index, column_ind] += self.epsilon * (x_sampled.data[b_index, column_ind].item() ** 2)
                            else:
                                for s in range(samples_per_num_train):
                                     # negative augmentation
                                    log_alpha_w_noise_w_e_theta[b_index + s*batches, row_index, column_ind] -= self.epsilon * (x_sampled.data[b_index, column_ind].item() ** 2)

                # Solve a matching problem for a batch of matrices.
                hungarian_matching_index_solution_with_epsilon_theta = my_ops.hungarian_matching(log_alpha_w_noise_w_e_theta)

                hungarian_matching_permutation_matrix_with_epsilon_theta = my_ops.my_listperm2matperm(hungarian_matching_index_solution_with_epsilon_theta)

                encoder_direction_matrix = (-1)*hungarian_matching_permutation_matrix_with_epsilon_theta + log_alpha_w_noise_permutation_matrix
                encoder_direction_matrix = encoder_direction_matrix.type(torch.float)

                n = log_alpha_w_noise.size()[1]
                batch_size = log_alpha_w_noise.size()[0]
                if torch.all(torch.eq(encoder_direction_matrix, torch.zeros([batch_size, n, n]))) and torch.sum(l2_diff) > 0.:
                    self.epsilon *= 1.1
                    print("*************************zero gradients loss positive")
                    print("*********increasing epsilon by 10%")
                else:
                    reattempt = False

            return encoder_direction_matrix

    def build_l2s_loss(self, ordered_tiled, random_tiled, hungarian_matching_permutation, perms):
        # The 3D output of permute_batch_split must be squeezed

        ordered_inf_learned_tiled = my_ops.my_permute_batch_split(random_tiled, hungarian_matching_permutation)
        ordered_inf_learned_tiled = to_var(ordered_inf_learned_tiled.view(-1, self.n_numbers))

        ordered_tiled = ordered_tiled.view(-1, self.n_numbers)

        l_diff = torch.abs(ordered_tiled - ordered_inf_learned_tiled)
        l2_diff = torch.mul(l_diff, l_diff)
        l2_diff = l2_diff.type(torch.float)

        diff_perms = torch.abs(hungarian_matching_permutation - perms)
        diff_perms = diff_perms.type(torch.float)

        prop_wrong = -torch.mean(torch.sign(-diff_perms))
        prop_any_wrong = -torch.mean(torch.sign(-torch.sum(diff_perms, dim=1)))
        kendall_tau = torch.mean(my_ops.my_kendall_tau(hungarian_matching_permutation, perms))

        return l2_diff, l_diff, prop_wrong, prop_any_wrong, kendall_tau

    def evaluate(self, eval_ordered, eval_random, eval_perms, samples_per_num):
        self.perm_encoder.eval()

        with torch.no_grad():
            if samples_per_num > 1:
                eval_ordered_tiled = eval_ordered.repeat(samples_per_num, 1)
                eval_random_tiled = eval_random.repeat(samples_per_num, 1)
                eval_perms_tiled = eval_perms.repeat(samples_per_num, 1)
            else:
                eval_ordered_tiled = eval_ordered
                eval_random_tiled = eval_random
                eval_perms_tiled = eval_perms

            eval_ordered_tiled, eval_random_tiled = self.dataset_loader(eval_ordered_tiled), self.dataset_loader(eval_random_tiled)

            x_in = eval_random

            if torch.cuda.is_available():
                x_in = x_in.cuda().detach()
                eval_ordered_tiled = eval_ordered_tiled.cuda().detach()
            else:
                x_in = x_in.detach()
                eval_ordered_tiled = eval_ordered_tiled.detach()

            # obtain log alpha, log_alpha_w_noise
            log_alpha  = self.perm_encoder(x_in)
            log_alpha_w_noise = self.my_phi_and_gamma(log_alpha, samples_per_num, self.noise_type, noise_factor = False)
            # Solve a matching problem for a batch of matrices.
            hungarian_matching_index_solution_z_phi_gamma = my_ops.hungarian_matching(log_alpha_w_noise)

            hungarian_matching_permutation = my_ops.my_invert_listperm(hungarian_matching_index_solution_z_phi_gamma)

            test_l2_loss, test_l1_loss, prop_wrong, prop_any_wrong, kendall_tau= self.build_l2s_loss(eval_ordered_tiled, eval_random_tiled, hungarian_matching_permutation, eval_perms_tiled)

            return test_l2_loss, test_l1_loss, prop_wrong, prop_any_wrong, kendall_tau


    def dataset_loader(self, data_set):
        data_set_tiled = data_set.view(-1, self.n_numbers, 1)

        return data_set_tiled

    def my_sample_logistic(self, shape):
        """Samples arbitrary-shaped standard gumbel variables.
        Args:
        shape: list of integers
        Returns:
        Sample from logistic distribution random variables
        """
        # Draw samples from a logistic distribution with specified parameters, loc (location or mean, also median), and scale (>0).
        samples = torch.from_numpy(np.random.logistic(loc=0.0, scale=1.0, size=shape))
        return samples.type(torch.float)

    def my_sample_gumbel(self, shape):
        """Samples arbitrary-shaped standard gumbel variables.
        Args:
        shape: list of integers
        eps: float, for numerical stability

        Returns:
        A sample of standard Gumbel random variables
        """
        # Sample from Gumbel with expectancy 0 and variance 3
        beta = np.sqrt(18./(np.square(np.pi)))

        mu = -beta*np.euler_gamma

        U = np.random.gumbel(loc=mu, scale=beta, size=shape)
        return torch.from_numpy(U).float()

    def my_phi_and_gamma(self, log_alpha, samples_per_num, noise_type, noise_factor):
        """
        Args:
        log_alpha: 2D tensor (a matrix of shape [N, N])
          or 3D tensor (a batch of matrices of shape = [batch_size, N, N])

        Returns:
        log_alpha_w_noise: a 4D tensor of [batch_size, n_samples, N, N] of
              noisy samples of log_alpha, If n_samples = 1 then the output is 3D.
        """
        n = log_alpha.size()[1]
        log_alpha = log_alpha.view(-1, n, n)
        batch_size = log_alpha.size()[0]

        if samples_per_num > 1:
            log_alpha_tiled = log_alpha.repeat(samples_per_num, 1 , 1)
        else:
            log_alpha_tiled = log_alpha
        if noise_factor == True:
            if noise_type == 'logistic':
                # samples from logistic distribution
                noise = to_var(self.my_sample_logistic([batch_size * samples_per_num, n, n]))
            elif noise_type == 'gumbel':
                noise = to_var(self.my_sample_gumbel([batch_size * samples_per_num, n, n]))

            log_alpha_w_noise = log_alpha_tiled + noise
        else:
            log_alpha_w_noise = log_alpha_tiled

        return log_alpha_w_noise

    def my_phi_and_sigma_gamma(self, log_alpha, samples_per_num, noise_type, noise_factor, sigma):
        """
        Args:
        log_alpha: 2D tensor (a matrix of shape [N, N])
          or 3D tensor (a batch of matrices of shape = [batch_size, N, N])

        Returns:
        log_alpha_w_noise: a 4D tensor of [batch_size, n_samples, N, N] of
              noisy samples of log_alpha, If n_samples = 1 then the output is 3D.
        """
        n = log_alpha.size()[1]
        log_alpha = log_alpha.view(-1, n, n)
        batch_size = log_alpha.size()[0]

        if samples_per_num > 1:
            log_alpha_tiled = log_alpha.repeat(samples_per_num, 1 , 1)
        else:
            log_alpha_tiled = log_alpha
        if noise_factor == True:
            if noise_type == 'logistic':
                noise = to_var(self.my_sample_logistic([batch_size * samples_per_num, n, n]))
            elif noise_type == 'gumbel':
                noise = to_var(self.my_sample_gumbel([batch_size * samples_per_num, n, n]))

            #rescale noise according to sigma
            noise_sigma_tiled = to_var(torch.zeros((batch_size * samples_per_num, n, n)))

            sigma_tiled = sigma.repeat(samples_per_num, 1)
            for bm in range(batch_size*samples_per_num):
                noise_sigma_tiled[bm] = sigma_tiled[bm] * noise[bm]

            log_alpha_w_noise = log_alpha_tiled + noise_sigma_tiled

        else:
            log_alpha_w_noise = log_alpha_tiled

        return log_alpha_w_noise, noise_sigma_tiled


def main():

    batch_size = params['batch_size']
    n_numbers = params['n_numbers']
    prob_inc = params['prob_inc']
    epsilon = params['epsilon']
    min_range_test = params['min_range_test']
    max_range_test = params['max_range_test']
    train_noise_factor = params['train_noise_factor']
    noise_type = params['noise_type']
    samples_per_num_train = params['samples_per_num_train']
    test_set_size = params['test_set_size']

    mu_lr = params['mu_lr']
    sigma_lr = params['sigma_lr']

    test_performance_file = os.path.join('Reports', 'Direct_bipartite_matching_signal-to-noise_log.txt')

    #create training set
    train_set = my_ops.my_sample_uniform_and_order(batch_size, n_numbers, prob_inc, 0 ,1)

    direct_permute = Direct_Permutation(params)
    # Train
    train_loss_history, epoch_history, sigma_history = direct_permute.train_model(train_set)

    trained_direct_permute = Direct_Permutation(params)
    trained_direct_permute.perm_encoder.load_state_dict(torch.load(os.path.join(trained_models_path, str(n_numbers) + '_' + str(samples_per_num_train) + '_' + 'best_state_dict.pkl')))

    test_iter = 0
    while test_iter < test_set_size:
        test_iter += 1
        # Create test set
        test_set = my_ops.my_sample_uniform_and_order(1, n_numbers, prob_inc, min_range_test, max_range_test)
        test_ordered, test_random, test_perms = test_set[0], test_set[1], test_set[2]
        # Evaluate on test set
        test_l2_loss, test_l1_loss, test_prop_wrong, test_prop_any_wrong, test_kendall_tau = trained_direct_permute.evaluate(test_ordered, test_random, test_perms, 1)
        #write test set performance to log
        with open(test_performance_file, 'a') as t_file:
            t_file.write(str(n_numbers) + '\t' + str(epsilon) + '\t' + str(mu_lr) + '\t' + str(sigma_lr) + '\t' + str(batch_size) + '\t' +
                         str(samples_per_num_train) + '\t' +
                         str(train_noise_factor) + '\t' + str(noise_type) + '\t' +
                         str("{0:.5f}".format(sigma_history[-1])) + '\t' +
                         str(test_prop_wrong.item()) + '\t' +
                         str(test_prop_any_wrong.item()) + '\t' + str(test_kendall_tau.item()) + '\n')

        to_print = True
        if to_print:
            print('TestSet Measures')
            print("test_l1_loss", torch.mean(test_l1_loss).item())
            print("test_l2_loss", torch.mean(test_l2_loss).item())
            print("test_prop_wrong", test_prop_wrong)
            print("test_prop_any_wrong", test_prop_any_wrong)
            print("test_kendall_tau", test_kendall_tau, '\n')

if __name__ == "__main__":
    main()
