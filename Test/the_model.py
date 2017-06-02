import theano
from theano import tensor as T
import numpy as np
import sys
import os 
sys.path.insert(0, '/home/rgoel/drmm/theano_implementation/drmm')

from CRM_no_factor_latest import CRM
from DRM_no_factor_latest import DRM_model
from nn_functions_latest import SoftmaxNonlinearity, \
    HiddenLayerInSoftmax
from tensorflow.examples.tutorials.mnist import input_data

class ModelT(DRM_model):
    '''
    `batch_size`:   size of minibatches used in each iteration
    `Cin`: The number of channels of the input
    `W`: The width of the input
    `H`: The height of the input
    `seed`: random seed
    `param_dir`: dir that saved the trained DRMM. Used in transfer learning or just to continue the training that stopped
                 for some reason
    `reconst_weights`: used to weigh reconstruction cost at each layer in the DRMM
    `xavier_init`: {True, False} use Xavier initialization or not
    `grad_min, grad_max`: clip the gradients
    `init_params`: {True, False} if True, load parameters from a trained DRMM
    `noise_std`: standard deviation of the noise added to the model
    `noise_weights`: control how much noise we want to add to the model. We always noise_weight to 0
    `is_bn_BU`: {True, False} use batch normalization in the Bottom-Up or not
    `nonlin`: {relu, abs} the nonlinearity to use in the Bottom-Up
    `method`: {SGD, adam} training method
    `KL_coef`: weight of the KL divergence cost
    `sign_cost_weight`: weight of the sign cost
    `is_prun`: {True, False} do neuron pruning or not
    `is_prun_synap`: {True, False} do synap pruning or not
    `is_prun_songhan`: {True, False} do synap pruning using Song Han's weight thresholding method as in Deep Compression
    `is_finetune_after_prun`: {True, False} finetune the model after the pruning finishes or not
    `is_dn`: {True, False} apply divisive normalization (DivNorm) or not
    `update_mean_var_with_sup`: {True, False} if True, only use data_4D_lab to update the mean and var in BatchNorm
    `cost_mode`: {'sup', 'unsup', 'both'} use labeled data, unlabeled data, or both to compute the costs
    `is_sup`: {True, False} do supervised learning or not
    '''
    # TODO: factor out common init code from all models

    def reshape_data(self, data, to_shape = (1, 28, 28)):
        data_sh = data.shape
        return data.reshape((data_sh[0], to_shape[0], to_shape[1], to_shape[2]))

    def __init__(self, batch_size, Cin, W, H, seed, param_dir=[],
                 reconst_weights=0.0, xavier_init=False, grad_min=-np.inf, grad_max=np.inf,
                 init_params=False, noise_std=0.45, noise_weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 is_bn_BU=False, nonlin='relu',
                 method='SGD', KL_coef=0.0, sign_cost_weight=0.0,
                 is_prun=False, is_prun_synap=False, is_prun_songhan=False, is_finetune_after_prun=False,
                 param_dir_for_prun_mat=[], is_dn=False,
                 update_mean_var_with_sup=False,
                 cost_mode='unsup',
                 is_sup=True):

        print('SHAPENET_Conv_Large_9_Layers')

        self.num_class = 10 # number of object classes in the BlenderRender training set

        self.noise_std = noise_std
        self.noise_weights = noise_weights
        self.reconst_weights = reconst_weights
        self.sign_cost_weight = sign_cost_weight
        self.is_bn_BU = is_bn_BU
        self.batch_size = batch_size
        self.grad_min = grad_min
        self.grad_max = grad_max
        self.nonlin = nonlin
        self.KL_coef = KL_coef
        self.method = method
        self.is_prun = is_prun
        self.is_prun_synap = is_prun_synap
        self.is_prun_songhan = is_prun_songhan
        self.is_dn = is_dn
        self.update_mean_var_with_sup = update_mean_var_with_sup
        self.cost_mode = cost_mode
        self.is_sup = is_sup

        # decide if there is noise in layer i or not
        
        # Specify the dimension of input, filters, and output at each layer
        self.H1 = H  # 64
        self.W1 = W  # 64
        self.Cin1 = Cin  # 1

        self.h1 = 3
        self.w1 = 3
        self.K1 = 1
        self.M1 = 1

        self.H8 = 13  # 32
        self.W8 = 13  # 32
        self.Cin8 = 1  # 96

        self.h8 = 3
        self.w8 = 3
        self.K8 = 1
        self.M8 = 1

        self.H9 = 5  # 34
        self.W9 = 5  # 34
        self.Cin9 = 1  # 96

        self.h9 = 3
        self.w9 = 3
        self.K9 = 1
        self.M9 = 1


        # self.H8 = self.H7 - self.h7 + 1  # 6
        # self.W8 = self.W7 - self.w7 + 1  # 6
        # self.Cin8 = self.K7  # 192

        # self.h8 = 1
        # self.w8 = 1
        # self.K8 = 192
        # self.M8 = 1

        # self.H9 = self.H8 - self.h8 + 1  # 6
        # self.W9 = self.W8 - self.w8 + 1  # 6
        # self.Cin9 = self.K8  # 192

        # self.h9 = 1
        # self.w9 = 1
        # self.K9 = self.num_class
        # self.M9 = 1

        # self.H_Softmax = 1
        # self.W_Softmax = 1
        self.Cin_Softmax = 1 # self.num_class

        # self.h_Softmax = 1
        # self.w_Softmax = 1
        self.K_Softmax = 10
        # self.M_Softmax = 1

        self.seed = seed
        np.random.seed(self.seed)

        # placeholders for labeled data and unlabeled data
        self.x_lab = T.tensor4('x_lab')
        self.x_unl = T.tensor4('x_unl')

        # placeholders for the object labels
        self.y_lab = T.ivector('y_lab')

        # placeholders for other hyper-paramaters that changes during the training
        self.lr = 0.00001
        # self.prun_threshold = T.vector('prun_threshold', dtype=theano.config.floatX)
        # self.prun_weights = T.vector('prun_weights', dtype=theano.config.floatX)
        # self.is_train = T.iscalar('is_train')
        # self.momentum_bn = T.scalar('momentum_bn', dtype=theano.config.floatX)
        self.momentum_bn = 0.99
        # Building the forward model
        self.conv1 = CRM(
            data_4D_lab=self.x_lab,
            data_4D_unl=self.x_unl,
            data_4D_unl_clean=self.x_unl,
            # data_4D_unl=None,
            # data_4D_unl_clean=None,
            noise_weight=0.0,
            noise_std=0.01,
            is_train=1, momentum_bn=self.momentum_bn,
            K=self.K1, M=self.M1, W=self.W1, H=self.H1,
            w=self.w1, h=self.h1, Cin=self.Cin1, Ni=self.batch_size,
            
        )

        self.conv1._E_step_Bottom_Up()

        
        self.conv8 = CRM(data_4D_lab=self.conv1.output_lab,
                         data_4D_unl=self.conv1.output,
                         data_4D_unl_clean=self.conv1.output_clean,
                         noise_weight=0.0,
                         noise_std=0.01,
                         is_train=1, momentum_bn=self.momentum_bn,
                         K=self.K8, M=self.M8, W=self.W8, H=self.H8,
                         w=self.w8, h=self.h8, Cin=self.Cin8, Ni=self.batch_size,
                         )

        self.conv8._E_step_Bottom_Up()

        self.conv9 = CRM(data_4D_lab=self.conv8.output_lab,
                         data_4D_unl=self.conv8.output,
                         data_4D_unl_clean=self.conv8.output_clean,
                         noise_weight=0.0,
                         noise_std=0.01,
                         is_train=1, momentum_bn=self.momentum_bn,
                         K=self.K9, M=self.M9, W=self.W9, H=self.H9,
                         w=self.w9, h=self.h9, Cin=self.Cin9, Ni=self.batch_size,
                         )

        self.conv9._E_step_Bottom_Up()

        # regression layer for softmax
        self.RegressionInSoftmax = HiddenLayerInSoftmax(input_lab=self.conv9.output_lab.flatten(2),
                                                        input_unl=self.conv9.output.flatten(2),
                                                        input_clean=self.conv9.output_clean.flatten(2),
                                                        # input_unl=None,
                                                        # input_clean=None,
                                                        n_in=self.Cin_Softmax, n_out=self.K_Softmax,
                                                        W_init=None, b_init=None)

        softmax_input_lab = self.RegressionInSoftmax.output_lab
        softmax_input = self.RegressionInSoftmax.output
        softmax_input_clean = self.RegressionInSoftmax.output_clean

        # softmax nonlinearity for object recognition
        self.softmax_layer_nonlin = SoftmaxNonlinearity(input_lab=softmax_input_lab, 
                                                        input_unl=softmax_input,
                                                        input_clean=softmax_input_clean)
                                                        # input_unl=None,
                                                        # input_clean=None)

        # Collecting the layers
        self.layers = [self.conv9, self.conv8, self.conv1]
        self.layer_names = ['conv9', 'conv8', 'conv1']
        self.N_layer = len(self.layers)
        self.survive_thres = [0.0, 0.0, 0.0]

        # build Top-Down pass
        # self.Build_TopDown_End_to_End()

        # build the cost function for the model
        self.Build_Cost()

        # build update rules for the model
        self.Build_Update_Rule(method=self.method)

        # self.Collect_Monitored_Vars()

    def compute_one_hot(self, input):
        sampling_indx = T.argmax(input, axis=1)
        one_hot_vec = T.extra_ops.to_one_hot(sampling_indx, self.num_class)
        output = input - input * one_hot_vec
        return one_hot_vec, output
