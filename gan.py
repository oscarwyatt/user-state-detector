import time
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image


class Generator(nn.Module):
    """Image generator

    Takes a noise vector as input and syntheses a single channel image accordingly
    """

    def __init__(self, input_dims, output_dims):
        """Init function

        Declare the network structure as indicated in CW2 Guidance

        Arguments:
            input_dims {int} -- Dimension of input noise vector
            output_dims {int} -- Dimension of the output vector (flatten image)
        """
        super(Generator, self).__init__()
        # Original architecture
        # self.fc0 = nn.Sequential(nn.Linear(input_dims, 128), nn.LeakyReLU(0.2))
        # # output hidden layer
        # self.fc1 = nn.Sequential(nn.Linear(128, output_dims), nn.Tanh())
        # CW2 arch
        # # TODO: leaky relu rate?
        self.fc0 = nn.Sequential(nn.Linear(input_dims, 256), nn.LeakyReLU(0.2))
        self.fc1 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.2))
        self.fc2 = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(0.2))
        self.fc3 = nn.Sequential(nn.Linear(1024, output_dims), nn.Tanh())

    def forward(self, x):
        """Forward function

        Arguments:
            x {Tensor} -- a batch of noise vectors in shape (<batch_size>x<input_dims>)

        Returns:
            Tensor -- a batch of flatten image in shape (<batch_size
            To make it easier to understand, here is a small example:>x<output_dims>)
        """
        # Original architecture
        # x = self.fc0(x)
        # x = self.fc1(x)
        # CW2 arch
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class Discriminator(nn.Module):
    """Image discriminator

    Takes a image as input and predict if it is real from the dataset or fake synthesised by the generator
    """

    def __init__(self, input_dims, output_dims=1):
        """Init function

        Declare the discriminator network structure as indicated in CW2 Guidance

        Arguments:
            input_dims {int} -- Dimension of the flatten input images

        Keyword Arguments:
            output_dims {int} -- Predicted probability (default: {1})
        """
        super(Discriminator, self).__init__()
        self.fc0 = nn.Sequential(
            nn.Linear(input_dims, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, output_dims),
        )

    def forward(self, x):
        """Forward function

        Arguments:
            x {Tensor} -- a batch of 2D image in shape (<batch_size>xHxW)

        Returns:
            Tensor -- predicted probabilities (<batch_size>)
        """
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def show_result(G_net, z_, num_epoch, show=False, save=False, path='result.png'):
    """Result visualisation

    Show and save the generated figures in the grid fashion

    Arguments:
        G_net {[nn.Module]} -- The generator instant
        z_ {[Tensor]} -- Input noise vectors
        num_epoch {[int]} -- Indicate how many epoch has the generator been trained

    Keyword Arguments:
        show {bool} -- If to display the images (default: {False})
        save {bool} -- If to store the images (default: {False})
        path {str} -- path to store the images (default: {'result.png'})
    """
    images = G_net(z_).cpu()
    images = images.reshape(images.size(0), 1, 28, 28)
    grid = make_grid(images, nrow=5, padding=20)
    if save:
        save_image(grid, path)
    if show:
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    """Loss tracker

    Plot the losses of generator and discriminator independently to see the trend

    Arguments:
        hist {[dict]} -- Tracking variables

    Keyword Arguments:
        show {bool} -- If to display the figure (default: {False})
        save {bool} -- If to store the figure (default: {False})
        path {str} -- path to store the figure (default: {'Train_hist.png'})
    """
    x = range(len(hist['D_losses']))
    # y1 = hist['D_losses']
    # y2 = hist['G_losses']
    # plt.plot(x, y1, label='D_loss')
    # plt.plot(x, y2, label='G_loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend(loc=4)
    # plt.grid(True)
    # plt.tight_layout()
    # if save:
    #     plt.savefig(path)
    # if show:
    #     plt.show()
    # else:
    #     plt.close()




def create_noise(num, dim):
    return torch.clone(torch.tensor(word).repeat(num, 1))


if __name__ == '__main__':
    # initialise the device for training, if gpu is available, device = 'cuda', else: device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word = [-0.11272, -0.42158, -0.054724, 0.17511, 0.23005, 0.037133, -0.18699, 1.0386, -0.71984, -0.59138, 0.060872,
            -0.37143, -0.31244, 0.21062, -0.58843, 0.72982, 0.32796, 0.30826, 0.031759, 0.63549, 0.048248, -0.065918,
            -0.4342, -0.19576, 0.34424, 0.45779, 0.12751, 0.41101, 0.37435, 0.46835, 0.16429, 0.12245, 0.12399,
            -0.18338, -0.23387, -0.36701, -0.43599, -0.071388, -0.88454, 0.47744, -0.81029, -0.37892, 0.53191, 0.058453,
            -0.042986, -0.32189, -0.085888, 0.10207, 0.19045, -0.12399, -0.19046, -0.1968, -0.66974, 0.06395, -0.25769,
            0.43543, -0.17955, 0.12638, 0.33385, 0.03394, -0.46499, -0.25811, -0.048065, 0.1609, -0.21227, 0.39308,
            -0.14959, 0.13104, 0.016878, 0.36418, -0.15341, -0.004074, 0.05628, 0.33139, -0.70755, -0.098256, -0.084336,
            0.45607, -0.4864, -0.37335, -0.18315, 0.66806, 0.14881, 0.14449, -0.020697, 0.037833, -0.21829, 0.17548,
            0.09821, 0.095063, -0.067286, 0.12093, -0.10567, 0.12575, 0.25427, -0.27207, 0.1368, -0.39013, 0.27505,
            0.18207, 0.0052732, 0.062133, -0.10624, 0.26587, -0.46035, -0.054226, -0.14459, -0.16375, 0.29461, 0.093399,
            -0.13583, 0.17082, 0.40184, 0.35787, -0.25825, 1.0998, 0.14802, 0.078193, -0.14152, 0.52229, -0.17104,
            -0.31357, 0.23503, 0.26565, -0.23881, -0.22962, -0.48619, -0.14935, -0.10039, 0.13635, 0.38913, 0.16611,
            0.021053, -0.096143, 0.18026, 0.22481, 0.15288, -0.070379, -0.51827, 0.17356, -0.70186, 0.43782, -0.1923,
            0.17959, -0.072058, -0.6678, -0.10451, -0.09928, 0.1499, -0.36037, 0.22174, 0.35784, -0.14012, -0.14161,
            -0.62277, -0.169, -0.50654, 0.29271, 0.29122, -0.091463, 0.0068196, -0.36687, 0.50026, 0.18899, -0.10203,
            -0.25002, 0.026899, 0.24129, -0.13515, 0.096403, 0.021971, -0.17198, 1.1234, -0.35151, 0.40785, -0.019896,
            -0.52718, -0.32154, -0.13553, 0.60554, -0.67273, -0.055533, -0.36644, -0.18987, 0.41901, 0.10489, 0.37942,
            -0.28263, 0.33935, 0.028845, -0.084864, 0.04416, 0.082183, -0.058066, 0.42427, -0.84007, 0.11945, -0.65825,
            -0.028356, -0.11734, 0.2948, -0.20501, -0.016065, -0.31638, -0.46293, 0.80685, -0.28279, -0.036496, -0.1304,
            0.13609, 0.043952, 0.37603, 0.55332, -0.25841, 0.1065, -0.34781, 0.67129, 0.16868, 0.25809, 0.32737, 0.2755,
            -0.36059, -0.30383, 0.68676, -0.13022, 0.25934, 0.78226, 0.012263, 0.16156, -0.10268, -0.16986, 0.052082,
            0.40366, -0.083519, -0.13499, -0.014329, 0.22201, 0.24566, 0.037495, 0.45147, -0.29039, 0.15555, 0.63546,
            0.12303, 0.16822, -0.23156, 0.32978, -0.39274, 0.067953, -0.45587, -0.23946, -0.34797, -0.071038, -0.16343,
            -0.75228, 0.017495, 0.21517, 0.57923, 0.45702, -0.005037, -0.032399, -0.69121, 0.18122, -0.40416, -0.075326,
            0.36137, 0.26551, -0.16032, 0.2004, -0.2484, 0.15482, -0.14363, 0.44453, 0.31137, -0.14788, -0.71284,
            0.46672, 0.0055175, -0.1141, -0.31578, -0.15032, 0.36135, -0.50308, -0.12795, 0.31364, -0.19191, 0.21289,
            -0.10063, 0.1632, -0.037179, -0.033809, 0.31473, 0.022643, -0.4907, 0.90337, -0.20343, -0.29348, -0.44367,
            -0.20158, -0.80454, -0.12212, -0.034882, 0.13706, -0.18312, -0.093515, -0.030064, -0.003921, -0.048983,
            -8.8458, -0.40814, 0.17704, 0.20032, 0.34801, -0.53328, -0.52973, 0.36427, -0.19809, -0.45938, -0.27293,
            -0.29121, 0.081799, 0.49918, -0.045823, -0.29397, 0.2891, -0.46607, -0.47384, 0.25643, -0.02419, -0.19571,
            0.10508, -0.13225, -0.41918, 0.39785, 0.11239, -0.16499, -0.43147, -0.16567, -0.30886, -0.37188, 0.53323,
            0.54217, -0.56593, 0.17482, -0.37584, -0.12295, 0.52019, 0.70921, -0.069715, -0.24575, 0.60152, 0.53154,
            0.17633, -0.29031, 0.02466, -0.215, 0.086558, 0.4218, 0.17852, 0.054063, 0.70315, -0.54084, -0.056616,
            0.033268, 1.0884, 0.63027, 0.44974, -1.0002, 0.094973, -0.2305, 0.021384, 0.25524, -0.25808, -0.18028,
            -0.7329, -0.40259, -0.064848, 0.11093, -0.11681, -0.14281, -0.16668, -1.742, -0.76441, -0.14027, -0.18599,
            -0.18212, 0.21163, -0.060507, -0.0016709, -0.46904, 0.18654, 0.056574, -0.48903, -0.49443, -0.0048215,
            -0.48009, -0.40297, 0.5806, 0.24942, -0.37202, -0.12483, 0.17185, -0.32639, 0.038666, 0.51991, -0.58472,
            0.40975, 0.0060941, 0.53956, -0.03219, 0.21582, 0.44079, 0.05169, -0.26976, -0.42387, 0.4204, 0.22856,
            0.093787, 0.589, -0.28679, 0.3262, -0.46502, -0.031707, -0.084862, 0.73229, 0.22303, 0.27578, 0.1182,
            0.21433, 0.39588, 0.18802, -0.49105, -0.2109, 0.32014, -0.24623, -0.75696, -0.56058, 0.44992, -0.036036,
            -0.40698, -0.31531, -0.039987, -0.30113, 0.21576, -0.52212, -0.34464, -0.23047, 0.22124, 0.37953, 0.048534,
            0.51055, -0.08626, -0.34697, -0.24601, -0.041027, -0.26159, 0.10039, -0.065045, 0.023757, -0.5115, -0.15968,
            -0.074402, -0.28282, -0.82208, 0.13997, -0.28945, -0.51438, -0.58233, 0.1634, -0.15759, -0.37231, 0.22692,
            -0.05859, 0.21479, 0.29464, -0.28443, -0.1418, 0.10166, -0.15292, -0.33876, -0.13808, 0.5138, 0.091901,
            0.3039, 0.025259, 0.045899, -0.33531, 0.45009, -0.20979, 0.34302, -0.55333, -0.050188, 0.35394, -0.20881,
            -0.12465, -0.31422, -0.0055219, 0.42422, 0.18648, 0.28002, -0.022852, -0.40197, -0.23885, -0.10734, 0.43852,
            -0.24279, 0.033549, 0.49093, 0.088161, -0.11902, 0.11161, 0.095993, -0.72172, 0.30469, 0.044417, 0.32412,
            -0.37982, -0.23029, -0.34987, -0.42498, 0.02477, -0.55692, -0.5775, 0.1842, -0.23091, -0.033785, -0.51871,
            0.19248, -0.092583, -0.21766, 0.27517, 0.048763, -0.19717, -0.51125, -0.061541, -0.10402, -0.39787,
            -0.21125, 0.054275, -0.44816, 0.23492, 0.073312, -0.30561, 0.38017, -0.20985, -0.096843, 0.30311, -0.046657,
            -0.047185, -0.6098, -0.45456, -0.2763, -0.22813, 0.23857, -0.15739, 0.086956, 0.0052625, 0.19503, -0.020647,
            0.40797, 0.017377, -0.31735, 0.00073151, -0.3272, -0.10167, 0.11584, 0.058311, -0.14194, -0.33995, 0.4741,
            0.57677, 0.081534, -0.40992, -0.49437, -0.11042, 0.43175, 0.4147, 0.044082, -0.44849, -0.093484, 0.037944,
            -0.060233, 0.30573, -0.21256, -0.36759, -0.41138, 0.26653, -0.84644, -0.26427, 0.019039, 0.30831, -0.10121,
            -0.24576, 0.076075, 0.073263, 0.62095, -0.1192, 0.20358, -0.025522, 0.55796, 0.33156, -0.028891, 0.2348,
            0.50475, 0.27822, 0.098097, -0.061749, -0.52471, -0.716, -0.35413, 0.39377, 0.015386, -0.29142, 0.5291,
            -0.14333, 0.049358, -0.11029, -0.37662, 0.051148, 0.3139, 0.068158, 0.51599, 0.32595, 0.06247, -0.19295,
            -0.1962, 0.052164, 0.72487, -0.037806, 0.0094144, 0.56233, -0.073511, 0.13338, 0.60825, -0.4384, -0.070543,
            0.19253, 0.36454, 0.062674, -0.15863, -0.28499, 0.27923, 0.059363, 0.14903, -0.22921, 0.27017, 0.24537,
            0.16428, 0.36569, 0.25539, 0.3836, -0.2806, 0.12361, -0.011526, -0.19266, -0.5968, 0.43066, 0.13559,
            0.30028, -0.059617, -0.14276, -0.37636, 0.33238, 0.16023, -0.13176, 0.047244, 0.16542, 0.22063, 0.28546,
            -0.061646, -0.47287, -0.24634, 0.095971, 0.76168, -0.36434, -0.052604, -0.39802, -0.29125, 0.12927,
            -0.30375, 0.26685, -0.16527, -0.014572, 0.29109, -0.19375, -0.00082894, 0.096895, 0.31722, 0.025358,
            0.33655, -0.66408, -0.047226, 0.33285, -0.41841, 0.16408, -0.15067, -0.37905, -0.040689, -0.069304, 0.21773,
            0.044702, -0.08609, 0.072685, -0.09137, -0.38193, -0.46545, 0.34848, -0.37423, 0.68712, -0.51667, 0.16739,
            -0.18621, -0.45266, -0.24555, 0.02351, 0.22828, -0.1829, -0.38778, -0.074615, 0.50491, -0.15686, 0.21906,
            0.32318, -0.17289, -0.10035, -0.2243, 0.11674, 0.095808, -0.0466, -0.41, -0.61778, 0.3214, 0.47372,
            -0.87243, 0.18615, 0.58243, -0.26814, 0.017639, -0.46681, 0.25475, -0.12259, -0.57225, -0.20738, 0.087355,
            0.026069, -0.3117, -0.067945, -0.51853, 0.15353, 0.25949, 0.33959, -0.062935, -0.13637, 0.43082, 0.017123,
            -0.20531, -0.12607, -0.69892, 0.17694, -0.4299, 0.40172, 0.41446, 0.074672, 0.05857, -0.0085846, -0.20707,
            0.071902, 0.15358]

    # training parameters
    batch_size = 100
    learning_rate = 0.0002
    # Changing this to 0.01 results in much worse generator losses, ~12 after a few epochs
    # While the lower LR has a loss of ~2 after just a couple of epochs
    # The higher LR fails to converge
    # learning_rate = 0.01
    epochs = 100
    # 200 epochs
    # Generator loss: 2.40878222, Discriminator loss: 0.67962512
    # 100 epochs
    # Generator loss: 2.35982579, Discriminator loss: 0.70470404
    # Generator performance was roughly equal with 200 epochs, also with
    # the obvious drawback that it takes about twice as long to train.
    # This suggests that the model had converged by 100 epochs and further
    # training is unnecessary

    # parameters for Models
    # image_size = 28
    corpus_size = 50
    bert_vector_size = 768
    G_input_dim = bert_vector_size
    G_output_dim = corpus_size
    D_input_dim = bert_vector_size
    D_output_dim = bert_vector_size
    # declare the generator and discriminator networks
    G_net = Generator(G_input_dim, G_output_dim).to(device)
    D_net = Discriminator(D_input_dim, D_output_dim).to(device)

    # Binary Cross Entropy Loss function (original)
    # criterion = nn.BCELoss().to(device)
    criterion_d = nn.MSELoss().to(device)
    criterion_g = nn.MSELoss().to(device)
    # Initialise the Optimizer
    G_optimizer = torch.optim.Adam(G_net.parameters(), lr=learning_rate)
    D_optimizer = torch.optim.Adam(D_net.parameters(), lr=learning_rate)

    # tracking variables
    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    print("About to start training, learning parameters:")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    torch.autograd.set_detect_anomaly(True)
    start_time = time.time()
    # training loop
    Loss_G = []
    Loss_D = []
    page_content_embeddings = create_noise(batch_size, G_input_dim).to(device)
    for epoch in range(epochs):
        G_net.train()
        D_net.train()
        epoch_start_time = time.time()
        # predict best words to represent pages
        word_predictions = G_net(page_content_embeddings)
        num_indicies = 1
        word_predictions_top_k_indicies = torch.topk(word_predictions, num_indicies)
        # total fudge to get it working - should be BERT embeddings from words chosen by G
        word_embeddings_from_g = create_noise(batch_size, G_input_dim).to(device)
        predicted_content = D_net(word_embeddings_from_g.to(device)).to(device)
        # This should be the index of the word chosen
        # predicted_content = torch.tensor(0).to(device).repeat(batch_size, 1).to(device)
        loss_d = criterion_d(predicted_content, create_noise(batch_size, G_input_dim).to(device))
        D_optimizer.zero_grad()
        loss_d.backward()
        D_optimizer.step()
        word.reverse()
        predicted_content_index_reconstruction = torch.tensor(1).repeat(batch_size, 1).float().to(device)
        word.reverse()
        loss_g = criterion_g(word_predictions, predicted_content_index_reconstruction)
        G_optimizer.zero_grad()
        loss_g.backward()
        G_optimizer.step()

        ## store the loss of each iter
        Loss_D.append(loss_d.item())
        Loss_G.append(loss_g.item())
        epoch_loss_g = np.mean(Loss_G)  # mean generator loss for the epoch
        epoch_loss_d = np.mean(Loss_D)  # mean discriminator loss for the epoch
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        print("Epoch %d of %d with %.2f s" % (epoch + 1, epochs, per_epoch_ptime))
        print("Generator loss: %.8f, Discriminator loss: %.8f" % (epoch_loss_g, epoch_loss_d))
        # path = image_save_dir + '/MNIST_GAN_' + str(epoch + 1) + '.png'
        # show_result(G_net, create_noise(25, 100).to(device), (epoch + 1), save=True, path=path)
        # record the loss for every epoch
        train_hist['G_losses'].append(epoch_loss_g)
        train_hist['D_losses'].append(epoch_loss_d)
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)

        print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (
            np.mean(train_hist['per_epoch_ptimes']), epochs, total_ptime))
        print("Training finish!... save training results")