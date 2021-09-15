import time
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from toy_data import universal_credit, passport, tax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from random import random

class Generator(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(Generator, self).__init__()
        self.fc0 = nn.Sequential(nn.Linear(input_dims, 1536), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.fc1 = nn.Sequential(nn.Linear(1536, 3072), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.fc2 = nn.Sequential(nn.Linear(3072, 3072), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.fc3 = nn.Sequential(nn.Linear(3072, 3072), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.fc4 = nn.Sequential(nn.Linear(3072, 3072), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.fc5 = nn.Sequential(nn.Linear(3072, 1536), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.fc6 = nn.Sequential(nn.Linear(1536, output_dims), nn.Tanh())

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(Discriminator, self).__init__()
        self.fc0 = nn.Sequential(nn.Linear(input_dims, 1536), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.fc1 = nn.Sequential(nn.Linear(1536, 3072), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.fc2 = nn.Sequential(nn.Linear(3072, 3072), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.fc3 = nn.Sequential(nn.Linear(3072, 3072), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.fc4 = nn.Sequential(nn.Linear(3072, 1536),  nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.fc5 = nn.Sequential(nn.Linear(1536, output_dims))

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
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


def corpus_embeddings():
    return torch.row_stack((torch.tensor(tax(), requires_grad=True), torch.tensor(passport(), requires_grad=True))).to(device)

def create_noise(num, dim):

    tax_noise = torch.clone(torch.tensor(tax()).repeat(int(num / 2), 1))
    passport_noise = torch.clone(torch.tensor(passport()).repeat(int(num / 2), 1))
    concatenated = torch.cat((tax_noise, passport_noise))
    # Shuffle the concatenated examples
    return concatenated[torch.randperm(concatenated.size()[0])]

def generator_word_embeddings(indices):
    return corpus_embeddings()[indices]


def nearest_content(bert_reconstructions, batch_size):
    dist = torch.cdist(bert_reconstructions, corpus_embeddings())
    closest_indices = torch.topk(dist, 1, largest=False).indices[:, 0][:,0]
    return closest_indices
    embeddings = corpus_embeddings()
    reshaped_embeddings = torch.reshape(embeddings, (1, embeddings.shape[0], embeddings.shape[1]))
    resized_reshaped_embeddings = reshaped_embeddings.repeat(batch_size, 1, 1)
    content = resized_reshaped_embeddings[:, closest_indices, :][0, :, :]
    return content

def get_predictions(g_net, corpus_size, batch_size):
    word_predictions = g_net(page_content_embeddings)
    num_indicies = 1
    word_predictions_top_k_indicies = torch.topk(word_predictions, num_indicies)[1]
    return generator_word_embeddings(word_predictions_top_k_indicies), word_predictions_top_k_indicies
    # The below was an experiment - didn't seem to work but I'm keeping it around just in case
    # if random() > 0.5:
    #     word_predictions = g_net(page_content_embeddings)
    #     num_indicies = 1
    #     word_predictions_top_k_indicies = torch.topk(word_predictions, num_indicies)[1]
    #     return generator_word_embeddings(word_predictions_top_k_indicies)
    # else:
    #     randomly_chosen_indices = torch.randint(0, corpus_size, (batch_size,)).reshape(batch_size, 1)
    #     return generator_word_embeddings(randomly_chosen_indices)


if __name__ == '__main__':
    # initialise the device for training, if gpu is available, device = 'cuda', else: device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # training parameters
    batch_size = 500
    g_learning_rate = 0.01
    d_learning_rate = 0.001
    epochs = 1000

    # parameters for Models
    corpus_size = 2
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
    criterion_g = nn.MSELoss().to(device)#nn.MSELoss().to(device)
    # Initialise the Optimizer
    G_optimizer = torch.optim.Adam(G_net.parameters(), lr=g_learning_rate)
    D_optimizer = torch.optim.Adam(D_net.parameters(), lr=d_learning_rate)

    G_scheduler = ReduceLROnPlateau(G_optimizer, 'max', patience=5, factor=0.7)
    D_scheduler = ReduceLROnPlateau(D_optimizer, 'max', patience=5, factor=0.99)

    # tracking variables
    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    print("About to start training, learning parameters:")
    print(f"G Learning rate: {g_learning_rate}")
    print(f"D Learning rate: {d_learning_rate}")
    print(f"Batch size: {batch_size}")
    torch.autograd.set_detect_anomaly(True)
    start_time = time.time()
    # training loop
    Loss_G = []
    Loss_D = []
    for epoch in range(epochs):
        page_content_embeddings = create_noise(batch_size, G_input_dim).to(device)
        G_net.train()
        D_net.train()
        epoch_start_time = time.time()
        # predict best words to represent pages
        word_embeddings_from_g, predicted_word_indices = get_predictions(G_net, corpus_size, batch_size)
        for i in range(10):
            predicted_content = D_net(word_embeddings_from_g.to(device)).to(device)
            loss_d = criterion_d(predicted_content, page_content_embeddings)
            D_optimizer.zero_grad()
            loss_d.backward(retain_graph=True)
            D_optimizer.step()
        predicted_content_index_reconstruction = nearest_content(predicted_content, batch_size)
        print(predicted_content_index_reconstruction)
        print("s")
        print(predicted_word_indices[:, 0])
        results = predicted_content_index_reconstruction - predicted_word_indices[:, 0]
        loss_g = criterion_g(predicted_content_index_reconstruction, predicted_word_indices[:, 0])
        # loss_g = criterion_g(page_content_embeddings, predicted_content_index_reconstruction)
        G_optimizer.zero_grad()
        loss_g.backward()
        G_optimizer.step()

        G_scheduler.step(loss_g)
        D_scheduler.step(loss_d)

        print(f"G learning_rate: {G_scheduler.state_dict()['_last_lr']}")
        print(f"D learning_rate: {D_scheduler.state_dict()['_last_lr']}")

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
    print("should be 0")
    result = G_net(torch.tensor(tax()).to(device))
    print(torch.topk(result, 1)[1])
    print("probs")
    print(result)
    print()
    print()
    print("should be 1")
    result = G_net(torch.tensor(passport()).to(device))
    print(torch.topk(result, 1)[1])
    print("probs")
    print(result)