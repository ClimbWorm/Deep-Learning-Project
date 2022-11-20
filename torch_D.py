import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pytorch_model_summary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# fake class label 0
# true class label 1

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        # input size: [batch_size, channel, height, width ] [64,3,32,32]
        self.name = "D"
        self.batch_size = 64
        self.conv = nn.Sequential(
            # normal [64,32,32] padding=1 to keep shape the same
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # down sampling [128,32,32]
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=17),
            # nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            #
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=17),#[256,32,32]
            # nn.LeakyReLU(negative_slope=0.2),
            # classifier
            nn.Flatten(),
            nn.Dropout(p=0.4)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=256 * 32 * 32, out_features=1),
            nn.Sigmoid()  # may be not proper to use sigmoid for binary classification problem?
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, 256 * 32 * 32)
        out = self.fc(out)
        return out


# no need of this by now
def train(net, tr_loader, te_loader, n_epochs=100, learning_rate=0.01):
    loss = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # tune the learning rate, optimize the result

    iter = 0
    for _ in tqdm(range(n_epochs)):
        for i, (images, labels) in enumerate(
                tr_loader):  # I wish the train_loader here be a combination of real and fake images
            # split the real and fake for the purpose of calculating accuracy separately
            images = images.to(device).float()
            labels = labels.to(device).float()
            # Load images
            # images = images.requires_grad_()

            # Forward pass to get output/logits
            y_pred = net(images)  # forward

            # Calculate Loss: softmax --> cross entropy loss
            l = loss(y_pred.squeeze(1), labels)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Getting gradients w.r.t. parameters
            l.backward()
            # Updating parameters
            optimizer.step()

            iter += 1
            # net.eval()
            if iter % 10 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for image, label in te_loader:
                    # Load images
                    image = image.requires_grad_()

                    # Forward pass only to get logits/output
                    outputs = net(image)

                    # Get predictions from the maximum value
                    # _, predicted = torch.max(outputs.data, 1) # this is for multi-class
                    predicted = outputs.squeeze(1) > 0.5

                    # Total number of labels
                    total += label.size(0)

                    # Total correct predictions
                    correct += (predicted == label).sum()

                accuracy = 100 * correct / total

                # Print Loss
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, l.item(), accuracy))

    return None


if __name__ == "__main__":
    d = D()
    print(summary(d, torch.zeros(64, 3, 32, 32)))
