#all needed imports
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
import matplotlib.pyplot as plt 

#import generator and discriminator and loss functions from model.py
#train augmentation
"""train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
"""
def visualize_images(tensor, i,no = 16, size=(1, 28, 28)):
    img = tensor.detach().cpu()
    grid = make_grid(img[:no], 4)
    plt.imshow(grid.permute(1,2,0).squeeze())
    plt.show()
    plt.savefig('image'+str(i)+'.jpg')
    files.download('image'+str(i)+'.jpg') 

#imports train set from Fashion-MNIST
train_set = datasets.FashionMNIST(
    root='./data',  
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomRotation((-20, +20)),
        transforms.ToTensor()
    ])
)

#load dataset into batches
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,
)
epochs = 20

disc = Discriminator()
disc.to('cuda')
gen = Generator()
gen.to('cuda')
disc_opt = discriminator_optimizer(disc.parameters())
gen_opt = generator_optimizer(gen.parameters())
disc.apply(weights_init)
gen.apply(weights_init)
for i in range(epochs):
  total_d_loss = 0.0
  total_g_loss = 0.0

  for real_img, _ in train_loader:
    real_img =  real_img.to('cuda')
    noise =  torch.randn(128, 64, device='cuda')
    disc_opt.zero_grad()
    fake_img =  gen(noise)
    #print(fake_img.shape)
    disc_pred = disc(fake_img)
    disc_fake_loss = discriminator_loss(disc_pred)

    disc_pred = disc(real_img)
    disc_real_loss = generator_loss(disc_pred)
    disc_loss = (disc_fake_loss + disc_real_loss) /2
    total_d_loss += disc_loss.item()
    disc_loss.backward()
    disc_opt.step()

    gen_opt.zero_grad()
    noise = torch.randn(128,64, device = 'cuda')
    fake_img =  gen(noise)
    disc_pred =  disc(fake_img)
    gen_loss =  generator_loss(disc_pred)

    total_g_loss += gen_loss.item()
    gen_loss.backward()
    gen_opt.step()
  avg_d_loss = total_d_loss / len(train_loader)
  avg_g_loss = total_g_loss / len(train_loader)



  print('Epoch: {}, average discr loss ={}, avg generator loss = {}'.format(i+1, avg_d_loss, avg_g_loss))
  visualize_images(fake_img, i)


   




