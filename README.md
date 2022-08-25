# Basic GAN Project on Fashion-Mnist
This project contains basic implementation of a Generative Adversarial Network on pytorch. It is written to understand how to code GANs.

## Usage
You can just run the train.py file without any changes needed. 
For experimenting & see the changes, some places that can be changed are commented.


## Train results

| Epoch-0    | Epoch-5   | 
|------------|-------------| 
| <img src="/results/epoch-0.png" width="250" alt="Epoch-0">     | <img src="/results/epoch-5.png" width="250">     | 

| Epoch-10    | Epoch-15   | 
|------------|-------------| 
| <img src="/results/epoch-10.png" width="250"> | <img src="/results/epoch-15.png" width="250"> |



## GAN
A GAN has 2 parts:
1. Generator: this learns to generate a reasonable data
2. Discriminator: this learns to distinguish the fake and real data

As generator generates a data discriminator distinguishes if it is fake or real. When discriminator detects a fake data, it punishes the generator through backward propogation. As training goes, generator learns how to generate realistic data and discriminator starts to fail detecting fake. 

## Next Steps
-- TODO: add how to advance this project

