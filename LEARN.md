## Generative Adversarial Networks (GANs)
A GAN has 2 parts:
1. Generator: this learns to generate a reasonable data
2. Discriminator: this learns to distinguish the fake and real data

As generator generates a data discriminator distinguishes if it is fake or real. When discriminator detects a fake data, it punishes the generator through backward propogation. As training goes, generator learns how to generate realistic data and discriminator starts to fail detecting fake. 

For better understanding & studying GANs, there are many introductry material that I found extremely helpful. Here are few of them: 
1. As an introduction to topic and gain some knowledge, there is a great article that can be accessed from [here](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/).
2. A free Google course that is a great way to grasp the topic, see real world references, and implementing GAN for handwritten digits in **tenserflow** can be accessed from [here](https://developers.google.com/machine-learning/gan/programming-exercise).
3. For having a more detailed understanding and build various GANs, DeepLearning.AI Generative Adversarial Networks (GANs) Specialization can be accessed from [here](https://www.coursera.org/specializations/generative-adversarial-networks-gans).

## Usage
You can just run the train.py file without any changes needed. 
Using colab is good for faster training.
For experimenting & see the changes, some places that can be changed are commented.

## Train results

| Epoch-0    | Epoch-5   | 
|------------|-------------| 
| <img src="/results/epoch-0.png" width="250" alt="Epoch-0">     | <img src="/results/epoch-5.png" width="250">     | 

| Epoch-10    | Epoch-15   | 
|------------|-------------| 
| <img src="/results/epoch-10.png" width="250"> | <img src="/results/epoch-15.png" width="250"> |





