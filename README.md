# Code of AA-BiGAN
Main Dependencies
torch 1.1.0
torchvision 0.3.0
sklearn 0.20.3
numpy 1.19.5
matplotlib 3.0.3
cuda 10.1


How to run:

You can use following command:

python main.py --normal_digit 0 --n_epochs 1000 --batch_size 200 --auxiliary_digit 1 --latent_dim 128 --name cifar --gamma_p 0 --gamma_l 0.2 --k 1 --dataset CIFAR --dir /CIFAR0.2/summary//

to train an Anomaly-Aware BiGAN on CIFAR-10 dataset. The result will save as  ./dir/name.csv

You can also use the following command:

bash bash_cifar.sh
bash bash_fmnist.sh
bash bash_mnist.sh

to run the .sh example file.


option choices:
dataset =[CIFAR,F-MNIST,MNIST]
gamma_l = [0.01,0.05,0.1,0.2]
gamma_p= [0,0.01,0.05,0.1,0.2]
k = [0,1,2,3,5]
latent_dim = [128 (CIFAR), 100 (F-MNIST,MNIST)]	



'main_unsupervised.py'  is for the scenario Anoamly-Aware BiGAN without the leverage of collected anomalies (gamma_l=0 scenario). 
Use command:
python main_unsupervised.py --normal_digit 0 --n_epochs 200 --batch_size 200  --latent_dim 100 --name fmnist --gamma_p 0  --dataset F-MNIST 

to run.
