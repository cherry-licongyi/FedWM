# FedWM

Code repository for paper "FedWM: Data-Free Watermarking for Model Ownership Protection in Federated Learning"

## Abstract

The widespread adoption of federated learning has been driven by growing demands for privacy protection in model training. Federated learning enables multiple clients to collaboratively train a global model coordinated by a central server without sharing their raw data. However, when distributing the global model to clients, the central server faces significant security risks from malicious clients who may steal and misuse the model, thereby compromising its ownership. While existing watermarking techniques typically rely on main task data for ownership protection, their application in federated learning is limited since the server lacks access to this data, which remains with the clients. To address this challenge, we propose a novel data-free watermarking method. We utilize substitute data unrelated to the main task and improve efficiency by filtering out redundant samples. To optimize the watermarking process, we introduce a logits alignment-based optimization strategy that uses the substitute dataset with watermark triggers for effective embedding. Additionally, we propose a dynamic optimization algorithm to balance the trade-off between watermark embedding and main task. We comprehensively evaluate our approach across four datasets, four model architectures, and three mainstream deep learning tasks. Our experimental results demonstrate nearly perfect watermark performance while maintaining minimal impact on the main task. Notably, our watermarking method proves resistant to existing backdoor detection techniques, establishing its effectiveness, robustness and stealthiness.

## **Code**

The cammand to embed a watermark during FL training process.

```shell
# fed training should copy config to current dir first
cp ./configs_backup/mnist-config.txt ./config.txt
python3 -W ignore main_fed.py --dataset mnist --model mnist  --epochs 100 --gpu 0 --iid --num_users 5 --poison --frac 1

cp ./configs_backup/fashion-config.txt ./config.txt
python3 -W ignore main_fed.py --dataset fashion --model mnist  --epochs 50 --gpu 0 --iid --num_users 5 --poison --frac 1

cp ./configs_backup/vgg16-config.txt ./config.txt
python3 -W ignore main_fed.py --dataset cifar --model vgg16  --epochs 50 --gpu 0 --iid --num_users 5 --poison --frac 1

```
