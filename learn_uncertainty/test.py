import pickle


prefix = '/home/thlarsen/ood_detection/learn_uncertainty/'
dataset = 'cifar'
model_ids = ['BaselineTransfer']


with open(f'{prefix}{dataset}_eval_save/{model_ids}acc.pickle', 'rb') as handle:
    acc = pickle.load(handle)
with open(f'{prefix}{dataset}_eval_save/{model_ids}ece.pickle', 'rb') as handle:
    ece = pickle.load(handle)

print(acc)
print(ece)
