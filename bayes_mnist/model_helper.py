import numpy as np 
import tensorflow as tf

def print_weights(models, names):
    for i in range(len(models[0].layers)):
        for j, model in enumerate(models): 

            layer = model.layers[i]
            w = layer.get_weights()

            print("---" + str(names[j]) + " layer " + str(i) + " weights---")
            for x in w:
                print(x.shape)
                print(x)

def load_model(path): 
    return tf.keras.models.load_model(path)

def get_weights_shapes(model): 
    for i, l in enumerate(model.layers): 
        # l = model.layers[0]
            weights = l.get_weights()
            # W = np.array(weights[0])
            # b = weights[1]
            # print(W.shape)
            # print(b.shape)
            print("---layer " + str(i) + " weights---")
            for x in weights:
                print(x.shape)

"""
returns two lists: 
weights, valid 
"""
# def extract_weights(model):
    

def copy_weights(m1, m2):
    print_weights([m1, m2], ["NN", "BNN"])

    
    for i, l1 in enumerate(m1.layers): 
        l2 = m2.layers[i]
        w1 = l1.get_weights()
        w2 = l2.get_weights()
        
        if w1 and w2:
            w2[0] = w1[0]
            w2[2] = w1[1]
            # w2[2] = w1[1]

            l2.set_weights(w2)
            m2.layers[i] = l2
    print_weights([m1, m2], ["NN", "BNN"])
