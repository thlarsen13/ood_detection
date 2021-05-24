import numpy as np 
import tensorflow_probability as tfp
import tensorflow.compat.v2 as tf
from absl import flags
from absl import app

from lenet import MNISTSequence

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
"""
goal: test how well epistemic uncertainty can directly distinguish between in distribution an OOD datapoints 
"""

GRAPH = False 


def load_model(path):
    return tf.keras.models.load_model(path)

def generate_ood_sample(N): 
    return np.random.randint(-1.0, 1.0, size=(N, 28, 28, 1))

def run_ood_experiment(): 
    """
        1) get BNN model
        2) get MNIST test samples and OOD test samples. Aggregate into one test set with "labels"
        3) predict with BNN model and get uncertainties, use the ?? uncertainty as the "prediction"
            - how to compute the uncertainties from the BNN? 
                - Some papers suggest specific ways, but I don't see connection to the theory
            - try E, A, and E + A to see which gives the best. 
                According to literature it should be just E, to me it seems that A could be good since it depends on the actual sample. 
                How to reconcile proposed ways of computing E/A with the theoretical definitions? 

        4) generate AUROC curve based on these predictions
    """
    path = '/home/thlarsen/ood_detection/bayes_mnist/lenet_save/'
    model = load_model(path)

    train_set, heldout_set = tf.keras.datasets.mnist.load_data()
    train_seq = MNISTSequence(data=train_set)
    heldout_seq = MNISTSequence(data=heldout_set)

    N = 10
    T = 50

    ood_samples = generate_ood_sample(N)

    print("Predicting ood...")
    ood_probs = predict(ood_samples, model, T) #N x T x output_dim
    print("Done")
    print("Predicting id...")
    id_probs = predict(heldout_seq.images[:N], model, T) #N x T x output_dim
    model.evaluate(heldout_seq)
    print("Done")


    # print(y_hat[:3])
    ood_uncertainties = compute_epistemic(ood_probs)
    id_uncertainties = compute_epistemic(id_probs)

    ood_labels = np.ones(N)
    id_labels = np.zeros(N)

    # print(ood_uncertainties)
    # print(id_uncertainties)
    labels = np.concatenate((ood_labels, id_labels), axis=0) 
    preds = np.concatenate((ood_uncertainties, id_uncertainties), axis=0)

    print(preds.shape)
    print(labels.shape)
    auc = roc_auc_score(labels, preds)
    print("@@@AUROC = " + str(auc))

    if GRAPH: 

        lr_fpr, lr_tpr, _ = roc_curve(labels, preds)
        # plot the roc curve for the model
        # pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()

def predict(image, model, T):
    # predict stochastic dropout model T times
    p_hat = []
    for t in range(T):
        p_hat.append(model.predict(image)) 
    p_hat = np.array(p_hat).swapaxes(0, 1)
    return p_hat 

"""
average over T, p_hat is N x T x output_dim
maps T x output_dim -> single number representing uncertainty
computes 1/T sum_t=1...T (mu_t - mu)(mu_t - mu)^T
outputs N-entry uncertainty verctor
"""
def compute_epistemic(p_hat):
    N, T, O = p_hat.shape
    eps = []
    # return np.mean(p_hat**2, axis=1) - np.mean(p_hat, axis=1)**2
    for i in range(N): 
        #Given a single T x output_dim sample we take 
        p_bar = np.mean(p_hat[i], axis=0) #output_dim array 
        ep = np.sum((p_hat[i] - p_bar)**2) / T
        eps.append(ep) #Take the average

    return np.array(eps)

# def compute_aleatoric(p_har):
#   return np.mean(p_hat*(1-p_hat), axis=1)


def main(argv): 
    del argv

    tfd = tfp.distributions

    IMAGE_SHAPE = [28, 28, 1]
    NUM_TRAIN_EXAMPLES = 60000
    NUM_HELDOUT_EXAMPLES = 10000
    NUM_CLASSES = 10

    # flags.DEFINE_string(
    #     'model_dir',
    #     default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
    #                                              'bayesian_neural_network/'),
    #     help="Directory to put the model's fit.")
    
    # flags.DEFINE_integer('num_monte_carlo',
    #                                      default=50,
    #                                      help='Network draws to compute predictive probabilities.')

    # FLAGS = flags.FLAGS
    run_ood_experiment()

if __name__ == '__main__':
    app.run(main)


