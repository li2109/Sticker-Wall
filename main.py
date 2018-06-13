import argparse
from keras.datasets import cifar10
import numpy as np
from keras.utils import np_utils, generic_utils
from keras.models import Model
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import model_from_json
import keras.backend as K
from keras.optimizers import Optimizer
from keras.utils.generic_utils import CustomObjectScope
def update_parameters_cdsgd(agentmodels,agent):
    global parameters
    parameters=globals()["parameters"]
    for nb in range(agent):
        parameters[nb]=agentmodels[nb]._collected_trainable_weights

def model_compilers_cdsgd(model,agent,optimizer,pi,opt):
    global parameters
    parameters= [0 for nb in range(agent)]
    optparam= [0 for nb in range(agent)]
    for nb in range(agent):
        optparam[nb]={"agent": agent,
                "pi":K.variable(value=pi),
                "agent_id":nb
                }
        opt[nb] = CDSGD(lr=1E-2, decay=0, momentum=0.0, nesterov=False, optparam=optparam[nb])
    # Compile model
    for nb in range(agent):
        model[nb].compile(optimizer=opt[nb], loss="categorical_crossentropy", metrics=["accuracy"])
        parameters[nb]=model[nb]._collected_trainable_weights
    return model

class CDSGD(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """
    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, optparam=[],**kwargs):
        super(CDSGD, self).__init__(**kwargs)
        self.iterations = K.variable(0., name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.momentum = K.variable(momentum, name='momentum')
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.agent=optparam['agent']
        self.pi=optparam['pi']
        self.agent_id=optparam['agent_id']

    def get_updates(self, params, constraints, loss):
        pi=self.pi
        agent=self.agent
        agent_id=self.agent_id
        parameters=globals()["parameters"]
        info_shapes = [K.get_variable_shape(p) for p in params]
        parameter_copy = [0 for nb in range(agent)]
        for nb in range(agent):
            parameter_copy[nb]=parameters[nb]
            if (nb==agent_id):
                parameter_copy[nb]=params


        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / K.sqrt(1. + self.decay * self.iterations))
            self.updates .append(K.update_add(self.iterations, 1))

        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        dist_accumulator = [K.zeros(shape) for shape in shapes]


        self.weights = [self.iterations] + moments
        #for p, g, m, d, p_agents in zip(params, grads, moments, dist_accumulator, *parameter_copy):
        for i in range(len(params)):
            p=params[i]
            g=grads[i]
            m=moments[i]
            d=dist_accumulator[i]
            # Momentum term
            v = self.momentum * m - lr * g  # velocity
            #v = - lr * g # no momentum
            self.updates.append(K.update(m, v))
            if self.nesterov:
                for nb in range(agent):
                    d+=pi[nb][agent_id]*parameter_copy[nb][i]
                new_p = d + self.momentum * v - lr * g
            else:
                # This is for Debug only
                # if count>5:
                #     raise ValueError('parameters: ' + str(p1) + str(p2) + str(p3) + str(p4) + str(p5)  )

                for nb in range(agent):
                    d+=pi[nb][agent_id]*parameter_copy[nb][i]
                    #raise ValueError('pi:' + str(K.eval(pi)))
                new_p = d + v
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(CDSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def standard_conv_block(x, nb_filter, subsample=(1,1), pooling=False, bn=False, dropout_rate=None):
    x = Conv2D(nb_filter, (3, 3), padding="same")(x)
    if bn:
        x = BatchNormalization(mode=2, axis=1)(x)
    x = Activation("relu")(x)
    if pooling:
        x = MaxPooling2D()(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

def slice_dataset(data,total,segment):
    seg_length = segment
    return [data[x:x+seg_length] for x in range(0,len(data),seg_length)]

def train(batch, epoch, agent):
    print("->Training Begins!")
    print("->Loading cifar10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print("->Converting labels to one-hot encoding")
    img_dim = x_train.shape[1:]
    nb_classes = len(np.unique(y_train))
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print("->Distributing data to each agent...")
    datasize = x_train.shape[0]
    print("-->Total datasize: "+str(datasize))
    agent_dataset_size = int(datasize/agent)
    print("-->Each agent's datasize: "+str(agent_dataset_size))
    sliced_x_train = slice_dataset(x_train,datasize,agent_dataset_size)
    sliced_y_train = slice_dataset(y_train,datasize,agent_dataset_size)
    print("->Data distribution done!")

    print("->constructing the edge matrix pi")
    pi=np.ones((agent,agent))
    degree=agent
    degreeval=1.0/agent
    pi*=degreeval
    print(pi)

    print("->CNN model initializing...")
    x_input = Input(shape=img_dim, name="input")
    x = standard_conv_block(x_input, 32)
    x = standard_conv_block(x, 32, pooling=True, dropout_rate=0.25)
    x = standard_conv_block(x, 64)
    x = standard_conv_block(x, 64, pooling=True, dropout_rate=0.25)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(nb_classes, activation="softmax")(x)

    model = Model(inputs=[x_input], outputs=[x])
    model.name = 'CNN'
    model.summary()

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    
    print("->Serialize weights to HDF5...")
    model.save_weights("model0.h5")
    del model
    
    agentmodels= [0 for nb in range(agent)]
    for nb in range(agent):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        agentmodels[nb] = model_from_json(loaded_model_json)
        # load weights into new model
        agentmodels[nb].load_weights("model0.h5")
    print("->Loading weights to each agent done!")
    print("->Generating and compiling CDSGD optimizer...")
    print("->Accuracy measurement initializing...")
    opt= [0 for nb in range(agent)]
    agentmodels=model_compilers_cdsgd(agentmodels,agent,"CDSGD",pi,opt)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    agent_training_loss_history=[[] for nb in range(agent)]
    agent_validation_loss_history=[[] for nb in range(agent)]
    agent_training_acc_history=[[] for nb in range(agent)]
    agent_validation_acc_history=[[] for nb in range(agent)]

    training_loss=np.zeros(agent)
    training_acc=np.zeros(agent)
    validation_loss=np.zeros(agent)
    validation_acc=np.zeros(agent)

    for e in range(epoch):
    	for nb in range(agent):
    		loss=agentmodels[nb].fit(sliced_x_train[nb],sliced_y_train[nb], batch_size=batch,validation_split=0.0, epochs=1,verbose=0)
        for nb in range(agent):
    		training_score=agentmodels[nb].evaluate(x_train,y_train,verbose=0,batch_size=512)
        	validation_score=agentmodels[nb].evaluate(x_test,y_test,verbose=0,batch_size=512)
        	training_loss[nb]=training_score[0]
        	training_acc[nb]=training_score[1]
        	validation_loss[nb]=validation_score[0]
        	validation_acc[nb]=validation_score[1]

		train_losses.append(np.average(training_loss))
        val_losses.append(np.average(validation_loss))
        train_accs.append(np.average(training_acc))
        val_accs.append(np.average(validation_acc))

        for nb in range(agent):
		    agent_training_loss_history[nb].append(training_loss[nb])
		    agent_validation_loss_history[nb].append(validation_loss[nb])
		    agent_training_acc_history[nb].append(training_acc[nb])
		    agent_validation_acc_history[nb].append(validation_acc[nb])

        print("->Epoch",(e+1),"is completed with following metrics:,loss:",np.average(training_loss),"accuracy:",np.average(training_acc),"val_loss",np.average(validation_loss),"val_acc",np.average(validation_acc))

        update_parameters_cdsgd(agentmodels,agent)
        print("->Agents share their information!")
        print("->Save experimental log...")
        d_log = {}
        Agent_log={}

        d_log["CDSGD"] = "CDSGD"+'_'+str(agent)+'Agents'
        for nb in range(agent):
            Agent_log["Agent%s training loss"%nb]=agent_training_loss_history[nb]
            Agent_log["Agent%s validation loss"%nb]=agent_validation_loss_history[nb]
            Agent_log["Agent%s training acc"%nb]=agent_training_acc_history[nb]
            Agent_log["Agent%s validation acc"%nb]=agent_validation_acc_history[nb]
        
        d_log["optimizer"] = opt[0].get_config()
        json_string = json.loads(agentmodels[0].to_json())

        print("->Adding model architecture...")
        for key in json_string.keys():
            d_log[key] = json_string[key]
        json_file = os.path.join("log", '%s_%s_%s_%sAgents.json' % (dataset, agentmodels[0].name, experiment_name,str(n_agents)))
        json_file1 = os.path.join("log", '%s_%s_%s_%sAgents_history.json' % (dataset, agentmodels[0].name, experiment_name,str(n_agents)))
        with open(json_file1, 'w') as fp1:
            json.dump(Agent_log, fp1, indent=4, sort_keys=True)
        
        with open(json_file, 'w') as fp:
            json.dump(d_log, fp, indent=4, sort_keys=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is the simulation of CDSGD optimization")
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--epoch', type=int, default=30, help='Number of epochs')
    parser.add_argument('--agent', type=int, default=5, help='Number of agents')
    args = parser.parse_args()
    print("")
    print("We will train our CNN model using CDSGD on cifar10 dataset assuming all agents are fully connected")
    print("Batch Size:" + str(args.batch))
    print("Number of Epoch: "+str(args.epoch))
    print("Number of Agents: "+str(args.agent))
    print("")
    train(args.batch,args.epoch, args.agent)