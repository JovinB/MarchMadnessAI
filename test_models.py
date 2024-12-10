import data_config
import NN_models
import numpy as np
from sklearn.model_selection import train_test_split

def baseline_by_seed(seeds_X, y):
    num_correct = 0

    for i, seed_pair in enumerate(seeds_X):
        if not seed_pair[0] == seed_pair[1]: # if seeds are the same, skip
            if y[i] == 1:
                if seed_pair[0] < seed_pair[1]:
                    num_correct += 1
            elif y[i] == 0:
                if seed_pair[0] > seed_pair[1]:
                    num_correct += 1
    
    accuracy = num_correct / len(y)

    return accuracy

def test():
    TEAM_DIRECTORY = data_config.establish_team_directory()
    np.random.seed(1)

    #baseline_acc = baseline_by_seed(data_X, data_y)
    
    data = data_config.get_data(2015, 2022, dataset="Regular")
    kenpom_data = data_config.get_data(2015, 2022, dataset="KenPom")
    both_data = data_config.get_data(2015, 2022, dataset="Both")

    np.random.shuffle(data)
    np.random.shuffle(kenpom_data)
    np.random.shuffle(both_data)
    
    train_data, test_data = train_test_split(data, test_size=0.1)
    train_data_X = train_data[:,:-1]
    train_data_y = train_data[:,-1]
    test_data_X = test_data[:,:-1]
    test_data_y = test_data[:,-1]

    KP_train_data, KP_test_data = train_test_split(data, test_size=0.1)
    KP_train_data_X = KP_train_data[:,:-1]
    KP_train_data_y = KP_train_data[:,-1]
    KP_test_data_X = KP_test_data[:,:-1]
    KP_test_data_y = KP_test_data[:,-1]

    CB_train_data, CB_test_data = train_test_split(data, test_size=0.1)
    CB_train_data_X = CB_train_data[:,:-1]
    CB_train_data_y = CB_train_data[:,-1]
    CB_test_data_X = CB_test_data[:,:-1]
    CB_test_data_y = CB_test_data[:,-1]
    
    num_features = len(train_data_X[0])
    KP_num_features = len(KP_train_data_X[0])
    CB_num_features = len(CB_train_data_X[0])

    # NN_1 = NN_models.very_simple_MLP(num_features)
    NN_2 = NN_models.simple_MLP(num_features)
    KP_NN_2 = NN_models.simple_MLP(KP_num_features)
    CB_NN_2 = NN_models.simple_MLP(CB_num_features)
    
    history = NN_2.fit(train_data_X, train_data_y, epochs=1000, batch_size=32, validation_data=(test_data_X, test_data_y))
    KP_history = KP_NN_2.fit(KP_train_data_X, KP_train_data_y, epochs=1000, batch_size=32, validation_data=(KP_test_data_X, KP_test_data_y))
    CB_history = CB_NN_2.fit(CB_train_data_X, CB_train_data_y, epochs=1000, batch_size=32, validation_data=(CB_test_data_X, CB_test_data_y))
    
    accuracies = history.history['val_accuracy']
    KP_accuracies = KP_history.history['val_accuracy']
    CB_accuracies = CB_history.history['val_accuracy']

    losses = history.history['val_loss']
    KP_losses = KP_history.history['val_loss']
    CB_losses = CB_history.history['val_loss']
    

    print(f"Test accuracy: {max(accuracies):.2f}, Test loss: {min(losses):.2f}")
    print(f"KP Test accuracy: {max(KP_accuracies):.2f}, KP Test loss: {min(KP_losses):.2f}")
    print(f"CB Test accuracy: {max(CB_accuracies):.2f}, CB Test loss: {min(CB_losses):.2f}")
    
    # print(f"Baseline accuracy: {baseline_acc:.2f}")

    # TODO: early stopping
    # TODO: normalize features?

test()