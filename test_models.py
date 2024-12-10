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
    
    data = data_config.get_data(2022, 2022, dataset="Regular")
    kenpom_data = data_config.get_data(2022, 2022, dataset="KenPom")
    both_data = data_config.get_data(2022, 2022, dataset="Both")

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

    combined_train_data, combined_test_data = train_test_split(data, test_size=0.1)
    combined_train_data_X = combined_train_data[:,:-1]
    combined_train_data_y = combined_train_data[:,-1]
    combined_test_data_X = combined_test_data[:,:-1]
    combined_test_data_y = combined_test_data[:,-1]
    
    num_features = len(train_data_X[0])
    KP_num_features = len(KP_train_data_X[0])
    combined_num_features = len(combined_train_data_X[0])

    # NN_1 = NN_models.very_simple_MLP(num_features)
    NN_2 = NN_models.simple_MLP(num_features)
    KP_NN_2 = NN_models.simple_MLP(KP_num_features)
    CB_NN_2 = NN_models.simple_MLP(combined_num_features)
    
    NN_2.fit(train_data_X, train_data_y, epochs=100, batch_size=32) #, validation_split=0.2)
    KP_NN_2.fit(KP_train_data_X, KP_train_data_y, epochs=100, batch_size=32)
    CB_NN_2.fit(combined_train_data_X, combined_train_data_y, epochs=100, batch_size=32)

    # TODO: early stopping
    # TODO: normalize features?

    # test_loss, test_accuracy = NN_2.evaluate(test_data_X, test_data_y)
    # print(f"Test loss: {test_loss:.2f}")
    # print(f"Test accuracy: {test_accuracy:.2f}")
    #print(f"Baseline accuracy: {baseline_acc:.2f}")

test()