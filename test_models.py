import data_config
import NN_models
import numpy as np

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

    #data = data_config.get_data(2003, 2022)
    #np.random.seed(1)

    data = data_config.get_data(2003, 2022)
    np.random.shuffle(data)

    data_X = data[:,:-1]
    data_y = data[:,-1]

    
    test = data_config.get_data(2022,2023)
    test_X = test[:,:-1]
    test_y = test[:,-1]
    #baseline_acc = baseline_by_seed(data_X, data_y)
    
    num_features = len(data_X[0])

    NN_1 = NN_models.very_simple_MLP(num_features)
    NN_2 = NN_models.simple_MLP2(num_features)


    # NN_2.fit(data_X, data_y, epochs=400, batch_size=32) #, validation_split=0.2)

    NN_1.fit(data_X, data_y, epochs=400, batch_size=3)


    # TODO: early stopping
    # TODO: normalize features?

    test_loss, test_accuracy = NN_1.evaluate(test_X, test_y)
    #test_loss, test_accuracy = NN_2.evaluate(data_X, data_y)
    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {test_accuracy:.2f}")
    #print(f"Baseline accuracy: {baseline_acc:.2f}")

test()