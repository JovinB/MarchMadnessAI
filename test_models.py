import data_config
import NN_models
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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
    np.random.seed(42)

    #baseline_acc = baseline_by_seed(data_X, data_y)
    
    # no data before 2003 season
    train_data = data_config.get_data(2003, 2021, dataset="Regular", engineered=True)    
    # KP_train_data = data_config.get_data(2003, 2021, dataset="KenPom")
    CB_train_data = data_config.get_data(2003, 2021, dataset="Both", engineered=True)

    test_data = data_config.get_data(2022, 2022, dataset="Regular", engineered=True)
    # KP_test_data = data_config.get_data(2022, 2022, dataset="KenPom")
    CB_test_data = data_config.get_data(2022, 2022, dataset="Both", engineered=True)

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    np.random.shuffle(CB_train_data)
    np.random.shuffle(CB_test_data)
    
    # train_data, test_data = train_test_split(data, test_size=0.2)
    train_data_X = train_data[:,:-1]
    train_data_y = train_data[:,-1]
    test_data_X = test_data[:,:-1]
    test_data_y = test_data[:,-1]

    # KP_train_data, KP_test_data = train_test_split(data, test_size=0.2)
    # KP_train_data_X = KP_train_data[:,:-1]
    # KP_train_data_y = KP_train_data[:,-1]
    # KP_test_data_X = KP_test_data[:,:-1]
    # KP_test_data_y = KP_test_data[:,-1]

    # CB_train_data, CB_test_data = train_test_split(data, test_size=0.2)
    CB_train_data_X = CB_train_data[:,:-1]
    CB_train_data_y = CB_train_data[:,-1]
    CB_test_data_X = CB_test_data[:,:-1]
    CB_test_data_y = CB_test_data[:,-1]
    
    # num_features = len(train_data_X[0])
    # KP_num_features = len(KP_train_data_X[0])
    CB_num_features = len(CB_train_data_X[0])

    NN_1 = NN_models.simple_MLP(CB_num_features)
    NN_2 = NN_models.longer_MLP(CB_num_features)
    NN_3 = NN_models.longest_MLP(CB_num_features)
    NN_4 = NN_models.GRU(CB_num_features)
    NN_5 = NN_models.longest_MLP_BN(CB_num_features)
    NN_6 = NN_models.longest_MLP_dropout(CB_num_features)

    epochs = 200
    batch_size = 32

    history1 = NN_1.fit(CB_train_data_X, CB_train_data_y, epochs=epochs, batch_size=batch_size, validation_data=(CB_test_data_X, CB_test_data_y))
    history2 = NN_2.fit(CB_train_data_X, CB_train_data_y, epochs=epochs, batch_size=batch_size, validation_data=(CB_test_data_X, CB_test_data_y))
    history3 = NN_3.fit(CB_train_data_X, CB_train_data_y, epochs=epochs, batch_size=batch_size, validation_data=(CB_test_data_X, CB_test_data_y))
    history4 = NN_4.fit(CB_train_data_X, CB_train_data_y, epochs=epochs, batch_size=batch_size, validation_data=(CB_test_data_X, CB_test_data_y))
    history5 = NN_5.fit(CB_train_data_X, CB_train_data_y, epochs=epochs, batch_size=batch_size, validation_data=(CB_test_data_X, CB_test_data_y))
    history6 = NN_6.fit(CB_train_data_X, CB_train_data_y, epochs=epochs, batch_size=batch_size, validation_data=(CB_test_data_X, CB_test_data_y))
    #history7 = NN_7.fit(train_data_X, train_data_y, epochs=500, batch_size=32, validation_data=(test_data_X, test_data_y))
    #history8 = NN_8.fit(tf.reshape(train_data_X, (-1, 1, 28)), train_data_y, epochs=500, batch_size=32, validation_data=(tf.reshape(test_data_X, (-1, 1, 28)), test_data_y))

    # KP_history = KP_NN_2.fit(KP_train_data_X, KP_train_data_y, epochs=2000, batch_size=32, validation_data=(KP_test_data_X, KP_test_data_y))
    # CB_history = CB_NN_2.fit(CB_train_data_X, CB_train_data_y, epochs=2000, batch_size=32, validation_data=(CB_test_data_X, CB_test_data_y))
    
    accuracies1 = history1.history['val_accuracy']
    accuracies2 = history2.history['val_accuracy']
    accuracies3 = history3.history['val_accuracy']
    accuracies4 = history4.history['val_accuracy']
    accuracies5 = history5.history['val_accuracy']
    accuracies6 = history6.history['val_accuracy']
    accuracies7 = [0.72 for _ in range(200)]
    #accuracies8 = history8.history['val_accuracy']

    # KP_accuracies = KP_history.history['val_accuracy']
    # CB_accuracies = CB_history.history['val_accuracy']

    # losses = history.history['val_loss']
    # KP_losses = KP_history.history['val_loss']
    # CB_losses = CB_history.history['val_loss']
    

    print(f"Test Accuracy Model 1: {max(accuracies1):.2f}, at Epoch {np.argmax(accuracies1)}")
    print(f"Test Accuracy Model 2: {max(accuracies2):.2f}, at Epoch {np.argmax(accuracies2)}")
    print(f"Test Accuracy Model 3: {max(accuracies3):.2f}, at Epoch {np.argmax(accuracies3)}")
    print(f"Test Accuracy Model 4: {max(accuracies4):.2f}, at Epoch {np.argmax(accuracies4)}")
    print(f"Test Accuracy Model 5: {max(accuracies5):.2f}, at Epoch {np.argmax(accuracies5)}")
    print(f"Test Accuracy Model 6: {max(accuracies6):.2f}, at Epoch {np.argmax(accuracies6)}")
    #print(f"Test Accuracy Model 7: {max(accuracies7):.2f}, at Epoch {np.argmax(accuracies7)}")
    #print(f"Test Accuracy Model 8: {max(accuracies8):.2f}, at Epoch {np.argmax(accuracies8)}")

    # print(f"KP Test Accuracy: {max(KP_accuracies):.2f}, at Epoch {np.argmax(KP_accuracies)}")
    # print(f"CB Test Accuracy: {max(CB_accuracies):.2f}, at Epoch {np.argmax(CB_accuracies)}")

    # Plot validation accuracy
    epochs = range(1, len(accuracies1) + 1, 5)  # Epoch numbers
    plt.plot(epochs, accuracies1[0::5], 'b-', label='MLP 1')
    plt.plot(epochs, accuracies2[0::5], 'g-', label='MLP 2')
    plt.plot(epochs, accuracies3[0::5], 'r-', label='MLP 3')
    plt.plot(epochs, accuracies4[0::5], 'c-', label='GRU')
    plt.plot(epochs, accuracies5[0::5], 'm-', label='MLP w/ BN')
    plt.plot(epochs, accuracies6[0::5], 'k-', label='MLP w/ Dropout')
    plt.plot(epochs, accuracies7[0::5], 'y-', label='Baseline')
    #plt.plot(epochs, accuracies8[0::10], 'w-', label='M8')

    # plt.plot(epochs, KP_accuracies[9::10], 'g-', label='KenPom')
    # plt.plot(epochs, CB_accuracies[9::10], 'r-', label='Both')

    plt.ylim(0.3, .9)
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # print(f"Baseline accuracy: {baseline_acc:.2f}")

    # TODO: early stopping
    # TODO: normalize features?

test()