import data_config
import NN_models

# TODO: setup baseline model that only chooses the higher seed to win

def test():
    TEAM_DIRECTORY = data_config.establish_team_directory()

    data = data_config.get_data(1985, 2022)
    print(data)
    data_X = data[:,:-1]
    data_Y = data[:,-1]
    
    num_features = len(data_X[0])

    NN_1 = NN_models.simple_MLP(num_features)

    NN_1.fit(data_X, data_Y, epochs=20, batch_size=32) #, validation_split=0.2)
    test_loss, test_accuracy = NN_1.evaluate(data_X, data_Y)
    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {test_accuracy:.2f}")

test()