import tensorflow as tf
import data_config

CB_test_data = data_config.get_data(2022, 2022, dataset="Both", engineered=False)

test_data_X = CB_test_data[:,:-1]
test_data_y = CB_test_data[:,-1]

model1 = tf.keras.models.load_model('model_checkpoints/m1_epoch_78_val_acc_0.84.keras')
model2 = tf.keras.models.load_model('model_checkpoints/m2_epoch_25_val_acc_0.84.keras')
model3 = tf.keras.models.load_model('model_checkpoints/m2_epoch_31_val_acc_0.85.keras')

predictions1 = model1.predict(test_data_X)
final_ps = []
for p in predictions1:
    if p >= 0.5:
        final_ps.append(1)
    else:
        final_ps.append(0)
for i, row in enumerate(test_data_X):
    print(row[0], row[24], final_ps[i])

'''
predictions2 = model2.predict(test_data_X)
predicted_classes2 = tf.argmax(predictions2, axis=1)
print(predicted_classes1.numpy())

predictions3 = model3.predict(test_data_X)
predicted_classes3 = tf.argmax(predictions3, axis=1)
print(predicted_classes3.numpy())
'''