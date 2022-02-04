import pandas as pd
import numpy as np
test_data = pd.read_csv("log.csv")

test_data_cp = test_data.copy()

test_data_cp = test_data_cp.drop("Remarks", axis=1)

test_data_cp = test_data_cp.dropna()

def Voltage_clean(data):
    voltage_data = str(data["Voltage"]).split(".")
    if len(voltage_data) > 1:
        res = float(voltage_data[0]+"."+voltage_data[1])
    else:
        res = float(voltage_data[0])
    return res



test_data_cp["Voltage"] = test_data_cp.apply(Voltage_clean, axis=1)

def Current_clean(data):
    current_data = str(data["Current"]).split(".")
    if len(current_data) > 1:
        res = float(current_data[0]+"."+current_data[1])
    else:
        res = float(current_data[0])
    return res



test_data_cp = test_data_cp.drop([2622284])

test_data_cp["Current"] = test_data_cp.apply(Current_clean, axis=1)

def RP_clean(data):
    rp_data = str(data["RealPower"]).split(".")
    if len(rp_data) > 1:
        res = float(rp_data[0]+"."+rp_data[1])
    else:
        res = float(rp_data[0])
    return res


test_data_cp["RealPower"] = test_data_cp.apply(RP_clean, axis=1)


def PF_clean(data):
    pf_data = str(data["PowerFactor"]).split(".")
    if len(pf_data) > 1:
#         print(pf_data)
        res = pf_data[0]+"."+pf_data[1]
    else:
        res = pf_data[0]
    return res


test_data_cp["PowerFactor"] = test_data_cp.apply(PF_clean, axis=1)

test_data_cp["PowerFactor"] = pd.to_numeric(test_data_cp["PowerFactor"], errors='coerce')

def ReactiveP_clean(data):
    rp_data = str(data["Reactivepower"]).split(".")
    if len(rp_data) > 1:
#         print(pf_data)
        res = float(rp_data[0]+"."+rp_data[1])
    else:
        res = float(rp_data[0])
    return res


test_data_cp["Reactivepower"] = test_data_cp.apply(ReactiveP_clean, axis=1)


def Freq_clean(data):
    freq_data = str(data["Frequency"]).split(".")
    if len(freq_data) > 1:
#         print(pf_data)
        res = float(freq_data[0]+"."+freq_data[1])
    else:
        res = float(freq_data[0])
    return res


test_data_cp["Frequency"] = test_data_cp.apply(Freq_clean, axis=1)


test_data_cp = test_data_cp.drop("TimeStamp", axis=1)

X = test_data_cp.iloc[200000:1000000]

from sklearn import preprocessing
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)

X_tansform = quantile_transformer.fit_transform(X)

print(X_tansform)

import tensorflow as tf
#import tensorflow as tf

# model = tf.keras.models.load_model('device_data.h5')
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# open("model.tflite", "wb").write(tflite_model)
interpreter = tf.lite.Interpreter(model_path="model.tflite")
print(interpreter.get_input_details())
print(interpreter.get_output_details())
print(interpreter.get_tensor_details())
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

# Test the model on input data.
input_shape = input_details[0]['shape']
print(input_shape)
#input_data = np.array(X_tansform, dtype=np.float32)
for x in range(0, len(X_tansform)):
    input_data = np.array(X_tansform[x].reshape(1,7), dtype=np.float32)
#input_data = np.expand_dims(X_tansform, axis=0)
#print(input_data.shape)
#input_data = input_data.astype(np.float64)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Output data :" + str(output_data))
    

