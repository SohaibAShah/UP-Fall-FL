import pickle
train_data = pickle.load(open('dataset/Train.pkl', 'rb'))
test_data = pickle.load(open('dataset/Test.pkl', 'rb'))
print(f"Training combinations: {len(train_data)}")  # Should print 329
print(f"Test combinations: {len(test_data)}")      # Should print 164
# Check structure of one sample
key, data = list(train_data.items())[0]
print(f"Sample key: {key}, Sensor data shapes: {[d.shape for d in data[:-1]]}, Label: {data[-1]}")
# Expected: Sample key: (1, 1, 1), Sensor data shapes: [(3, 140, 140), ... (5 times)], Label: 0