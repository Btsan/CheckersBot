import numpy as np
from keras.models import model_from_json

json_file = open('disc.json', 'r')
rl_json = json_file.read()
json_file.close()

model = model_from_json(rl_json)
model.load_weights('disc.h5')
model.compile(optimizer='nadam', loss='mean_squared_error')

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

for i, L in enumerate(model.layers):
	print('\nlayer %d:' %(i))
	weights = L.get_weights()
	for W in weights:
		print(W)