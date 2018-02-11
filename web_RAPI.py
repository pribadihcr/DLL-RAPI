# import the necessary packages
from PIL import Image
import numpy as np
import settings
import helpers
import flask
import redis
import uuid
import time
import json
import io
import tempfile

UPLOAD_FOLDER = './data/tmp/'

# initialize our Flask application and Redis server
app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)

@app.route("/")
def homepage():
	return "Welcome to the Deep Learning Libraries based REST API!"

@app.route("/predict_bibnumber", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format and prepare it for
			# detection
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			rfid  = flask.request.files["rfid"]
			tempFileRFIDPath = tempfile.NamedTemporaryFile().name + '.csv'
			rfid.save(tempFileRFIDPath)
			# ensure our NumPy array is C-contiguous as well,
			# otherwise we won't be able to serialize it
			image = np.array(image)
			print(image.shape, image.dtype)
			image = np.expand_dims(image, axis=0)
                        image = image.copy(order='C')

			# generate an ID for the classification then add the
			# classification ID + image to the queue
			k = str(uuid.uuid4())
			image = helpers.base64_encode_image(image)
			print(len(image))
			d = {"id": k, "image": image}
			db.rpush(settings.IMAGE_QUEUE, json.dumps(d))

			# keep looping until our model server returns the output
			# predictions
			while True:
				# attempt to grab the output predictions
				output = db.get(k)

				# check to see if our model has classified the input
				# image
				if output is not None:
					# add the output predictions to our data
					# dictionary so we can return it to the client
					output = output.decode("utf-8")
					data["predictions"] = json.loads(output)

					# delete the result from the database and break
					# from the polling loop
					db.delete(k)
					break

				# sleep for a small amount to give the model a chance
				# to classify the input image
				time.sleep(settings.CLIENT_SLEEP)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# for debugging purposes, it's helpful to start the Flask testing
# server (don't use this for production
if __name__ == "__main__":
	if not os.path.exists(UPLOAD_FOLDER):
        	os.makedirs(UPLOAD_FOLDER)

	print("* Starting web service...")
	#app.run()
	port = int(os.environ.get("PORT", 5000))
    	app.run(host='0.0.0.0', port=port)
