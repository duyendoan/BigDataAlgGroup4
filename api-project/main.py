import flask
from flask import request, jsonify
from moviecontroller import *

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = flask.Flask(__name__)


@app.route("/", methods=["GET"])
def mainpage():
    """ Return a friendly HTTP greeting. """
    return "Hello World!\n"

@app.route('/movies', methods=['GET'])
def api_id():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'movieId' in request.args:
        movieId = int(request.args['movieId'])
    else:
        return "Error: No movieId field provided. Please specify an movieId."
        
    if 'imdbId' in request.args:
        imdbId = int(request.args['imdbId'])
    else:
        return "Error: No imdbId field provided. Please specify an imdbId."

    results = get_movies_by_movieId_userId(movieId, imdbId)

    return jsonify(results)


# if __name__ == "__main__":
    # Used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # app.run(host="localhost", port=8080, debug=True)