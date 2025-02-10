from flask import Flask

app = Flask(__name__)

@app.route('/generate', methods=['GET'])
def generate_content():
    print('Hello world')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True ,threaded=False)