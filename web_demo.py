#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler,HTTPServer
from os import curdir, sep
import cgi

from keras.models import load_model

from vectorize import load_word2vec, word2vec_sentences
from config import (model_file,
                    minibatch_size,
                    port
)

BASE_DIR = 'www'

class myHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=="/":
            self.path="/index.html"

        try:
            sendReply = False
            if self.path.endswith(".html"):
                mimetype='text/html'
                sendReply = True
            if self.path.endswith(".js"):
                mimetype='application/javascript'
                sendReply = True
            if self.path.endswith(".css"):
                mimetype='text/css'
                sendReply = True

            if sendReply == True:
                path = curdir + sep + BASE_DIR + self.path
                print("Respond {} as {}".format(path, mimetype))
                self.send_response(200)
                self.send_header('Content-type',mimetype)
                self.end_headers()
                with open(path) as f:
                    self.wfile.write(f.read().encode())
            return
        except IOError:
            self.send_error(404,'File Not Found: %s' % self.path)

    #Handler for the POST requests
    def do_POST(self):
        if self.path=="/analyse":
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD':'POST',
                         'CONTENT_TYPE':self.headers['Content-Type'],
            })

            self.send_response(200)
            self.end_headers()

            if 'review' in form:
                sentence = form['review'].value
                data = word2vec_sentences([sentence], self.server.wv, print_stat=False)
                print("Predict ...")
                label = self.server.model.predict(data)[0, 0]
            else:
                label = -1
            self.wfile.write("{:.5f}".format(label).encode())
            return


def main():
    print("Load model {} ...".format(model_file))
    model = load_model(model_file)
    wv = load_word2vec()
    try:
        server = HTTPServer(('', port), myHandler)
        server.wv = wv
        server.model = model
        print("Start http server on port {}".format(port))
        server.serve_forever()
    finally:
        print("Cleanup.")
        server.socket.close()

if __name__ == '__main__':
    main()
