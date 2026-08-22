from http.server import BaseHTTPRequestHandler, HTTPServer

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(f'IP: {self.client_address[0]}, URL: {self.path}')
        self.send_response(200)
        self.end_headers()

HTTPServer(('0.0.0.0', 80), RequestHandler).serve_forever()