from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):

  def do_GET(self):
    self.send_response(200)
    self.send_header('Content-type', 'text/plain')
    self.end_headers()
    self.message = (
        'PhishDetect AI is running! (Streamlit frontend is hosted locally/elsewhere)'
    )
    self.wfile.write(bytes(self.message, 'utf-8'))
    return