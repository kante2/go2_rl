"""Comparison dashboard: same telemetry server as web_dashboard.py, v2 assets."""
from pathlib import Path

import web_dashboard

ASSETS = {'/': ('dashboard_v2.html', 'text/html; charset=utf-8'),
          '/dashboard_v2.js': ('dashboard_v2.js', 'text/javascript; charset=utf-8'),
          '/dashboard_v2.css': ('dashboard_v2.css', 'text/css; charset=utf-8')}


class Dashboard(web_dashboard.Dashboard):
    def __init__(self, host='127.0.0.1', port=8080):
        super().__init__(host, port)
        base = self.server.RequestHandlerClass

        # Reuse the original handler (API, headers, CSP) and swap only the page assets.
        class Handler(base):
            def do_GET(self):
                path = self.path.split('?', 1)[0]
                if path not in ASSETS:
                    return super().do_GET()
                name, content_type = ASSETS[path]
                body = (Path(__file__).parent/name).read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self'; object-src 'none'; frame-ancestors 'none'")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server.RequestHandlerClass = Handler


if __name__ == '__main__':
    # The original demo builds its Dashboard from the module global.
    web_dashboard.Dashboard = Dashboard
    web_dashboard.main()
