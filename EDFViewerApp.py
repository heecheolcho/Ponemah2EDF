"""
EDFViewerApp.py — native desktop wrapper for the EDF Viewer (EP analysis tool).
Loads the bundled EDF_Viewer.html in a native OS webview window (WebView2 on
Windows, WKWebView on macOS) via pywebview. No browser tabs/address bar; appears
as a standalone application.
"""
import os
import sys
import webview

def resource_path(name):
    # When frozen by PyInstaller, data files live in sys._MEIPASS.
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)

def main():
    html = resource_path("EDF_Viewer.html")
    webview.create_window(
        "EDF Viewer — EP Analysis",
        url=html,
        width=1360, height=880, min_size=(900, 600),
    )
    webview.start()   # blocks until the window is closed

if __name__ == "__main__":
    main()
