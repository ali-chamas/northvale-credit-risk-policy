"""Capture README screenshots of the running Streamlit app with headless Chrome.

Usage (with the app running: `make app` in another terminal):
    .venv/bin/python scripts/screenshot_app.py [--url http://localhost:8501]

Writes reports/figures/app_policy.png and reports/figures/app_applicant.png.
Drives Chrome through the DevTools protocol so it can wait for Streamlit to render
and click a tab. Needs Google Chrome; uses a throwaway profile in a temp folder.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from tornado.websocket import websocket_connect

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = Path(__file__).resolve().parents[1] / "reports" / "figures"
PORT = 9333


class Tab:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method: str, **params):
        self.n += 1
        await self.ws.write_message(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.read_message())
            if msg.get("id") == self.n:
                return msg.get("result", {})

    async def js(self, expr: str):
        return await self.call("Runtime.evaluate", expression=expr, awaitPromise=True)

    async def shot(self, path: Path, height: int):
        await self.call("Emulation.setDeviceMetricsOverride", width=1440, height=height, deviceScaleFactor=2, mobile=False)
        await asyncio.sleep(2.5)
        r = await self.call("Page.captureScreenshot", format="png")
        path.write_bytes(base64.b64decode(r["data"]))
        print(f"wrote {path.relative_to(OUT.parents[1])}")


async def run(url: str) -> None:
    targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list").read())
    page = next(t for t in targets if t["type"] == "page")
    tab = Tab(await websocket_connect(page["webSocketDebuggerUrl"], max_message_size=200 * 1024 * 1024))
    await tab.call("Emulation.setDeviceMetricsOverride", width=1440, height=1500, deviceScaleFactor=2, mobile=False)
    await tab.call("Page.navigate", url=url)
    await asyncio.sleep(10)  # Streamlit loads over a websocket after the page load event
    await tab.shot(OUT / "app_policy.png", 1280)
    await tab.js("[...document.querySelectorAll('[role=tab]')].find(b => b.innerText.includes('Applicant')).click()")
    await asyncio.sleep(3)
    await tab.js("window.scrollTo(0, 0)")
    await tab.shot(OUT / "app_applicant.png", 1000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as prof:
        chrome = subprocess.Popen(
            [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
             f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(2)
            asyncio.run(run(args.url))
        finally:
            chrome.terminate()
            chrome.wait()


if __name__ == "__main__":
    main()
