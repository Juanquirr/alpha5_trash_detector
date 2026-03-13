import json
import uuid
import urllib.request, urllib.parse
import websocket
import io
from PIL import Image
from typing import Any, Dict, List

from core.dependencies.ai.generative_ai.image_generators.image_generator import ImageGenerator

class ComfyuiImageGenerator(ImageGenerator):
    server_address : str = "comfyui.autoescuelaseco.cloud"
    client_id : str = str(uuid.uuid4())
    workflow: Any

    def _find_ws_node(self, workflow: Dict) -> str:
        for node_id, node in workflow.items():
            if node.get("class_type") == "SaveImageWebsocket":
                return node_id
        raise ValueError("SaveImageWebsocket node not found")

    def queue_prompt(self, prompt: Dict) -> str:
        data = json.dumps({"prompt": prompt, "client_id": self.client_id}).encode("utf-8")
        req = urllib.request.Request(f"https://{self.server_address}/prompt", data=data, headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())["prompt_id"]

    def _get_history_images(self, prompt_id: str) -> Dict[str, List[bytes]]:
        output = {}
        hist = json.loads(urllib.request.urlopen(f"https://{self.server_address}/history/{prompt_id}").read())[prompt_id]
        for node_id, node_out in hist.get("outputs", {}).items():
            imgs = []
            for img in node_out.get("images", []):
                params = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
                data = urllib.request.urlopen(f"https://{self.server_address}/view?{params}").read()
                imgs.append(data)
            if imgs:
                output[node_id] = imgs
        return output

    def get_images(self, prompt: Dict) -> Dict[str, List[bytes]]:
        prompt_id = self.queue_prompt(prompt)
        ws_node = self._find_ws_node(prompt)

        ws = websocket.WebSocket()
        ws.connect(f"wss://{self.server_address}/ws?clientId={self.client_id}")

        output_images: Dict[str, List[bytes]] = {}
        current = None

        try:
            while True:
                msg = ws.recv()
                if isinstance(msg, str):
                    msg = json.loads(msg)
                    data = msg.get("data", {})
                    if msg.get("type") == "executing" and data.get("prompt_id") == prompt_id:
                        if data["node"] is None:
                            break
                        current = data["node"]
                else:
                    if current == ws_node:
                        imgs = output_images.setdefault(current, [])
                        imgs.append(msg[8:])
        except websocket.WebSocketTimeoutException:
            pass
        finally:
            ws.close()

        # Fallback via history if none collected
        if ws_node not in output_images:
            output_images = self._get_history_images(prompt_id)

        return output_images

    def generate(self, prompt, negative_prompt) -> Image.Image:
        full_prompt = self.workflow(prompt, negative_prompt)
        imgs_data = self.get_images(full_prompt)
        imgs = []
        for lst in imgs_data.values():
            for b in lst:
                imgs.append(Image.open(io.BytesIO(b)))
        return imgs[0]
