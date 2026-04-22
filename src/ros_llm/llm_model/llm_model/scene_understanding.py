#!/usr/bin/env python3
# llm_decision_node.py
import json, os, time, base64, io
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge
import cv2
import openai  # will point to Groq via BASE url
from dotenv import load_dotenv
from ultralytics_ros.msg import YoloResult 
from std_msgs.msg import String
from std_srvs.srv import Trigger

# ───────────────────── fixed context ──────────────────────
KINEMATICS = """
Robot base: diff-drive
max_lin_vel: 0.50 m/s
max_ang_vel: 0.10 rad/s
footprint: width 0.80 m × length 1.45 m × height 1.80 m
"""

MAP_PRIORS = """
Indoor office lobby.
Static obstacles: chairs, plants.
Dynamic obstacles: humans.
Robot direction of motion: straight.
"""

SYSTEM_PROMPT = f"""You are the high-level navigation & interaction brain.

{KINEMATICS}
{MAP_PRIORS}

Return ONLY JSON:
{{ "action": "<stop|yield|proceed>", "speech": "<one-line utterance>" }}
"""

USER_TEMPLATE = """### Person detections (YOLO JSON)
{detections_json}

[Image attached below]
"""

# ───────────────────── node class ─────────────────────────
class LlmDecisionNode(Node):
    def __init__(self):
        super().__init__("llm_decision_node")

        # load Groq creds from env
        load_dotenv()                      # useful if you keep keys in .env
        openai.api_key  = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1")
        self.model_id   = os.getenv(
            "MODEL_ID", "meta-llama/llama-4-scout-17b-16e-instruct"
        )
                # subs / pubs
        self.bridge = CvBridge()
        self.latest_image_bgr = None
        self.create_subscription(Image, "/image_raw",
                                 self.image_cb, 10)
        self.create_subscription(YoloResult, "/yolo_result",
                                 self.dets_cb, 10)

        self.action_pub = self.create_publisher(String, "/llm_action", 10)
        self.speech_pub = self.create_publisher(String, "/llm_speech", 10)

        # Expose a ROS2 service so the BT can trigger the LLM on demand:
        self.srv = self.create_service(
            Trigger,
            'run_llm',
            self.sub
        )

        self.get_logger().info("LLM decision node up – using Groq Scout.")

    # callback: stash most recent image
    def image_cb(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image_bgr = cv_img
        except Exception as e:
            self.get_logger().warning(f"cv_bridge error: {e}")

    # callback: run LLM when a detection JSON arrives
    # ---------------------------------------------------------------------------
    # --------------------------------------------------------------
    #  callback: /yolo_result  (YoloResult msg with Detection2DArray)
    # --------------------------------------------------------------
    def dets_cb(self, msg):
        if self.latest_image_bgr is None:
            self.latest_yolo_msg = msg
            self.get_logger().warning("No image yet, skipping detection message.")
            return

        # 1. Detection2DArray  →  list[dict]   (all are persons)
        det_list = []
        for det in msg.detections.detections:
            if not det.results:
                continue
            hyp = det.results[0]
        
            cx, cy = det.bbox.center.position.x, det.bbox.center.position.y
            w,  h  = det.bbox.size_x, det.bbox.size_y
            x_min, y_min = cx - w / 2, cy - h / 2
            x_max, y_max = cx + w / 2, cy + h / 2
        
            score_val = getattr(hyp, "score",
                         getattr(hyp.hypothesis, "score", 0.0))
        
            det_list.append(
                dict(bbox=[x_min, y_min, x_max, y_max],
                     label="person",
                     score=round(float(score_val), 3))
            )


        det_json = json.dumps(det_list)
        user_txt = USER_TEMPLATE.format(detections_json=det_json)

        # 2. Current image  →  base-64 PNG
        _, buf = cv2.imencode(".png", self.latest_image_bgr)
        img_b64 = base64.b64encode(buf).decode()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": [
                 {"type": "text", "text": user_txt},
                 {"type": "image_url",
                  "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]}
        ]

        # 3. Groq Scout call
        t0 = time.time()
        try:
            resp = openai.ChatCompletion.create(
                model      = self.model_id,
                messages   = messages,
                max_tokens = 128,
                stream     = False
            )
        except Exception as e:
            self.get_logger().error(f"Groq API error: {e}")
            return
        latency_ms = (time.time() - t0) * 1000

        # 4. Parse JSON reply
        try:
            js = json.loads(resp.choices[0].message.content.strip())
            action = js.get("action", "")
            speech = js.get("speech", "")
        except json.JSONDecodeError:
            self.get_logger().warning("LLM returned non-JSON; ignoring.")
            return

        self.get_logger().info(f"LLM reply ({latency_ms:.0f} ms) → {action} | {speech}")

        # 5. Publish
        self.action_pub.publish(String(data=action))
        self.speech_pub.publish(String(data=speech))

    
    def call_llm_once(self):
        """
        Pulls the last image & YoloResult, runs the Groq/OpenAI call,
        parses JSON, returns (action, speech).
        """
        # 0) Preconditions
        if self.latest_image_bgr is None:
            raise RuntimeError("No image available for LLM call")
        if not hasattr(self, 'latest_yolo_msg') or self.latest_yolo_msg is None:
            raise RuntimeError("No detections available for LLM call")

        # 1) Build det_list from stored YoloResult
        det_list = []
        for det in self.latest_yolo_msg.detections.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            cx, cy = det.bbox.center.position.x, det.bbox.center.position.y
            w, h  = det.bbox.size_x, det.bbox.size_y
            x_min, y_min = cx - w/2, cy - h/2
            x_max, y_max = cx + w/2, cy + h/2
            score_val = getattr(hyp, "score",
                         getattr(hyp.hypothesis, "score", 0.0))
            det_list.append({
                "bbox":[x_min, y_min, x_max, y_max],
                "label":"person",
                "score":round(float(score_val), 3)
            })
        det_json = json.dumps(det_list)
        user_txt = USER_TEMPLATE.format(detections_json=det_json)

        # 2) Encode image as b64 PNG
        _, buf = cv2.imencode(".png", self.latest_image_bgr)
        img_b64 = base64.b64encode(buf).decode()

        # 3) Assemble messages
        messages = [
            {"role":"system", "content":SYSTEM_PROMPT},
            {"role":"user", "content":[
                {"type":"text", "text":user_txt},
                {"type":"image_url",
                 "image_url":{"url":f"data:image/png;base64,{img_b64}"}}
            ]}
        ]

        # 4) Call the LLM
        t0 = time.time()
        try:
            resp = openai.ChatCompletion.create(
                model      = self.model_id,
                messages   = messages,
                max_tokens = 128,
                stream     = False
            )
        except Exception as e:
            raise RuntimeError(f"Groq API error: {e}")
        latency_ms = (time.time() - t0)*1000

        # 5) Parse JSON reply
        try:
            js = json.loads(resp.choices[0].message.content.strip())
            action = js.get("action", "")
            speech = js.get("speech", "")
        except json.JSONDecodeError:
            raise RuntimeError("LLM returned invalid JSON")

        self.get_logger().info(
            f"LLM reply ({latency_ms:.0f} ms) → {action} | {speech}"
        )
        return action, speech
    
    
    
    def handle_run_llm(self, request, response):
        """
        ROS2 Service callback: run the LLM once,
        publish /llm_action & /llm_speech, then
        fill response.success/message.
        """
        try:
            action, speech = self.call_llm_once()
            # re‐publish exactly as before
            self.action_pub.publish(String(data=action))
            self.speech_pub.publish(String(data=speech))
            response.success = True
            response.message = action
        except Exception as e:
            self.get_logger().error(f"LLM service error: {e}")
            response.success = False
            response.message = str(e)
        return response
    


# ───────────────────── main ───────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = LlmDecisionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
