#!/usr/bin/env python3
# llm_decision_node.py

import json, os, time, base64
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from nav_msgs.msg import Path
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Trigger

from cv_bridge import CvBridge
import cv2
import openai  # will point to Groq via BASE url
from dotenv import load_dotenv
from ultralytics_ros.msg import YoloResult 

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

# Extended template: detections + trajectory + velocity
USER_TEMPLATE = """### Person detections (YOLO JSON)
{detections_json}

### Planned trajectory (5 m ahead)
{traj_json}

### Current velocity
{vel_json}

[Image attached below]
"""

class LlmDecisionNode(Node):
    def __init__(self):
        super().__init__("llm_decision_node")

        # ─── groq/openai setup ─────────────────────────────────────────────────
        load_dotenv()
        openai.api_key  = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1")
        self.model_id   = os.getenv("MODEL_ID", "meta-llama/llama-4-scout-17b-16e-instruct")

        # ─── stashers ───────────────────────────────────────────────────────────
        self.bridge             = CvBridge()
        self.latest_image_bgr   = None
        self.latest_yolo_msg    = None
        self.latest_trajectory_ = None      # <-- new
        self.latest_cmd_vel_    = None      # <-- new

        # ─── subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(Image, "/camera/camera/color/image_raw",
                                 self.image_cb, 10)
        self.create_subscription(YoloResult, "/yolo_result",
                                 self.dets_cb, 10)
        # NEW: listen to the 5 m‐ahead path
        self.create_subscription(Path, "/trajectory_ahead",
                                 self.trajectory_cb, 10)
        # NEW: listen to current cmd_vel
        self.create_subscription(TwistStamped, "/moby_hardware_controller/cmd_vel",
                                 self.cmd_vel_cb, 10)

        # ─── publishers & service ───────────────────────────────────────────────
        self.action_pub = self.create_publisher(String, "/llm_action", 10)
        self.speech_pub = self.create_publisher(String, "/llm_speech", 10)
        # expose trigger so BT can call LLM on demand
        self.srv = self.create_service(Trigger, 'run_llm', self.handle_run_llm)

        self.get_logger().info("LLM decision node up – using Groq Scout.")

    # ─── callbacks to stash data ─────────────────────────────────────────────
    def image_cb(self, msg: Image):
        try:
            self.latest_image_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warning(f"cv_bridge error: {e}")

    def dets_cb(self, msg: YoloResult):
        self.latest_yolo_msg = msg

    def trajectory_cb(self, msg: Path):
        self.latest_trajectory_ = msg      # <-- stash the 5 m ahead path

    def cmd_vel_cb(self, msg: TwistStamped):
        self.latest_cmd_vel_ = msg         # <-- stash the current velocity

    # ─── LLM helper ─────────────────────────────────────────────────────────
    def call_llm_once(self):
        # Preconditions
        if self.latest_image_bgr   is None: raise RuntimeError("No image available")
        if self.latest_yolo_msg    is None: raise RuntimeError("No detections available")
        if self.latest_trajectory_ is None: raise RuntimeError("No trajectory available")
        if self.latest_cmd_vel_    is None: raise RuntimeError("No velocity available")

        # 1) Build detection list
        det_list = []
        for det in self.latest_yolo_msg.detections.detections:
            if not det.results: continue
            hyp = det.results[0]
            cx, cy = det.bbox.center.position.x, det.bbox.center.position.y
            w, h   = det.bbox.size_x, det.bbox.size_y
            x_min, y_min = cx - w/2, cy - h/2
            x_max, y_max = cx + w/2, cy + h/2
            score = getattr(hyp, "score", getattr(hyp.hypothesis, "score", 0.0))
            det_list.append({
                "bbox":[x_min, y_min, x_max, y_max],
                "label":"person",
                "score": round(float(score),3)
            })
        det_json = json.dumps(det_list)

        # 2) Serialize trajectory as list of {x,y}
        traj_pts = [
            {'x': p.pose.position.x, 'y': p.pose.position.y}
            for p in self.latest_trajectory_.poses
        ]
        traj_json = json.dumps(traj_pts)

        # 3) Serialize current velocity
        lin = self.latest_cmd_vel_.twist.linear
        ang = self.latest_cmd_vel_.twist.angular
        vel_json = json.dumps({
            'linear':  {'x':lin.x,'y':lin.y,'z':lin.z},
            'angular': {'x':ang.x,'y':ang.y,'z':ang.z},
        })

        # 4) Prepare the prompt
        user_txt = USER_TEMPLATE.format(
            detections_json=det_json,
            traj_json=traj_json,
            vel_json=vel_json
        )
        _, buf = cv2.imencode(".png", self.latest_image_bgr)
        img_b64 = base64.b64encode(buf).decode()
        messages = [
            {"role":"system", "content":SYSTEM_PROMPT},
            {"role":"user",   "content":[
                {"type":"text",      "text":user_txt},
                {"type":"image_url", "image_url":{"url":f"data:image/png;base64,{img_b64}"}}
            ]}
        ]

        # 5) Call the LLM
        t0 = time.time()
        resp = openai.ChatCompletion.create(
            model      = self.model_id,
            messages   = messages,
            max_tokens = 128,
            stream     = False
        )
        latency_ms = (time.time()-t0)*1000

        # 6) Parse JSON reply
        js = json.loads(resp.choices[0].message.content.strip())
        action, speech = js.get("action",""), js.get("speech","")
        self.get_logger().info(f"LLM reply ({latency_ms:.0f} ms) → {action} | {speech}")
        return action, speech

    # ─── ROS2 Service handler ──────────────────────────────────────────────
    def handle_run_llm(self, request, response):
        try:
            action, speech = self.call_llm_once()
            self.action_pub.publish(String(data=action))
            self.speech_pub.publish(String(data=speech))
            response.success = True
            response.message = action
        except Exception as e:
            self.get_logger().error(f"LLM service error: {e}")
            response.success = False
            response.message = str(e)
        return response

def main(args=None):
    rclpy.init(args=args)
    node = LlmDecisionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__=="__main__":
    main()
