from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from greenpatch.tracker import Rect
from greenpatch.config import RepairConfig

console = Console()


@dataclass(frozen=True)
class Selection:
    target_rect: Rect
    source_rect: Rect
    target_mask: np.ndarray
    source_mask: np.ndarray


class FirstFrameSelector:
    def __init__(self, frame, config: RepairConfig) -> None:
        self.frame = frame
        self.config = config
        self.display = frame.copy()
        self.target_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        self.source_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        self.target_rect: Rect | None = None
        self.source_rect: Rect | None = None
        self.drawing_target = False
        self.drawing_source = False
        self.start_point = None
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.panning = False
        self.pan_start = None
        self.finished = False
        self.quit = False

    def run(self) -> Selection | None:
        cv2.namedWindow("greenpatch", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("greenpatch", self._mouse_callback)
        while True:
            vis = self._render()
            cv2.imshow("greenpatch", vis)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                self.quit = True
                break
            if key == 32:
                if self.target_rect is not None and self.source_rect is not None:
                    self.finished = True
                    break
            if key == ord("r"):
                self.target_mask[:] = 0
                self.source_mask[:] = 0
                self.target_rect = None
                self.source_rect = None
            if key == ord("q"):
                self.quit = True
                break
        cv2.destroyWindow("greenpatch")
        if self.quit:
            return None
        return Selection(
            target_rect=self.target_rect,
            source_rect=self.source_rect,
            target_mask=self.target_mask,
            source_mask=self.source_mask,
        )

    def _mouse_callback(self, event, x, y, flags, userdata):
        real_x = int((x - self.offset_x) / self.zoom)
        real_y = int((y - self.offset_y) / self.zoom)
        if event == cv2.EVENT_MBUTTONDOWN:
            self.panning = True
            self.pan_start = (x, y)
        elif event == cv2.EVENT_MBUTTONUP:
            self.panning = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom = max(0.25, min(8.0, self.zoom - (flags > 0) * 0.25 + (flags < 0) * 0.25))
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_target = True
            self.start_point = (real_x, real_y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing_target:
            self.drawing_target = False
            x1, y1 = self.start_point
            x2, y2 = real_x, real_y
            rect = Rect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            if rect.width >= 4 and rect.height >= 4:
                self.target_rect = rect
                self.target_mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
                cv2.rectangle(self.target_mask, (int(rect.x), int(rect.y)), (int(rect.x + rect.width), int(rect.y + rect.height)), 255, -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing_source = True
            self.start_point = (real_x, real_y)
        elif event == cv2.EVENT_RBUTTONUP and self.drawing_source:
            self.drawing_source = False
            x1, y1 = self.start_point
            x2, y2 = real_x, real_y
            rect = Rect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            if rect.width >= 4 and rect.height >= 4:
                self.source_rect = rect
                self.source_mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
                cv2.rectangle(self.source_mask, (int(rect.x), int(rect.y)), (int(rect.x + rect.width), int(rect.y + rect.height)), 255, -1)

    def _rect_coords(self, rect: Rect):
        return int(rect.x), int(rect.y), int(rect.width), int(rect.height)

    def _render(self):
        vis = self.frame.copy()
        if self.target_rect is not None:
            x, y, w, h = self._rect_coords(self.target_rect)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 180), 2)
        if self.source_rect is not None:
            x, y, w, h = self._rect_coords(self.source_rect)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (180, 0, 0), 2)
        if self.zoom != 1.0:
            vis = cv2.resize(vis, None, fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_NEAREST)
        status = f"Zoom: {self.zoom:.2f}x | Target: {self.target_rect is not None} | Source: {self.source_rect is not None}"
        vis = cv2.putText(vis, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return vis
