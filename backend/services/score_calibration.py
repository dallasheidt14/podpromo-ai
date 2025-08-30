import json, os, math
from collections import deque
from statistics import mean, pstdev

CALIB_PATH = os.path.join("metrics","score_calib.json")

class ScoreCalibrator:
    """
    Maintains rolling distribution of raw scores (0..1) and maps to display 0..100.
    Uses robust percentiles (P10,P90) fallback to mean/std.
    """
    def __init__(self, max_points=500):
        self.max_points = max_points
        self.samples = deque([], maxlen=max_points)
        self._load()

    def _load(self):
        if os.path.exists(CALIB_PATH):
            try:
                with open(CALIB_PATH,"r",encoding="utf-8") as f:
                    data = json.load(f)
                self.samples = deque(data.get("samples",[]), maxlen=self.max_points)
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
        with open(CALIB_PATH,"w",encoding="utf-8") as f:
            json.dump({"samples": list(self.samples)}, f)

    def observe(self, raw_score: float):
        if 0.0 <= raw_score <= 1.0:
            self.samples.append(float(raw_score))
            self._save()

    def _percentiles(self):
        arr = sorted(self.samples) if self.samples else []
        if not arr:
            return 0.3, 0.7
        def p(q):
            i = max(0, min(len(arr)-1, int(round(q*(len(arr)-1)))))
            return arr[i]
        return p(0.10), p(0.90)

    def to_display(self, raw_score: float) -> int:
        """
        Map raw 0..1 -> calibrated 0..100 with conservative top-end:
        - P10 ~ 35
        - Median ~ 60
        - P90 ~ 80
        - >P90 compresses into 80..92 (very few 93+ unless raw ~1.0)
        """
        if not self.samples:
            return int(round(raw_score * 100))

        p10, p90 = self._percentiles()
        if p90 <= p10 + 1e-6:
            m = mean(self.samples); sd = pstdev(self.samples) or 0.1
            z = (raw_score - m) / sd
            disp = 50 + 16 * max(-3, min(3, z))  # narrower stretch
            return int(max(0, min(100, round(disp))))

        if raw_score <= p10:
            x = 0.0 if p10 == 0 else raw_score / p10
            disp = 35 * x
        elif raw_score >= p90:
            # compress tail into 80..92; only extreme raw gets >92
            x = (raw_score - p90) / max(1e-6, 1.0 - p90)
            disp = 80 + 12 * min(1.0, x)
        else:
            x = (raw_score - p10) / (p90 - p10)
            disp = 35 + 45 * x  # midrange = 35..80
        return int(max(0, min(100, round(disp))))
