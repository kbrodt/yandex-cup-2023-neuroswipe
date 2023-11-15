import typing as tp
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Hitbox:
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0
    cx: int = field(init=False)
    cy: int = field(init=False)

    def __post_init__(self):
        self.cx, self.cy = (self.left + self.right) / 2, (self.top + self.bottom) / 2

    def is_in(self, x, y):
        return self.left <= x < self.right and self.top <= y < self.bottom

    @property
    def central_coords(self):
        return [self.cx, self.cy]


class Grid:
    def __init__(self, grid_info: tp.Dict):
        self.hitboxes = self._build_hitboxes(grid_info)
        self.w = grid_info["width"]
        self.h = grid_info["height"]
        self.wh = max(self.w, self.h)

    def get_the_nearest_hitbox(self, x, y) -> str:
        for label, hitbox in self.hitboxes.items():
            if hitbox.is_in(x, y):
                return label
        return ''

    def get_word(self, xy):
        word = []
        for x, y in xy:
            c = self.get_the_nearest_hitbox(x, y)
            if c == ".":
                break

            word.append(c)

        word = "".join(word)

        return word

    def get_centered_curve(self, word: str) -> np.ndarray:
        curve = []
        for l in word:
            if l not in self.hitboxes:
                continue
            curve.append(self.hitboxes[l].central_coords)

        return np.array(curve, dtype=np.float32)

    @staticmethod
    def _build_hitboxes(grid_info: tp.Dict) -> tp.Dict[str, Hitbox]:
        def hitbox_from_key(key) -> tp.Optional[tp.Tuple[str, Hitbox]]:
            h_name = key.get('label') or key.get('action')
            if h_name is None or len(h_name) > 1:
                return None, None

            h = key['hitbox']
            x, y, w, h = h['x'], h['y'], h['w'], h['h']

            return h_name, Hitbox(top=y, bottom=y + h, left=x, right=x + w)

        hitboxes = {h_name: hitbox for (h_name, hitbox) in map(hitbox_from_key, grid_info['keys']) if h_name}
        return hitboxes

    def normalize(self, xy):
        image_dims = [self.w - 1, self.h - 1]
        xy = 2 * xy / image_dims - 1

        return xy
