# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from PIL import Image

from cli import LatexOCR

img = Image.open("tests/test_files/repo.png")
model = LatexOCR()
print(model(img))
