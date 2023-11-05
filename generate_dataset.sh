#! /bin/bash
# @Author: SWHL
# @Contact: liekkaskono@163.com
python -m data.dataset --equations datasets/tiny_origin/math.txt \
                       --images datasets/tiny_origin/test \
                       --out datasets/pkl/test.pkl \
                       --tokenizer datasets/tiny_origin/tokenizer.json