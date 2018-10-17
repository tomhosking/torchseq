from absl import app
from args import FLAGS as FLAGS

import os, json

def main(_):
    with open(FLAGS.data_path+'/squad/train-v1.1.json') as fp:
        data = json.load(fp)
    print(data.keys())

if __name__ == '__main__':
  app.run(main)
